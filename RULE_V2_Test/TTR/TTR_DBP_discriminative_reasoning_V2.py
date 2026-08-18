"""
RULE - Training-free discriminative reasoning reranking for DBP15K.

The script first applies the original TTR skip conditions, then selects at most
`max_rerank_entities` source entities for discriminative Qwen reranking.

For top1_correct_ratio:
- 0.0: select only entities whose original Top-1 prediction is wrong.
- 0.2: target 20% original Top-1 correct and 80% original Top-1 wrong.
- 1.0: special mode; randomly sample from the full rerankable pool without
  controlling the correct/wrong ratio.
"""

import argparse
import json
import os
import random
import re
import time
from urllib.parse import unquote

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import NeighborGenerator, evaluate_alignment


# ==================== Argument Parsing ====================

parser = argparse.ArgumentParser(description="Discriminative reasoning TTR for DBP15K")
parser.add_argument("--data_choice", default="DBP15K")
parser.add_argument("--data_split", default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--use_surface", type=int, default=0)
# 【新增-消融】控制是否使用属性名称模态
parser.add_argument("--use_attribute", type=int, default=1, choices=[0, 1])
# 【新增-消融】控制是否使用属性值模态
parser.add_argument("--use_attribute_value", type=int, default=1, choices=[0, 1])
# 【新增-消融】控制是否使用关系模态
parser.add_argument("--use_relation", type=int, default=1, choices=[0, 1])
# 【新增-消融】控制是否使用视觉模态
parser.add_argument("--use_visual", type=int, default=1, choices=[0, 1])
# 【新增-消融】视觉开启时，是否使用视觉补充；关闭时直接使用原图Image Score
parser.add_argument("--use_visual_supplement", type=int, default=1, choices=[0, 1])
# 【新增-消融】控制是否使用候选判别性差异模块
parser.add_argument("--use_discriminative_difference", type=int, default=1, choices=[0, 1])
parser.add_argument("--threshold", type=float, default=0.2)
# 新增：最多允许多少个源实体进入LLM重排
parser.add_argument("--max_rerank_entities", type=int, default=80)
# 新增：控制进入重排实体中Top-1正确实体比例；1.0表示完全随机
parser.add_argument("--top1_correct_ratio", type=float, default=0.5)
# 新增：固定随机抽样，保证实验可复现
parser.add_argument("--seed", type=int, default=42)
# 新增：是否输出每个实体的详细分析
parser.add_argument("--analysis", type=int, default=1, choices=[0, 1])
parser.add_argument("--save_step", type=int, default=10)
args = parser.parse_args()

if args.max_rerank_entities < 0:
    raise ValueError("--max_rerank_entities must be greater than or equal to 0.")

if not 0.0 <= args.top1_correct_ratio <= 1.0:
    raise ValueError("--top1_correct_ratio must be in [0.0, 1.0].")

use_name = bool(args.use_surface)
# 【新增-消融】模态与模块开关
use_attribute = bool(args.use_attribute)
use_attribute_value = bool(args.use_attribute_value)
use_relation = bool(args.use_relation)
use_visual = bool(args.use_visual)
use_visual_supplement = bool(args.use_visual_supplement)
use_discriminative_difference = bool(args.use_discriminative_difference)

# 修改：analysis控制是否打印完整重排分析过程
show_analysis = bool(args.analysis)

setting = f"DNC_{args.eta}" + ("_use_surface" if use_name else "")
cand_file_path = os.path.join("./candidate_json", args.data_choice, f"{setting}_{args.data_split}.json")
data_file_path = os.path.join("./data", args.data_choice, args.data_split)

description_dir = os.path.join(data_file_path, "descriptions")
# 修改：统一描述缓存，不再针对固定ID
description_file = os.path.join(description_dir, "TTR_discriminative_descriptions.json")
os.makedirs(description_dir, exist_ok=True)

ratio_tag = str(args.top1_correct_ratio).replace(".", "p")
output_dir = os.path.join("./result", args.data_choice)
# 【修改-消融】结果文件名加入消融配置，避免覆盖
ablation_tag = (
    f"_attr{args.use_attribute}_value{args.use_attribute_value}_rel{args.use_relation}"
    f"_vis{args.use_visual}_vissup{args.use_visual_supplement}_diff{args.use_discriminative_difference}"
)
output_file = os.path.join(
    output_dir,
    f"discriminative_{setting}_{args.data_split}_max{args.max_rerank_entities}"
    f"_ratio{ratio_tag}_seed{args.seed}{ablation_tag}.json"
)
os.makedirs(output_dir, exist_ok=True)

MLLM_PATH = "/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
IMG_HEIGHT, IMG_WIDTH = 150, 200

ng = NeighborGenerator(cand_file=cand_file_path, data_file_path=data_file_path)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MLLM_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MLLM_PATH)

if os.path.exists(description_file):
    with open(description_file, "r", encoding="utf-8") as f:
        description_cache = json.load(f)
else:
    description_cache = {}


# ==================== Dataset Loading ====================

def get_uri_name(uri):
    return unquote(uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]).replace("_", " ")


def get_property_name(uri):
    name = get_uri_name(uri)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)


def load_entity_ids(path, selected_ids):
    result = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            ent_id = int(parts[0])

            if ent_id in selected_ids:
                result[ent_id] = {
                    "uri": parts[1],
                    "name": get_uri_name(parts[1])
                }

    return result


def load_relation_ids(path):
    result = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) == 2:
                result[int(parts[0])] = get_property_name(parts[1])

    return result


def load_attributes(path, selected_entities):
    result = {ent_id: [] for ent_id in selected_entities}
    uri_to_id = {data["uri"]: ent_id for ent_id, data in selected_entities.items()}
    pattern = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+(.+)\s+\.\s*$')

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.rstrip("\n"))

            if not match:
                continue

            ent_uri, property_uri, value = match.groups()

            if ent_uri not in uri_to_id:
                continue

            if value.startswith('"'):
                value_match = re.match(r'^"(.*)"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?$', value)

                if value_match:
                    value = value_match.group(1)

            # 【修改-消融】属性名称和属性值分开保存
            result[uri_to_id[ent_uri]].append((get_property_name(property_uri), value))

    return result


def load_relations(triple_path, relation_path, selected_entities, all_entity_path):
    result = {ent_id: [] for ent_id in selected_entities}
    relation_names = load_relation_ids(relation_path)
    all_entity_names = {}

    with open(all_entity_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) == 2:
                all_entity_names[int(parts[0])] = get_uri_name(parts[1])

    with open(triple_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) != 3:
                continue

            head, relation, tail = map(int, parts)
            relation = relation_names.get(relation, str(relation))

            if head in selected_entities:
                tail_name = all_entity_names.get(tail, f"entity {tail}")
                result[head].append(f"has relation {relation} to {tail_name}")

            if tail in selected_entities:
                head_name = all_entity_names.get(head, f"entity {head}")
                result[tail].append(f"is the target of relation {relation} from {head_name}")

    return result


def load_current_entity_data(ent_id, candidate_ids):
    source_ids = {ent_id}
    target_ids = set(candidate_ids)

    source_entities = load_entity_ids(os.path.join(data_file_path, "ent_ids_1"), source_ids)
    target_entities = load_entity_ids(os.path.join(data_file_path, "ent_ids_2"), target_ids)

    if args.data_split == "zh_en":
        source_attribute_path = os.path.join(data_file_path, "zh_att_triples.txt")
    elif args.data_split == "ja_en":
        source_attribute_path = os.path.join(data_file_path, "ja_att_triples.txt")
    else:
        source_attribute_path = os.path.join(data_file_path, "fr_att_triples.txt")

    source_attributes = load_attributes(source_attribute_path, source_entities)
    target_attributes = load_attributes(
        os.path.join(data_file_path, "en_att_triples.txt"),
        target_entities
    )

    source_relations = load_relations(
        os.path.join(data_file_path, "triples_1"),
        os.path.join(data_file_path, "rel_ids_1"),
        source_entities,
        os.path.join(data_file_path, "ent_ids_1")
    )

    target_relations = load_relations(
        os.path.join(data_file_path, "triples_2"),
        os.path.join(data_file_path, "rel_ids_2"),
        target_entities,
        os.path.join(data_file_path, "ent_ids_2")
    )

    return (
        source_entities,
        target_entities,
        source_attributes,
        target_attributes,
        source_relations,
        target_relations
    )


# ==================== Qwen Inference ====================

def batch_inference(requests, max_new_tokens=384):
    processor.tokenizer.padding_side = "left"
    messages = []

    for req in requests:
        if "image_prompt" in req:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {
                        "type": "image",
                        "image": req["image"],
                        "resized_height": IMG_HEIGHT,
                        "resized_width": IMG_WIDTH
                    },
                    {"type": "text", "text": req["image_prompt"]}
                ]}
            ])
        else:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req["text_prompt"]}
            ])

    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)]

    responses = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del inputs
    del generated_ids
    del trimmed
    del image_inputs
    del video_inputs
    del messages
    del texts

    torch.cuda.empty_cache()

    return responses


# ==================== Description Construction ====================

def build_initial_description(entity_info, attributes, relations):
    parts = [f"Entity name: {entity_info['name']}."]

    # 【修改-消融】属性名称与属性值可以独立控制
    for property_name, value in attributes:
        if use_attribute and use_attribute_value:
            parts.append(f"Attribute: {property_name}: {value}.")
        elif use_attribute:
            parts.append(f"Attribute: {property_name}.")
        elif use_attribute_value:
            parts.append(f"Attribute value: {value}.")

    # 【修改-消融】只有use_relation=1时才加入关系模态
    if use_relation:
        for relation in relations:
            parts.append(f"Relation: {relation}.")

    words = " ".join(parts).split()

    if len(words) > 128:
        words = words[:128]

    return " ".join(words)


def build_visual_prompt(initial_description):
    return (
        "The following description was created only from knowledge-graph facts:\n\n"
        f"{initial_description}\n\n"
        "Examine the supplied image and find complementary information that helps characterize or distinguish the "
        "entity. First explain what the image directly depicts and whether it shows the entity itself or an indirectly "
        "related portrait, logo, flag, building, product, event, document, or scene. Then state what the image confirms, "
        "supplements, contradicts, or leaves uncertain. Do not perform entity alignment and do not add unsupported facts. "
        "Write one concise paragraph."
    )


# 【新增-消融】视觉补充关闭但视觉模态开启时，参考TTR_DBP.py直接对原图打Image Score
def build_direct_image_score_prompt():
    return (
        "The two provided images represent a query entity and a candidate entity.\n"
        "Please evaluate the probability that they belong to the same real-world entity STEP BY STEP:\n"
        "1. Analyze the similarities of detailed visual contents between the two images.\n"
        "2. Consider whether the images may depict indirectly related content associated with the same entity.\n"
        "[Output Format]: [IMAGE SIMILARITY] = A out of 10, where A is in range [0,1,2,3,4,5,6,7,8,9,10].\n"
        "NOTICE: You MUST output strictly in this format: [IMAGE SIMILARITY] = A out of 10."
    )


def parse_image_score(response):
    match = re.search(r"\[IMAGE SIMILARITY\]\s*=\s*(\d+(?:\.\d+)?)\s*out of 10", response)
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(10.0, score))


# 【新增-消融】直接比较Query/Candidate原图；Image Score只作为最终判别证据，不和hhea_sim融合
def get_direct_image_scores(ent_id, unique_candidates):
    scores = {candidate["ent_id"]: 0.0 for candidate in unique_candidates}

    if not use_visual or use_visual_supplement:
        return scores

    main_image = os.path.join(data_file_path, "concat_images", f"{ent_id}.jpg")
    if not os.path.exists(main_image):
        return scores

    for candidate in unique_candidates:
        candidate_id = candidate["ent_id"]
        candidate_image = os.path.join(data_file_path, "concat_images", f"{candidate_id}.jpg")

        if not os.path.exists(candidate_image):
            continue

        messages = [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": main_image, "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                {"type": "image", "image": candidate_image, "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                {"type": "text", "text": build_direct_image_score_prompt()}
            ]}
        ]]

        prompt = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=64)

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        scores[candidate_id] = parse_image_score(response)

        del inputs, generated_ids, trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()

    return scores


def anonymize_description(text, names):
    result = text

    for name in sorted(set(names), key=len, reverse=True):
        if name:
            result = re.sub(re.escape(name), "[ANONYMOUS ENTITY]", result, flags=re.IGNORECASE)

    return re.sub(r"https?://\S+", "[ANONYMOUS ENTITY]", result)


def save_description_cache():
    with open(description_file, "w", encoding="utf-8") as f:
        json.dump(description_cache, f, ensure_ascii=False, indent=4)


def get_complete_description(ent_id, side, entity_info, attributes, relations, all_names):
    # 【修改-消融】缓存key加入当前模态配置，避免不同消融实验复用错误描述
    cache_key = (
        f"{side}_{ent_id}"
        f"_a{args.use_attribute}_v{args.use_attribute_value}_r{args.use_relation}"
        f"_vis{args.use_visual}_vs{args.use_visual_supplement}"
    )

    if cache_key in description_cache:
        return description_cache[cache_key]

    initial_description = build_initial_description(entity_info, attributes, relations)
    image_path = os.path.join(data_file_path, "concat_images", f"{ent_id}.jpg")

    # 【修改-消融】只有视觉模态和视觉补充模块同时开启时才生成视觉补充
    if use_visual and use_visual_supplement and os.path.exists(image_path):
        visual_supplement = batch_inference([{
            "image": image_path,
            "image_prompt": build_visual_prompt(initial_description)
        }], max_new_tokens=64)[0].strip()
    elif use_visual and use_visual_supplement:
        visual_supplement = "No image is available, so no complementary visual information is used."
    else:
        visual_supplement = ""

    full_description = initial_description
    if visual_supplement:
        full_description += "\nVisual complementary information: " + visual_supplement

    anonymous_description = anonymize_description(full_description, all_names)

    description_cache[cache_key] = {
        "initial_description": initial_description,
        "visual_supplement": visual_supplement,
        "full_description": full_description,
        "anonymous_description": anonymous_description,
        "image_available": os.path.exists(image_path)
    }

    save_description_cache()

    return description_cache[cache_key]


def get_candidate_descriptions_batch(
    unique_candidates,
    target_entities,
    target_attributes,
    target_relations,
    all_names
):
    records = {}
    image_requests = []
    request_ids = []

    for candidate in unique_candidates:
        candidate_id = candidate["ent_id"]
        # 【修改-消融】缓存key加入当前模态配置
        cache_key = (
            f"target_{candidate_id}"
            f"_a{args.use_attribute}_v{args.use_attribute_value}_r{args.use_relation}"
            f"_vis{args.use_visual}_vs{args.use_visual_supplement}"
        )

        if cache_key in description_cache:
            records[candidate_id] = description_cache[cache_key]
            continue

        initial_description = build_initial_description(
            target_entities[candidate_id],
            target_attributes[candidate_id],
            target_relations[candidate_id]
        )

        image_path = os.path.join(
            data_file_path,
            "concat_images",
            f"{candidate_id}.jpg"
        )

        # 【修改-消融】只有use_visual=1且use_visual_supplement=1时才生成视觉补充
        if use_visual and use_visual_supplement and os.path.exists(image_path):
            image_requests.append({
                "image": image_path,
                "image_prompt": build_visual_prompt(initial_description)
            })

            request_ids.append(candidate_id)

            records[candidate_id] = {
                "initial_description": initial_description,
                "image_path": image_path,
                "image_available": True
            }
        else:
            if use_visual and use_visual_supplement:
                visual_supplement = "No image is available, so no complementary visual information is used."
            else:
                visual_supplement = ""

            full_description = initial_description
            if visual_supplement:
                full_description += "\nVisual complementary information: " + visual_supplement

            records[candidate_id] = {
                "initial_description": initial_description,
                "visual_supplement": visual_supplement,
                "full_description": full_description,
                "anonymous_description": anonymize_description(full_description, all_names),
                "image_available": os.path.exists(image_path)
            }

    # if image_requests:
    #     responses = batch_inference(image_requests, max_new_tokens=64)

    if image_requests:
        responses = []

        for candidate_id, request in zip(request_ids, image_requests):
            print(f"Testing candidate image: ID={candidate_id}, image={request['image']}")
            response = batch_inference([request], max_new_tokens=64)[0]
            responses.append(response)

        for candidate_id, response in zip(request_ids, responses):
            visual_supplement = response.strip()
            initial_description = records[candidate_id]["initial_description"]

            full_description = initial_description + "\nVisual complementary information: " + visual_supplement

            records[candidate_id] = {
                "initial_description": initial_description,
                "visual_supplement": visual_supplement,
                "full_description": full_description,
                "anonymous_description": anonymize_description(full_description, all_names),
                "image_available": True
            }

    cache_changed = False

    for candidate_id, record in records.items():
        cache_key = (
            f"target_{candidate_id}"
            f"_a{args.use_attribute}_v{args.use_attribute_value}_r{args.use_relation}"
            f"_vis{args.use_visual}_vs{args.use_visual_supplement}"
        )

        if cache_key not in description_cache:
            description_cache[cache_key] = record
            cache_changed = True

    if cache_changed:
        save_description_cache()

    return records


# ==================== Difference and Alignment Prompts ====================

def build_difference_prompt(candidate_descriptions):
    content = []

    for label, description in candidate_descriptions.items():
        content.append(f"[{label}]\n{description}")

    return (
        "Compare the following ten anonymous candidate entities jointly. For every candidate, identify the information "
        "that distinguishes it from the other candidates, including entity type, attributes, relations, visual evidence, "
        "contradictions, and ambiguities. Do not match them to a source entity yet. Do not infer or output real entity "
        "names.\n\n"
        + "\n\n".join(content)
    )


def build_alignment_prompt(source_description, candidate_descriptions, difference_response):
    content = []

    for label, description in candidate_descriptions.items():
        content.append(f"[{label}]\n{description}")

    name_rule = (
        "Entity names and surface-form similarity must not be considered because all names are anonymized."
        if not use_name
        else
        "Entity names may be considered together with the other evidence."
    )

    # 【修改-消融】关闭Difference模块时，最终Prompt中彻底删除该部分
    difference_part = ""
    if use_discriminative_difference:
        difference_part = "\n\n[CANDIDATE DIFFERENCES]\n" + difference_response

    return (
        "Align the source entity with exactly one of the ten candidate entities. Compare all candidates jointly and "
        "produce a complete ranking with no ties. "
        + name_rule
        + " Do not use the original retrieval similarities. Give greater weight to precise and consistent evidence "
        "than to generic resemblance.\n\n"
        f"[SOURCE ENTITY]\n{source_description}\n\n"
        "[CANDIDATE ENTITIES]\n"
        + "\n\n".join(content)
        + difference_part
        + "\n\nOutput exactly one final line in this format:\n"
        "[RANKING] = CANDIDATE_1 > CANDIDATE_2 > CANDIDATE_3 > CANDIDATE_4 > CANDIDATE_5 > "
        "CANDIDATE_6 > CANDIDATE_7 > CANDIDATE_8 > CANDIDATE_9 > CANDIDATE_10\n"
        "The shown order is only a format example. Include every candidate exactly once."
    )


def parse_ranking(response):
    match = re.search(r"\[RANKING\]\s*=\s*(.+)", response)
    labels = re.findall(r"CANDIDATE_\d+", match.group(1) if match else response)

    ranking = []

    for label in labels:
        if label not in ranking and 1 <= int(label.split("_")[-1]) <= 10:
            ranking.append(label)

    for index in range(1, 11):
        label = f"CANDIDATE_{index}"

        if label not in ranking:
            ranking.append(label)

    return ranking[:10]


# ==================== Rerankable Pool and Sampling ====================

# 新增：复用TTR_DBP.py的skip规则，判断是否允许进入重排
def passes_original_ttr_conditions(ent_id):
    candidates = ng.get_candidates(ent_id)
    base_rank = ng.get_base_rank(ent_id)
    unique_candidates = list({candidate["ent_id"]: candidate for candidate in candidates if isinstance(candidate, dict)}.values())

    if base_rank >= 10:
        return False

    if len(unique_candidates) != 10:
        return False

    ori_scores = sorted([candidate.get("hhea_sim", 0) for candidate in unique_candidates], reverse=True)

    if not ori_scores:
        return False

    if ori_scores[0] >= args.threshold:
        return False

    if len(ori_scores) > 1 and ori_scores[0] - ori_scores[1] > 0.2:
        return False

    return True


# 新增：建立可重排实体池
def build_rerankable_pool():
    rerankable_pool = []

    for ent_id in ng.get_entities():
        if passes_original_ttr_conditions(ent_id):
            rerankable_pool.append(ent_id)

    return rerankable_pool


# 新增：根据max_rerank_entities和top1_correct_ratio抽样
def select_rerank_entities(rerankable_pool):
    rng = random.Random(args.seed)
    maximum = min(args.max_rerank_entities, len(rerankable_pool))

    if maximum == 0:
        return []

    # 特殊模式：完全随机抽样
    if args.top1_correct_ratio == 1.0:
        selected = rng.sample(rerankable_pool, maximum)
        rng.shuffle(selected)
        return selected

    correct_pool = [
        ent_id
        for ent_id in rerankable_pool
        if ng.get_base_rank(ent_id) == 0
    ]

    wrong_pool = [
        ent_id
        for ent_id in rerankable_pool
        if ng.get_base_rank(ent_id) != 0
    ]

    correct_target = round(maximum * args.top1_correct_ratio)
    wrong_target = maximum - correct_target

    selected_correct = rng.sample(correct_pool, min(correct_target, len(correct_pool)))

    selected_wrong = rng.sample(wrong_pool, min(wrong_target, len(wrong_pool)))

    selected = selected_correct + selected_wrong
    selected_set = set(selected)
    shortage = maximum - len(selected)

    if shortage > 0:
        # 某一类数量不足时，从另一类补足
        remaining_pool = [
            ent_id
            for ent_id in rerankable_pool
            if ent_id not in selected_set
        ]

        selected.extend(rng.sample(remaining_pool, min(shortage, len(remaining_pool))))

    rng.shuffle(selected)

    return selected


# ==================== Single Entity Reranking ====================

def run_single_entity(ent_id):
    main_entity = ng.get_main_entity(ent_id)
    candidates = ng.get_candidates(ent_id)
    ref_ent = ng.get_ref_ent(ent_id)
    base_rank = ng.get_base_rank(ent_id)

    unique_candidates = list({candidate["ent_id"]: candidate for candidate in candidates if isinstance(candidate, dict)}.values())

    if len(unique_candidates) != 10:
        raise ValueError(
            f"Entity {ent_id} has {len(unique_candidates)} unique candidates instead of 10."
        )

    if show_analysis:
        print("\n" + "=" * 80)
        print(f"Query ID: {ent_id}")
        print(f"Query name: {main_entity['name']}")
        print(f"Reference ID: {ref_ent}")
        print(f"Base rank: {base_rank} ({base_rank + 1}th position)")
        print(f"Candidates: {[candidate['ent_id'] for candidate in unique_candidates]}")

    start = time.time()
    candidate_ids = [candidate["ent_id"] for candidate in unique_candidates]

    (
        source_entities,
        target_entities,
        source_attributes,
        target_attributes,
        source_relations,
        target_relations
    ) = load_current_entity_data(ent_id, candidate_ids)

    all_names = [source_entities[ent_id]["name"]] + [target_entities[candidate_id]["name"] for candidate_id in candidate_ids]

    source_record = get_complete_description(
        ent_id, "source", source_entities[ent_id], source_attributes[ent_id], source_relations[ent_id], all_names
    )

    if show_analysis:
        print("\nSource initial description:\n" + source_record["initial_description"])
        print("\nSource visual supplement:\n" + source_record["visual_supplement"])

    candidate_records = get_candidate_descriptions_batch(
        unique_candidates, target_entities, target_attributes, target_relations, all_names
    )

    # 【新增-消融】视觉开启但视觉补充关闭时，直接计算Query-Candidate原图Image Score
    direct_image_scores = get_direct_image_scores(ent_id, unique_candidates)

    candidate_descriptions = {}
    label_to_candidate = {}

    for index, candidate in enumerate(unique_candidates, 1):
        candidate_id = candidate["ent_id"]
        label = f"CANDIDATE_{index}"
        record = candidate_records[candidate_id]

        candidate_descriptions[label] = record["full_description"] if use_name else record["anonymous_description"]

        # 【新增-消融】Image Score只作为判别证据，不和原hhea_sim做数值融合
        if use_visual and not use_visual_supplement:
            candidate_descriptions[label] += (
                f"\nDirect visual evidence: Image similarity score = "
                f"{direct_image_scores[candidate_id]:.1f} out of 10."
            )

        label_to_candidate[label] = candidate

        if show_analysis:
            print(
                f"\n{label} ID={candidate_id} initial description:\n"
                + record["initial_description"]
            )
            print(
                f"\n{label} visual supplement:\n"
                + record["visual_supplement"]
            )

    # 【修改-消融】可以完全关闭Candidate Difference模块
    if use_discriminative_difference:
        difference_response = batch_inference([{
            "text_prompt": build_difference_prompt(candidate_descriptions)
        }], max_new_tokens=64)[0].strip()
    else:
        difference_response = ""

    if show_analysis and use_discriminative_difference:
        print(
            "\nCandidate discriminative differences:\n"
            + difference_response
        )

    source_description = source_record["full_description"] if use_name else source_record["anonymous_description"]

    alignment_response = batch_inference([{
        "text_prompt": build_alignment_prompt(source_description, candidate_descriptions, difference_response)
    }])[0].strip()

    ranking_labels = parse_ranking(alignment_response)

    ranked = []

    for rank_index, label in enumerate(ranking_labels):
        ranked.append({
            "label": label,
            "ent_id": label_to_candidate[label]["ent_id"],
            "name": label_to_candidate[label].get("name", ""),
            "ori_score": label_to_candidate[label].get("hhea_sim", 0),
            "unique_score": 9 - rank_index
        })

    llm_rank = next(index for index, item in enumerate(ranked) if item["ent_id"] == ref_ent)

    if show_analysis:
        print("\nFinal alignment response:\n" + alignment_response)
        print("\nDiscriminative reasoning reranking result")

        for rank, item in enumerate(ranked, 1):
            mark = "  <-- ground truth" if item["ent_id"] == ref_ent else ""
            name = f" | {item['name']}" if use_name else ""

            print(
                f"{rank:2d}. {item['label']} | ID={item['ent_id']}{name} "
                f"| ori={item['ori_score']:.3f} "
                f"| unique_score={item['unique_score']}{mark}"
            )

        print(f"\nBase rank: {base_rank} ({base_rank + 1}th position)")
        print(f"LLM rank: {llm_rank} ({llm_rank + 1}th position)")
        print(f"Rank improved: {llm_rank < base_rank}")
        print(f"Hits@1 correct: {llm_rank == 0}")
        print(f"Time cost: {time.time() - start:.2f}s")

    return {
        "base_rank": int(base_rank),
        "llm_rank": int(llm_rank),
        "selected_for_rerank": True,
        "time_cost": time.time() - start,
        "ranking": ranked,
        "candidate_differences": difference_response,
        "alignment_response": alignment_response
    }


# ==================== Metrics and Saving ====================

# 新增：统计Correct→Correct等四类Top-1变化
def calculate_transition_statistics(selected_results):
    counts = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0
    }

    for result in selected_results:
        base_correct = result["base_rank"] == 0
        llm_correct = result["llm_rank"] == 0

        if base_correct and llm_correct:
            counts["correct_to_correct"] += 1
        elif base_correct and not llm_correct:
            counts["correct_to_wrong"] += 1
        elif not base_correct and llm_correct:
            counts["wrong_to_correct"] += 1
        else:
            counts["wrong_to_wrong"] += 1

    total = len(selected_results)
    rates = {key: count / total if total else 0.0 for key, count in counts.items()}

    return counts, rates


def save_result_file(result):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


# ==================== Main Evaluation ====================

def run_evaluation(hit_k=[1, 5, 10]):
    all_entities = ng.get_entities()
    rerankable_pool = build_rerankable_pool()
    selected_entities = select_rerank_entities(rerankable_pool)
    selected_set = set(selected_entities)

    pool_correct = sum(ng.get_base_rank(ent_id) == 0 for ent_id in rerankable_pool)

    selected_correct = sum(ng.get_base_rank(ent_id) == 0 for ent_id in selected_entities)

    print("=" * 80)
    print(f"Total entities in candidate JSON: {len(all_entities)}")
    print(f"Rerankable pool after original skip rules: {len(rerankable_pool)}")
    print(f"Rerankable pool composition: Top-1 correct={pool_correct}, Top-1 wrong={len(rerankable_pool) - pool_correct}")
    print(f"Requested maximum rerank entities: {args.max_rerank_entities}")
    print(f"Selected rerank entities: {len(selected_entities)}")
    print(f"Selected composition: Top-1 correct={selected_correct}, Top-1 wrong={len(selected_entities) - selected_correct}")
    print(f"top1_correct_ratio: {args.top1_correct_ratio}")
    print(f"seed: {args.seed}")
    print(f"analysis: {args.analysis}")
    print(
        f"Ablation: surface={args.use_surface}, attribute={args.use_attribute}, "
        f"attribute_value={args.use_attribute_value}, relation={args.use_relation}, "
        f"visual={args.use_visual}, visual_supplement={args.use_visual_supplement}, "
        f"discriminative_difference={args.use_discriminative_difference}"
    )
    print("=" * 80)

    if len(rerankable_pool) < args.max_rerank_entities:
        print(
            "WARNING: The rerankable pool is smaller than "
            "--max_rerank_entities; all available entities were selected."
        )

    result = {
        "config": {
            "data_choice": args.data_choice,
            "data_split": args.data_split,
            "eta": args.eta,
            "use_surface": args.use_surface,
            "threshold": args.threshold,
            # 【新增-消融】记录本次实验的模态与模块开关
            "use_attribute": args.use_attribute,
            "use_attribute_value": args.use_attribute_value,
            "use_relation": args.use_relation,
            "use_visual": args.use_visual,
            "use_visual_supplement": args.use_visual_supplement,
            "use_discriminative_difference": args.use_discriminative_difference,
            "max_rerank_entities": args.max_rerank_entities,
            "top1_correct_ratio": args.top1_correct_ratio,
            "seed": args.seed,
            "analysis": args.analysis
        },
        "selection": {
            "total_entities": len(all_entities),
            "rerankable_pool_size": len(rerankable_pool),
            "selected_count": len(selected_entities),
            "selected_ids": selected_entities,
            "selected_top1_correct": selected_correct,
            "selected_top1_wrong": len(selected_entities) - selected_correct
        },
        "entities": {},
        "metrics": {}
    }

    base_ranks = []
    llm_ranks = []

    for ent_id in all_entities:
        base_rank = int(ng.get_base_rank(ent_id))

        result["entities"][str(ent_id)] = {
            "base_rank": base_rank,
            "llm_rank": base_rank,
            "selected_for_rerank": ent_id in selected_set,
            "time_cost": 0.0
        }

    selected_results = []

    for index, ent_id in enumerate(tqdm(selected_entities, desc="Discriminative Reranking"), 1):
        entity_result = run_single_entity(ent_id)
        result["entities"][str(ent_id)] = entity_result
        selected_results.append(entity_result)

        if index % args.save_step == 0 or index == len(selected_entities):
            save_result_file(result)

    for ent_id in all_entities:
        entity_result = result["entities"][str(ent_id)]
        base_ranks.append(entity_result["base_rank"])
        llm_ranks.append(entity_result["llm_rank"])

    # 保留：全数据集指标仍采用TTR_DBP.py计算方式
    base_hits, base_mrr = evaluate_alignment(base_ranks, hit_k)
    llm_hits, llm_mrr = evaluate_alignment(llm_ranks, hit_k)

    # 新增：统计进入重排子集的四类变化
    transition_counts, transition_rates = calculate_transition_statistics(selected_results)

    result["metrics"] = {
        "full_dataset": {
            "base": {
                "Hits@1": base_hits[0],
                "Hits@5": base_hits[1],
                "Hits@10": base_hits[2],
                "MRR": base_mrr
            },
            "ttr": {
                "Hits@1": llm_hits[0],
                "Hits@5": llm_hits[1],
                "Hits@10": llm_hits[2],
                "MRR": llm_mrr
            }
        },
        "selected_subset_transitions": {
            "total": len(selected_results),
            "counts": transition_counts,
            "rates": transition_rates
        }
    }

    save_result_file(result)

    print("\n" + "=" * 80)
    print("Full Dataset Metrics")
    print(
        f"Base: Hits@{hit_k}={base_hits}, "
        f"MRR={base_mrr:.6f}"
    )
    print(
        f"TTR:  Hits@{hit_k}={llm_hits}, "
        f"MRR={llm_mrr:.6f}"
    )

    print("\nSelected Rerank Subset Top-1 Transitions")

    labels = [
        ("correct_to_correct", "Correct -> Correct"),
        ("correct_to_wrong", "Correct -> Wrong"),
        ("wrong_to_correct", "Wrong -> Correct"),
        ("wrong_to_wrong", "Wrong -> Wrong")
    ]

    for key, label in labels:
        count = transition_counts[key]
        rate = transition_rates[key]

        print(
            f"{label}: {count}/{len(selected_results)} "
            f"({rate * 100:.2f}%)"
        )

    print(f"\nDescriptions saved to: {description_file}")
    print(f"Results saved to: {output_file}")

    return result


if __name__ == "__main__":
    start = time.time()
    run_evaluation(hit_k=[1, 5, 10])
    print(f"Total time: {time.time() - start:.2f}s")