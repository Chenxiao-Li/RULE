"""Run the batched training-free reranking process for selected DBP15K entities."""

import argparse
import json
import os
import re
import time
from urllib.parse import unquote

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import NeighborGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--data_choice", default="DBP15K")
parser.add_argument("--data_split", default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--use_surface", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--target_ids", nargs="+", type=int, default=[24135, 25148])
args = parser.parse_args()

use_name = bool(args.use_surface)
setting = f"DNC_{args.eta}" + ("_use_surface" if use_name else "")
cand_file_path = os.path.join("./candidate_json", args.data_choice, f"{setting}_{args.data_split}.json")
data_file_path = os.path.join("./data", args.data_choice, args.data_split)
description_dir = os.path.join(data_file_path, "descriptions")
description_file = os.path.join(description_dir, "TTR_descriptions_24135_25148.json")
os.makedirs(description_dir, exist_ok=True)

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


def get_uri_name(uri):
    return unquote(uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]).replace("_", " ")


def get_property_name(uri):
    name = get_uri_name(uri)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)


# ==================== 新增：只读取当前需要的实体ID，不再一次性加载整个数据集 ====================
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


# ==================== 新增：只扫描当前实体涉及的属性三元组 ====================
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

            result[uri_to_id[ent_uri]].append(f"{get_property_name(property_uri)}: {value}")

    return result


# ==================== 新增：只扫描当前实体涉及的一跳关系 ====================
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


def batch_inference(requests, max_new_tokens=384):
    processor.tokenizer.padding_side = "left"
    messages = []

    for req in requests:
        if "image_prompt" in req:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": req["image"], "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                    {"type": "text", "text": req["image_prompt"]}
                ]}
            ])
        else:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req["text_prompt"]}
            ])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    responses = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    del inputs
    del generated_ids
    del trimmed
    del image_inputs
    del video_inputs
    del messages
    del texts

    torch.cuda.empty_cache()

    return responses


# ==================== 修改：不再调用Qwen生成初始描述，直接用Python拼接数据集事实 ====================
def build_initial_description(entity_info, attributes, relations):
    parts = [f"Entity name: {entity_info['name']}."]

    for attribute in attributes:
        parts.append(f"Attribute: {attribute}.")

    for relation in relations:
        parts.append(f"Relation: {relation}.")

    words = " ".join(parts).split()

    if len(words) > 128:
        words = words[:128]

    return " ".join(words)


# ==================== 新增：根据初始描述，让Qwen从图片中寻找补充信息 ====================
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


def anonymize_description(text, names):
    result = text

    for name in sorted(set(names), key=len, reverse=True):
        if name:
            result = re.sub(re.escape(name), "[ANONYMOUS ENTITY]", result, flags=re.IGNORECASE)

    return re.sub(r"https?://\S+", "[ANONYMOUS ENTITY]", result)


# ==================== 新增：生成完整描述（初始描述+图片补充），并缓存到descriptions目录 ====================
def get_complete_description(ent_id, side, entity_info, attributes, relations, all_names):
    cache_key = f"{side}_{ent_id}"

    if cache_key in description_cache:
        return description_cache[cache_key]

    # ==================== 修改：初始描述由Python直接拼接，不再产生一次额外LLM推理 ====================
    initial_description = build_initial_description(entity_info, attributes, relations)
    image_path = os.path.join(data_file_path, "concat_images", f"{ent_id}.jpg")

    if os.path.exists(image_path):
        visual_supplement = batch_inference([{"image": image_path,"image_prompt": build_visual_prompt(initial_description)}], max_new_tokens=64)[0].strip()
    else:
        visual_supplement = "No image is available, so no complementary visual information is used."

    full_description = initial_description + "\nVisual complementary information: " + visual_supplement
    anonymous_description = anonymize_description(full_description, all_names)

    description_cache[cache_key] = {
        "initial_description": initial_description,
        "visual_supplement": visual_supplement,
        "full_description": full_description,
        "anonymous_description": anonymous_description,
        "image_available": os.path.exists(image_path)
    }

    with open(description_file, "w", encoding="utf-8") as f:
        json.dump(description_cache, f, ensure_ascii=False, indent=4)

    return description_cache[cache_key]


# ==================== 新增优化：将10个候选的图片补充请求合并成一个batch ====================
def get_candidate_descriptions_batch(unique_candidates, target_entities, target_attributes, target_relations, all_names):
    records = {}
    image_requests = []
    request_ids = []

    for candidate in unique_candidates:
        candidate_id = candidate["ent_id"]
        cache_key = f"target_{candidate_id}"

        if cache_key in description_cache:
            records[candidate_id] = description_cache[cache_key]
            continue

        initial_description = build_initial_description(
            target_entities[candidate_id],
            target_attributes[candidate_id],
            target_relations[candidate_id]
        )
        image_path = os.path.join(data_file_path, "concat_images", f"{candidate_id}.jpg")

        if os.path.exists(image_path):
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
            visual_supplement = "No image is available, so no complementary visual information is used."
            full_description = initial_description + "\nVisual complementary information: " + visual_supplement
            records[candidate_id] = {
                "initial_description": initial_description,
                "visual_supplement": visual_supplement,
                "full_description": full_description,
                "anonymous_description": anonymize_description(full_description, all_names),
                "image_available": False
            }

    # 所有未缓存且有图片的候选一次性送入Qwen，而不是循环调用10次
    if image_requests:
        responses = batch_inference(image_requests, max_new_tokens=64)

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

    # 批量生成完成后统一写入缓存，避免每个候选都写一次文件
    cache_changed = False

    for candidate_id, record in records.items():
        cache_key = f"target_{candidate_id}"

        if cache_key not in description_cache:
            description_cache[cache_key] = record
            cache_changed = True

    if cache_changed:
        with open(description_file, "w", encoding="utf-8") as f:
            json.dump(description_cache, f, ensure_ascii=False, indent=4)

    return records


# ==================== 新增：让Qwen先分析10个候选之间的判别性差异 ====================
def build_difference_prompt(candidate_descriptions):
    content = []

    for label, description in candidate_descriptions.items():
        content.append(f"[{label}]\n{description}")

    return (
        "Compare the following ten anonymous candidate entities jointly. For every candidate, identify the information "
        "that distinguishes it from the other candidates, including entity type, attributes, relations, visual evidence, "
        "contradictions, and ambiguities. Do not match them to a source entity yet. Do not infer or output real entity "
        "names.\n\n" + "\n\n".join(content)
    )


# ==================== 新增：最终匿名对齐Prompt（use_surface=0时严格不考虑名称） ====================
def build_alignment_prompt(source_description, candidate_descriptions, difference_response):
    content = []

    for label, description in candidate_descriptions.items():
        content.append(f"[{label}]\n{description}")

    name_rule = (
        "Entity names and surface-form similarity must not be considered because all names are anonymized."
        if not use_name else
        "Entity names may be considered together with the other evidence."
    )

    return (
        "Align the source entity with exactly one of the ten candidate entities. Use the initial descriptions, visual "
        "complementary information, and candidate discriminative differences. Compare all candidates jointly and produce "
        "a complete ranking with no ties. " + name_rule + " Do not use the original retrieval similarities. Give greater "
        "weight to precise and consistent facts than to generic visual resemblance.\n\n"
        f"[SOURCE ENTITY]\n{source_description}\n\n"
        "[CANDIDATE ENTITIES]\n" + "\n\n".join(content) + "\n\n"
        "[CANDIDATE DIFFERENCES]\n" + difference_response + "\n\n"
        "Output exactly one final line in this format:\n"
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


# ==================== 新增：仅加载当前源实体及其10个候选的数据，不预加载整个数据集 ====================
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
    target_attributes = load_attributes(os.path.join(data_file_path, "en_att_triples.txt"), target_entities)

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

    return source_entities, target_entities, source_attributes, target_attributes, source_relations, target_relations


def run_single_entity(ent_id):
    if ent_id not in ng.get_entities():
        print(f"\nEntity {ent_id} is not found in {cand_file_path}")
        return None

    main_entity = ng.get_main_entity(ent_id)
    candidates = ng.get_candidates(ent_id)
    ref_ent = ng.get_ref_ent(ent_id)
    base_rank = ng.get_base_rank(ent_id)
    unique_candidates = list({c["ent_id"]: c for c in candidates}.values())

    print("\n" + "=" * 80)
    print(f"Query ID: {ent_id}")
    print(f"Query name: {main_entity['name']}")
    print(f"Reference ID: {ref_ent}")
    print(f"Base rank: {base_rank} ({base_rank + 1}th position)")
    print(f"Candidates: {[c['ent_id'] for c in unique_candidates]}")

    if base_rank >= 10:
        print("Reference entity is outside Top-10; TTR is skipped.")
        return {"ent_id": ent_id, "base_rank": base_rank, "llm_rank": base_rank, "correct": False, "skipped": True}

    ori_scores = sorted([c["hhea_sim"] for c in unique_candidates], reverse=True)
    if ori_scores[0] >= args.threshold or (len(ori_scores) > 1 and ori_scores[0] - ori_scores[1] > 0.2):
        print("TTR is skipped by the confidence rule.")
        print(f"Final rank: {base_rank} ({base_rank + 1}th position)")
        print(f"Hits@1 correct: {base_rank == 0}")
        return {"ent_id": ent_id, "base_rank": base_rank, "llm_rank": base_rank, "correct": base_rank == 0, "skipped": True}

    if len(unique_candidates) != 10:
        raise ValueError(f"Entity {ent_id} has {len(unique_candidates)} unique candidates instead of 10.")

    start = time.time()
    # ==================== 新增流程：只读取当前Query及其10个候选的数据 ====================
    candidate_ids = [candidate["ent_id"] for candidate in unique_candidates]
    source_entities, target_entities, source_attributes, target_attributes, source_relations, target_relations = load_current_entity_data(ent_id, candidate_ids)
    all_names = [source_entities[ent_id]["name"]] + [target_entities[candidate_id]["name"] for candidate_id in candidate_ids]

    # ==================== 第一步：生成源实体完整描述 ====================
    source_record = get_complete_description(
        ent_id, "source", source_entities[ent_id], source_attributes[ent_id], source_relations[ent_id], all_names
    )

    print("\nSource initial description:\n" + source_record["initial_description"])
    print("\nSource visual supplement:\n" + source_record["visual_supplement"])

    candidate_descriptions = {}
    label_to_candidate = {}

    # ==================== 第二步：一次batch生成所有候选实体的图片补充信息 ====================
    candidate_records = get_candidate_descriptions_batch(
        unique_candidates, target_entities, target_attributes, target_relations, all_names
    )

    for index, candidate in enumerate(unique_candidates, 1):
        candidate_id = candidate["ent_id"]
        label = f"CANDIDATE_{index}"
        record = candidate_records[candidate_id]
        candidate_descriptions[label] = record["full_description"] if use_name else record["anonymous_description"]
        label_to_candidate[label] = candidate

        print(f"\n{label} ID={candidate_id} initial description:\n{record['initial_description']}")
        print(f"\n{label} visual supplement:\n{record['visual_supplement']}")

    # ==================== 第三步：分析10个候选之间的判别性差异 ====================
    difference_response = batch_inference([{
        "text_prompt": build_difference_prompt(candidate_descriptions)
    }], max_new_tokens=64)[0].strip()

    print("\nCandidate discriminative differences:\n" + difference_response)

    # ==================== 第四步：根据候选差异进行最终匿名对齐 ====================
    source_description = source_record["full_description"] if use_name else source_record["anonymous_description"]
    alignment_response = batch_inference([{
        "text_prompt": build_alignment_prompt(source_description, candidate_descriptions, difference_response)
    }])[0].strip()

    # ==================== 第五步：将最终排序映射为唯一分数（9~0） ====================
    ranking_labels = parse_ranking(alignment_response)
    ranked = []

    for rank_index, label in enumerate(ranking_labels):
        ranked.append((label_to_candidate[label], 9 - rank_index, label))

    llm_rank = next(i for i, (candidate, _, _) in enumerate(ranked) if candidate["ent_id"] == ref_ent)

    print("\nFinal alignment response:\n" + alignment_response)
    print("\nDiscriminative reasoning reranking result")

    for rank, (candidate, score, label) in enumerate(ranked, 1):
        mark = "  <-- ground truth" if candidate["ent_id"] == ref_ent else ""
        name = f" | {candidate['name']}" if use_name else ""
        print(f"{rank:2d}. {label} | ID={candidate['ent_id']}{name} | ori={candidate['hhea_sim']:.3f} "
              f"| unique_score={score}{mark}")

    print(f"\nBase rank: {base_rank} ({base_rank + 1}th position)")
    print(f"LLM rank: {llm_rank} ({llm_rank + 1}th position)")
    print(f"Rank improved: {llm_rank < base_rank}")
    print(f"Hits@1 correct: {llm_rank == 0}")
    print(f"Time cost: {time.time() - start:.2f}s")

    return {
        "ent_id": ent_id,
        "base_rank": base_rank,
        "llm_rank": llm_rank,
        "correct": llm_rank == 0,
        "improved": llm_rank < base_rank,
        "skipped": False
    }


def main():
    results = []
    total_start = time.time()

    for ent_id in args.target_ids:
        result = run_single_entity(ent_id)
        if result is not None:
            results.append(result)

    print("\n" + "=" * 80)
    print("Discriminative Reasoning Reranking Summary")
    print(f"Total entities: {len(results)}")
    print(f"Hits@1 correct: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Improved: {sum(r.get('improved', False) for r in results)}/{len(results)}")
    print(f"Skipped: {sum(r['skipped'] for r in results)}/{len(results)}")

    for r in results:
        print(f"ID={r['ent_id']} | base_rank={r['base_rank']} | llm_rank={r['llm_rank']} "
              f"| correct={r['correct']} | skipped={r['skipped']}")

    print(f"Descriptions saved to: {description_file}")
    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
