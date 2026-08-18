import os
import re
import json
import argparse
from collections import defaultdict
from urllib.parse import unquote

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


PROMPT_TEMPLATE = """
You are generating descriptions for cross-lingual knowledge graph entity alignment.

Write a concise English description of the entity using only the supplied information.

Requirements:
1. Do not add facts that are not supported by the supplied information.
2. Preserve important entity names, dates, locations, organizations, categories, and numerical values.
3. Combine the entity name, attribute values, relations, neighboring entities, and useful visual evidence.
4. Ignore duplicated, empty, malformed, or obviously corrupted information.
5. Do not mention images, triples, prompts, attributes, relations, neighbors, or knowledge graphs.
6. Write 2 to 4 complete sentences.
7. Use no more than {max_words} words.
8. Output only the description in English.

Entity name:
{name}

Attribute facts:
{attributes}

Relation facts:
{relations}

Neighboring entities:
{neighbors}
""".strip()


def read_id_map(file_path):
    id_map = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            try:
                item_id = int(parts[0])
            except ValueError:
                continue

            id_map[item_id] = parts[1].strip()

    return id_map


def uri_to_name(uri):
    if not uri:
        return ""

    text = uri.strip().strip("<>")

    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]

    text = unquote(text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_ntriple_line(line):
    match = re.match(r'^\s*<([^>]*)>\s+<([^>]*)>\s+(.+?)\s*\.\s*$', line)

    if match is None:
        return None

    return match.group(1), match.group(2), match.group(3).strip()


def parse_literal(obj):
    if not obj:
        return ""

    if obj.startswith("<") and obj.endswith(">"):
        return uri_to_name(obj[1:-1])

    if not obj.startswith('"'):
        return obj.strip()

    escaped = False
    end_quote = None

    for index in range(1, len(obj)):
        char = obj[index]

        if char == '"' and not escaped:
            end_quote = index
            break

        escaped = char == "\\" and not escaped

    if end_quote is None:
        return obj.strip('"')

    value = obj[1:end_quote]

    try:
        value = bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        pass

    value = value.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", value).strip()


def read_attribute_triples(file_path, ent_map):
    attr_dic = defaultdict(list)
    uri_to_id = {}

    for ent_id, uri in ent_map.items():
        uri_to_id[uri] = ent_id
        uri_to_id[uri.strip("<>")] = ent_id

    parsed_count = 0
    matched_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_ntriple_line(line)
            if parsed is None:
                continue

            parsed_count += 1
            subject, predicate, obj = parsed
            ent_id = uri_to_id.get(subject)

            if ent_id is None:
                continue

            attr_name = uri_to_name(predicate)
            attr_value = parse_literal(obj)

            if not attr_name or not attr_value:
                continue

            fact = f"{attr_name}: {attr_value}"

            if fact not in attr_dic[ent_id]:
                attr_dic[ent_id].append(fact)
                matched_count += 1

    print(f"Attribute triples parsed: {parsed_count}")
    print(f"Attribute facts matched to entities: {matched_count}")

    return attr_dic


def read_triples(file_path):
    triples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) != 3:
                parts = line.strip().split()
            if len(parts) != 3:
                continue

            try:
                head, relation, tail = map(int, parts)
            except ValueError:
                continue

            triples.append((head, relation, tail))

    return triples


def build_graph_information(triples, ent_map, rel_map):
    rel_dic = defaultdict(list)
    neighbor_dic = defaultdict(list)

    for head, relation, tail in triples:
        if head not in ent_map or tail not in ent_map:
            continue

        rel_name = uri_to_name(rel_map.get(relation, str(relation)))
        head_name = uri_to_name(ent_map[head])
        tail_name = uri_to_name(ent_map[tail])

        outgoing_fact = f"{rel_name}: {tail_name}"
        incoming_fact = f"inverse {rel_name}: {head_name}"

        if outgoing_fact not in rel_dic[head]:
            rel_dic[head].append(outgoing_fact)
        if incoming_fact not in rel_dic[tail]:
            rel_dic[tail].append(incoming_fact)
        if tail_name and tail_name not in neighbor_dic[head]:
            neighbor_dic[head].append(tail_name)
        if head_name and head_name not in neighbor_dic[tail]:
            neighbor_dic[tail].append(head_name)

    return rel_dic, neighbor_dic


def find_image_path(image_dir, ent_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        image_path = os.path.join(image_dir, f"{ent_id}{ext}")
        if os.path.isfile(image_path):
            return image_path

    return None


def format_items(items, max_items):
    if not items:
        return "None"

    result = [f"- {str(item).strip()}" for item in items[:max_items] if str(item).strip()]
    return "\n".join(result) if result else "None"


def build_prompt(entity, args):
    return PROMPT_TEMPLATE.format(
        max_words=args.max_words,
        name=entity["name"],
        attributes=format_items(entity["attributes"], args.max_attributes),
        relations=format_items(entity["relations"], args.max_relations),
        neighbors=format_items(entity["neighbors"], args.max_neighbors)
    )


def build_messages(entity, args):
    content = []

    if entity["image_path"] is not None:
        content.append({"type": "image", "image": "file://" + entity["image_path"]})

    content.append({"type": "text", "text": build_prompt(entity, args)})
    return [{"role": "user", "content": content}]


def get_input_device(model):
    if hasattr(model, "hf_device_map"):
        for module_name in ["model.embed_tokens", "model.visual", "visual"]:
            if module_name not in model.hf_device_map:
                continue

            device = model.hf_device_map[module_name]

            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in ["cpu", "disk"]:
                return torch.device(device)

        for device in model.hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)

    return next(model.parameters()).device


def move_inputs_to_device(inputs, device):
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)

    return inputs


def generate_description(entity, model, processor, args):
    messages = build_messages(entity, args)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = move_inputs_to_device(inputs, get_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()


def load_completed_ids(output_path):
    completed_ids = set()

    if not os.path.isfile(output_path):
        return completed_ids

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "entity_id" in item and item.get("description", "").strip():
                completed_ids.add(int(item["entity_id"]))

    return completed_ids


def load_entities(split_dir, kg_id, attr_file):
    ent_map = read_id_map(os.path.join(split_dir, f"ent_ids_{kg_id}"))
    rel_map = read_id_map(os.path.join(split_dir, f"rel_ids_{kg_id}"))
    triples = read_triples(os.path.join(split_dir, f"triples_{kg_id}"))
    attr_dic = read_attribute_triples(os.path.join(split_dir, attr_file), ent_map)
    rel_dic, neighbor_dic = build_graph_information(triples, ent_map, rel_map)
    image_dir = os.path.join(split_dir, "concat_images")

    entities = []

    for ent_id, uri in ent_map.items():
        entities.append({
            "entity_id": ent_id,
            "entity_uri": uri,
            "name": uri_to_name(uri),
            "attributes": attr_dic.get(ent_id, []),
            "relations": rel_dic.get(ent_id, []),
            "neighbors": neighbor_dic.get(ent_id, []),
            "image_path": find_image_path(image_dir, ent_id),
            "kg_id": kg_id
        })

    entities.sort(key=lambda item: item["entity_id"])

    print(f"\nKG {kg_id} statistics")
    print(f"Entities: {len(entities)}")
    print(f"Entities with attribute values: {sum(1 for item in entities if item['attributes'])}")
    print(f"Entities with relations: {sum(1 for item in entities if item['relations'])}")
    print(f"Entities with images: {sum(1 for item in entities if item['image_path'] is not None)}\n")

    return entities


def write_result(output_file, result):
    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    output_file.flush()


def process_kg(kg_id, attr_file, split_dir, output_dir, model, processor, args):
    entities = load_entities(split_dir, kg_id, attr_file)
    output_path = os.path.join(output_dir, f"entity_descriptions_{kg_id}.jsonl")
    completed_ids = load_completed_ids(output_path) if args.resume else set()

    print(f"KG {kg_id} already completed: {len(completed_ids)}")
    print(f"KG {kg_id} output: {output_path}")

    with open(output_path, "a", encoding="utf-8") as output_file:
        for entity in tqdm(entities, desc=f"Generating KG {kg_id}"):
            ent_id = entity["entity_id"]

            if ent_id in completed_ids:
                continue

            try:
                description = generate_description(entity, model, processor, args)

                result = {
                    "entity_id": ent_id,
                    "entity_uri": entity["entity_uri"],
                    "name": entity["name"],
                    "description": description,
                    "kg_id": kg_id,
                    "attributes": entity["attributes"],
                    "relations": entity["relations"],
                    "neighbors": entity["neighbors"],
                    "image_path": entity["image_path"] or "",
                    "has_image": entity["image_path"] is not None
                }

                write_result(output_file, result)
                completed_ids.add(ent_id)

            except Exception as error:
                print(f"KG {kg_id} entity {ent_id} failed: {repr(error)}")

                if args.save_error:
                    result = {
                        "entity_id": ent_id,
                        "entity_uri": entity["entity_uri"],
                        "name": entity["name"],
                        "description": "",
                        "kg_id": kg_id,
                        "error": repr(error)
                    }
                    write_result(output_file, result)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
    )
    parser.add_argument("--dataset", default="DBP15K")
    parser.add_argument("--data_split", default="fr_en", choices=["fr_en", "ja_en", "zh_en"])
    parser.add_argument("--data_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--kg", default="both", choices=["1", "2", "both"])
    parser.add_argument("--output_dir", default="")

    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--max_words", type=int, default=100)
    parser.add_argument("--max_attributes", type=int, default=30)
    parser.add_argument("--max_relations", type=int, default=30)
    parser.add_argument("--max_neighbors", type=int, default=30)

    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)

    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_error", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()

    split_dir = os.path.join(args.data_root, args.dataset, args.data_split)
    output_dir = args.output_dir if args.output_dir else os.path.join(split_dir, "descriptions")

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    os.makedirs(output_dir, exist_ok=True)

    kg_ids = [1, 2] if args.kg == "both" else [int(args.kg)]

    if args.data_split == "fr_en":
        attr_files = {1: "fr_att_triples", 2: "en_att_triples"}
    elif args.data_split == "ja_en":
        attr_files = {1: "ja_att_triples", 2: "en_att_triples"}
    else:
        attr_files = {1: "zh_att_triples.txt", 2: "en_att_triples.txt"}

    for kg_id in kg_ids:
        check_file(os.path.join(split_dir, f"ent_ids_{kg_id}"))
        check_file(os.path.join(split_dir, f"rel_ids_{kg_id}"))
        check_file(os.path.join(split_dir, f"triples_{kg_id}"))
        check_file(os.path.join(split_dir, attr_files[kg_id]))

    print(f"Dataset directory: {split_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {args.model_path}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True
    )

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    for kg_id in kg_ids:
        process_kg(kg_id, attr_files[kg_id], split_dir, output_dir, model, processor, args)

    print("Description generation completed.")


if __name__ == "__main__":
    main()
