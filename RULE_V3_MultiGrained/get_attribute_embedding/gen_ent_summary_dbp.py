"""Generate CLIP-friendly entity summaries for DBP15K using Qwen2.5-VL-72B-Instruct."""
import os
import json
import argparse
import gc

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MLLM_PATH = "/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


def build_entity_text(entity_id, entity_info, use_name, name_dict):
    parts = []

    # 修改：名称由超参数控制，便于后续做 with-name / without-name 消融
    if use_name:
        name = name_dict.get(str(entity_id), "")
        if name:
            parts.append(f"Entity name: {name}")

    attributes = entity_info.get("attributes", [])
    parts.append("Attributes:")
    if attributes:
        for item in attributes:
            attribute = item.get("attribute", "").strip()
            value = str(item.get("value", "")).strip()
            if attribute and value:
                parts.append(f"- {attribute}: {value}")
            elif attribute:
                parts.append(f"- {attribute}")
            elif value:
                parts.append(f"- value: {value}")
    else:
        parts.append("- none")

    relations = entity_info.get("relations", [])
    parts.append("Relations:")
    if relations:
        for item in relations:
            direction = item.get("direction", "")
            relation = item.get("relation", "").strip()
            neighbor = item.get("neighbor", "").strip()
            if direction == "outgoing":
                parts.append(f"- this entity --{relation}--> {neighbor}")
            elif direction == "incoming":
                parts.append(f"- {neighbor} --{relation}--> this entity")
            else:
                parts.append(f"- {relation}: {neighbor}")
    else:
        parts.append("- none")

    return "\n".join(parts)


def build_summary_prompt(entity_text):
    # 修改：摘要专门面向后续 CLIP Text Encoder，而不是生成百科式长描述
    return (
        "Create one concise English entity description for CLIP text embedding from the structured knowledge below.\n"
        "Use ONLY the provided information. Do not use external knowledge, prior knowledge, or unsupported inference.\n"
        "The description must be ONE sentence and no more than 40 English words.\n"
        "Preserve the entity's identity when an entity name is provided. "
        "Prioritize the most discriminative semantic information, such as entity type, role, occupation, location, organization, work, or other identity-defining facts. "
        "Use relation information only when it helps characterize the entity.\n"
        "Ignore or strongly downweight noisy metadata that is unlikely to help image-text semantic matching, such as image sizes, pixel sizes, table widths, raw IDs, URLs, formatting fields, file names, color codes, or template-specific fields.\n"
        "Do not invent missing facts. Do not explain your reasoning. Output only the final English sentence.\n\n"
        f"{entity_text}"
    )


def generate_summary(model, processor, entity_id, entity_info, use_name, name_dict, max_new_tokens):
    entity_text = build_entity_text(entity_id, entity_info, use_name, name_dict)
    prompt = build_summary_prompt(entity_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    summary = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    summary = " ".join(summary.splitlines()).strip()
    del inputs, generated_ids
    return summary


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(args):
    data_file_path = os.path.join("./data", "DBP15K", args.data_split)
    input_path = os.path.join(data_file_path, args.input_name)
    name_dict_path = os.path.join(data_file_path, "candidates", "name_dict")

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = "ent_summary_with_name.json" if args.use_name else "ent_summary_without_name.json"
    output_path = os.path.join(data_file_path, output_name)

    with open(input_path, "r", encoding="utf-8") as f:
        semantic_info = json.load(f)

    # 修改：带 name 时，实体名统一从 candidates/name_dict 获取，而不是从 ent_semantic_info.json 获取
    if args.use_name:
        with open(name_dict_path, "r", encoding="utf-8") as f:
            name_dict = json.load(f)["ent"]
    else:
        name_dict = {}

    # 修改：支持断点续跑，已有 Entity ID 直接跳过
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
    else:
        summaries = {}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MLLM_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MLLM_PATH)

    processed_count = 0
    total_count = len(semantic_info)

    for entity_id, entity_info in semantic_info.items():
        if entity_id in summaries:
            continue

        try:
            summary = generate_summary(
                model,
                processor,
                entity_id,
                entity_info,
                args.use_name,
                name_dict,
                args.max_new_tokens
            )
            summaries[entity_id] = {"summary": summary}
            processed_count += 1

            if processed_count % args.save_every == 0:
                save_json(summaries, output_path)
                print(f"Processed {processed_count} new entities. Saved {len(summaries)}/{total_count}.")

        except Exception as e:
            print(f"Error processing Entity ID {entity_id}: {e}")
            continue

    save_json(summaries, output_path)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Complete. Newly processed: {processed_count}. Total saved: {len(summaries)}.")
    print(f"Use name: {args.use_name}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP-friendly DBP15K entity summaries")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--input_name", default="ent_semantic_info.json", type=str)
    parser.add_argument("--output_name", default=None, type=str)
    # 修改：默认不带 name；加 --use_name 时生成带 name 的摘要，方便后续消融
    parser.add_argument("--use_name", action="store_true")
    parser.add_argument("--max_new_tokens", default=80, type=int)
    parser.add_argument("--save_every", default=100, type=int)
    args = parser.parse_args()
    main(args)
