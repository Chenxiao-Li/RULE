"""Generate CLIP text features from selected DBP15K semantic information."""
import os
import re
import json
import pickle
import argparse

import torch
import clip


def pre_caption(caption, max_words=77):
    """Preprocess text before CLIP tokenization."""
    caption = re.sub(r'([.!\"()*#:;~])', ' ', caption.lower())
    caption = re.sub(r'\s{2,}', ' ', caption)
    caption = caption.rstrip('\n').strip(' ')
    words = caption.split(' ')
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    return caption


def build_entity_text(entity_info, use_name):
    parts = []

    # 修改：name 直接从 ent_semantic_info_select.json 中读取，不再读取 name_dict
    if use_name:
        name = str(entity_info.get("name", "")).strip()
        if name:
            parts.append(name)

    # 修改：直接拼接筛选后的 attribute-value
    for item in entity_info.get("attributes", []):
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()

        if attribute and value:
            parts.append(f"{attribute}: {value}")
        elif attribute:
            parts.append(attribute)
        elif value:
            parts.append(value)

    # 修改：直接拼接 1-hop relation，并保留方向
    for item in entity_info.get("relations", []):
        direction = str(item.get("direction", "")).strip()
        relation = str(item.get("relation", "")).strip()
        neighbor = str(item.get("neighbor", "")).strip()

        if direction == "outgoing":
            parts.append(f"relation: this entity --{relation}--> {neighbor}")
        elif direction == "incoming":
            parts.append(f"relation: {neighbor} --{relation}--> this entity")
        else:
            parts.append(f"relation: {relation}: {neighbor}")

    return ". ".join(parts)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    data_file_path = os.path.join("./data", "DBP15K", args.data_split)
    input_path = os.path.join(data_file_path, args.input_name)

    # 修改：根据是否使用 name 自动区分输出文件
    output_name = "txt_feature_with_name.pkl" if args.use_name else "txt_feature_without_name.pkl"
    output_dir = os.path.join("./data", "pkls", "DBP15K", args.data_split)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            features_dict = pickle.load(f)
    else:
        features_dict = {}

    processed_count = 0

    for entity_id, entity_info in data.items():
        if entity_id in features_dict:
            continue

        try:
            text = build_entity_text(entity_info, args.use_name)
            text = pre_caption(text, max_words=77)

            if not text:
                print(f"Empty text for Entity ID {entity_id}, skipped.")
                continue

            text_tokens = clip.tokenize(text, truncate=True).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy().squeeze()

            features_dict[entity_id] = text_features
            processed_count += 1

            if processed_count % args.save_every == 0:
                with open(output_path, "wb") as f:
                    pickle.dump(features_dict, f)
                print(f"Processed {processed_count} new entities. Saved {len(features_dict)}/{len(data)}.")

        except Exception as e:
            print(f"Error processing Entity ID {entity_id}: {e}")
            continue

    with open(output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Complete. Newly processed: {processed_count}. Total saved: {len(features_dict)}.")
    print(f"Use name: {args.use_name}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP text features from selected DBP15K semantics")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--input_name", default="ent_semantic_info_select.json", type=str)
    parser.add_argument("--use_name", action="store_true")
    parser.add_argument("--save_every", default=100, type=int)
    args = parser.parse_args()
    main(args)
