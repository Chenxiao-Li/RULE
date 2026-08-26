"""Generate CLIP text features from selected DBP15K semantic information."""
import os
import re
import json
import pickle
import argparse

import numpy as np
import torch
import clip


def pre_caption(caption):
    """Preprocess text before CLIP tokenization."""
    caption = re.sub(r'([.!\"()*#:;~])', ' ', caption.lower())
    caption = re.sub(r'\s{2,}', ' ', caption)
    caption = caption.rstrip('\n').strip(' ')
    return caption


def build_entity_text(entity_info, use_surface):
    parts = []

    # 修改：name 直接从 ent_semantic_info_select.json 中读取，不再读取 name_dict
    if use_surface:
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



def load_char_bigram(name_path):
    """Build RULE-style character-bigram vocabulary."""
    with open(name_path, "r", encoding="utf-8") as f:
        ent_names = json.load(f)

    char2id = {}
    for _, name in ent_names:
        for word in name:
            word = str(word).lower()
            for idx in range(len(word) - 1):
                bigram = word[idx:idx + 2]
                if bigram not in char2id:
                    char2id[bigram] = len(char2id)
    return ent_names, char2id


def generate_char_features(data_split):
    """Generate normalized RULE-style char-bigram features and return them in memory."""
    name_path = os.path.join("./data", "DBP15K", "translated_ent_name", f"dbp_{data_split}.json")
    if not os.path.exists(name_path):
        raise FileNotFoundError(f"Translated entity-name file not found: {name_path}")

    ent_names, char2id = load_char_bigram(name_path)
    char_dim = len(char2id)
    char_features = {}

    for entity_id, name in ent_names:
        entity_id = int(entity_id)
        char_vec = np.zeros(char_dim, dtype=np.float32)

        for word in name:
            word = str(word).lower()
            for idx in range(len(word) - 1):
                char_vec[char2id[word[idx:idx + 2]]] += 1.0

        # 保留 RULE 的处理：没有任何 bigram 时使用随机向量。
        if np.sum(char_vec) == 0:
            char_vec = np.random.random(char_dim).astype(np.float32) - 0.5

        norm = np.linalg.norm(char_vec)
        if norm > 0:
            char_vec = char_vec / norm

        char_features[entity_id] = char_vec.astype(np.float32)

    print(f"Char vocabulary size: {char_dim}")
    print(f"Char entities: {len(char_features)}")
    return char_features, char_dim


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    data_file_path = os.path.join("./data", "DBP15K", args.data_split)
    input_path = os.path.join(data_file_path, args.input_name)

    # 修改：根据是否使用 name 自动区分输出文件
    output_name = "txt_feature_with_surface.pkl" if args.use_surface else "txt_feature_without_surface.pkl"
    output_dir = os.path.join("./data", "pkls", "DBP15K", args.data_split)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # use_surface=True:
    # 1) name 加入 CLIP 文本
    # 2) char bigram 与 CLIP text feature 直接 concat
    # 3) 最终只保存一个 txt_feature_with_surface.pkl
    char_features = None
    char_dim = 0
    if args.use_surface:
        char_features, char_dim = generate_char_features(args.data_split)

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
            text = build_entity_text(entity_info, args.use_surface)
            text = pre_caption(text)

            if not text:
                print(f"Empty text for Entity ID {entity_id}, skipped.")
                continue

            text_tokens = clip.tokenize(text, truncate=True).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy().squeeze().astype(np.float32)

            if args.use_surface:
                entity_id_int = int(entity_id)
                if entity_id_int not in char_features:
                    raise KeyError(f"Missing char feature for Entity ID {entity_id}")

                char_feature = char_features[entity_id_int]

                # text 和 char 已分别 L2-normalize，直接 concat 成一个 surface-aware text feature。
                final_feature = np.concatenate([text_features, char_feature], axis=0).astype(np.float32)
            else:
                final_feature = text_features

            features_dict[entity_id] = final_feature
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
    print(f"Use surface: {args.use_surface}")
    if args.use_surface:
        print(f"CLIP text dim: 768")
        print(f"Char dim: {char_dim}")
        print(f"Final with-surface dim: {768 + char_dim}")
    else:
        print(f"Final without-surface dim: 768")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP text features from selected DBP15K semantics")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--input_name", default="ent_semantic_info_select.json", type=str)
    parser.add_argument("--use_surface", action="store_true")
    parser.add_argument("--save_every", default=100, type=int)
    args = parser.parse_args()
    main(args)