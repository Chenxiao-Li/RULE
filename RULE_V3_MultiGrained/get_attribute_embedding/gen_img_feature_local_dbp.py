"""Generate local image features from segmented DBP15K images using CLIP."""
import os
import argparse
import pickle

import torch
import clip
from PIL import Image


def load_entity_ids(data_dir):
    """读取 ent_ids_1 和 ent_ids_2 中的全部实体 ID。"""
    entity_ids = []
    for filename in ["ent_ids_1", "ent_ids_2"]:
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entity_ids.append(int(line.split("\t")[0]))
    return entity_ids


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(args.data_dir, args.data_split)
    run_name = "with_name" if args.use_name else "without_name"
    img_folder = os.path.join(data_dir, "seg_images", run_name)
    output_dir = os.path.join("./data/pkls/DBP15K", args.data_split)
    os.makedirs(output_dir, exist_ok=True)
    output_name = "img_feature_local_with_name.pkl" if args.use_name else "img_feature_local_without_name.pkl"
    output_path = os.path.join(output_dir, output_name)

    print(f"Dataset: {args.data_split}")
    print(f"Local image folder: {img_folder}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")

    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    entity_ids = load_entity_ids(data_dir)
    print(f"Total entities: {len(entity_ids)}")

    # 如果已有特征文件，则继续断点处理。
    if os.path.exists(output_path) and not args.overwrite:
        with open(output_path, "rb") as f:
            features_dict = pickle.load(f)
        print(f"Loaded existing features: {len(features_dict)}")
    else:
        features_dict = {}

    processed_count = 0
    skipped_missing = 0
    skipped_existing = 0
    error_count = 0

    for entity_id in entity_ids:
        if entity_id in features_dict:
            skipped_existing += 1
            continue

        image_path = os.path.join(img_folder, f"{entity_id}.jpg")

        # local 图片不存在时直接跳过，不补零向量，与 global feature 保持一致。
        if not os.path.exists(image_path):
            skipped_missing += 1
            continue

        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features = image_features.cpu().numpy().squeeze()
            features_dict[entity_id] = image_features
            processed_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error processing entity {entity_id}: {type(e).__name__}: {e}")
            continue

        if processed_count % max(1, args.save_every) == 0:
            with open(output_path, "wb") as f:
                pickle.dump(features_dict, f)
            print(f"Processed new: {processed_count}, total saved: {len(features_dict)}")

    with open(output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print("\nComplete.")
    print(f"Newly processed: {processed_count}")
    print(f"Existing skipped: {skipped_existing}")
    print(f"Missing local images skipped: {skipped_missing}")
    print(f"Errors: {error_count}")
    print(f"Total saved features: {len(features_dict)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DBP15K local image features using CLIP")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--data_dir", default="./data/DBP15K", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--use_name", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save_every", default=100, type=int)
    args = parser.parse_args()
    main(args)
