"""Generate single-image features for ICEWS using CLIP model."""
import os
import json
import argparse
import pickle

import torch
import clip
from PIL import Image


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 修改：每次只取第 image_index 张图，方便分别生成三个单图 pkl
    image_index = args.image_index - 1

    if os.path.exists(args.output_path):
        with open(args.output_path, "rb") as f:
            features_dict = pickle.load(f)
    else:
        features_dict = {}

    processed_count = 0
    for item in data:
        entity_id = item["Entity ID"]
        image_ids = item["image_ids"]

        # 修改：如果该实体没有对应位置的图片则跳过
        if image_index >= len(image_ids):
            print(f"Image {args.image_index} not found for entity: {entity_id}")
            continue

        # 修改：JSON 中保存相对 images/ 下的 pic_ent_ids_x/entity_id/image 路径
        image_path = os.path.join(args.img_folder, image_ids[image_index])

        if entity_id in features_dict:
            continue

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy().squeeze()

        features_dict[entity_id] = image_features
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} entities.")

    with open(args.output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Complete. Total: {processed_count}. Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate single-image features for ICEWS using CLIP")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # 修改：1/2/3 分别表示排序后的第 1/2/3 张图片
    parser.add_argument("--image_index", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    main(args)
