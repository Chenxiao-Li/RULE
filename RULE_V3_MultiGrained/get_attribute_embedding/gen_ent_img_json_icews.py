"""Generate entity-image JSON for ICEWS."""
import os
import json
import argparse


def read_entity_ids(file_path):
    entity_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"Invalid line: {line}")
                continue
            entity_ids.append(int(parts[0]))
    return entity_ids


def get_images(entity_folder):
    # 修改：按文件名排序，仅保留 jpg/png，并取前三张图片
    image_names = [name for name in sorted(os.listdir(entity_folder)) if name.lower().endswith((".jpg", ".jpeg", ".gif", ".png"))]
    return image_names[:3]


def main(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "ICEWS", args.dataset)
    ent_ids_1_path = os.path.join(dataset_dir, "ent_ids_1")
    ent_ids_2_path = os.path.join(dataset_dir, "ent_ids_2")
    # 修改：ICEWS 图片分别位于 pic_ent_ids_1 / pic_ent_ids_2 下
    img_folder_1 = os.path.join(dataset_dir, "images", "pic_ent_ids_1")
    img_folder_2 = os.path.join(dataset_dir, "images", "pic_ent_ids_2")
    output_path = os.path.join(dataset_dir, args.output_name)

    ent_ids_1 = read_entity_ids(ent_ids_1_path)
    ent_ids_2 = read_entity_ids(ent_ids_2_path)

    data = []
    missing_count = 0

    # 修改：分别从 pic_ent_ids_1 和 pic_ent_ids_2 中读取实体图片
    for entity_id, img_root in [(entity_id, img_folder_1) for entity_id in ent_ids_1] + [(entity_id, img_folder_2) for entity_id in ent_ids_2]:
        entity_folder = os.path.join(img_root, str(entity_id))
        if not os.path.isdir(entity_folder):
            print(f"Image folder not found: {entity_folder}")
            missing_count += 1
            continue

        image_names = get_images(entity_folder)
        if not image_names:
            print(f"No valid image found: {entity_folder}")
            missing_count += 1
            continue

        # 修改：每个实体保存排序后的前三张图片路径，后续可分别生成单图特征
        image_ids = [os.path.join(os.path.basename(img_root), str(entity_id), image_name) for image_name in image_names]
        data.append({"Entity ID": entity_id, "image_ids": image_ids})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Dataset: {args.dataset}")
    print(f"ent_ids_1: {len(ent_ids_1)}")
    print(f"ent_ids_2: {len(ent_ids_2)}")
    print(f"Entities with images: {len(data)}")
    print(f"Missing entities: {missing_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity-image JSON for ICEWS")
    parser.add_argument("--dataset", type=str, default="icews_wiki", choices=["icews_wiki", "icews_yago"])
    parser.add_argument("--output_name", type=str, default="ent_img_mapping.json")
    args = parser.parse_args()
    main(args)
