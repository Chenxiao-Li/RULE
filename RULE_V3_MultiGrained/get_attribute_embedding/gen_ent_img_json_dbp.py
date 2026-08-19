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


def main(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "DBP15K", args.dataset)
    ent_ids_1_path = os.path.join(dataset_dir, "ent_ids_1")
    ent_ids_2_path = os.path.join(dataset_dir, "ent_ids_2")
    img_folder = os.path.join(dataset_dir, "concat_images")
    output_path = os.path.join(dataset_dir, args.output_name)

    ent_ids_1 = read_entity_ids(ent_ids_1_path)
    ent_ids_2 = read_entity_ids(ent_ids_2_path)
    entity_ids = ent_ids_1 + ent_ids_2

    data = []
    missing_count = 0
    for entity_id in entity_ids:
        image_name = f"{entity_id}.jpg"
        image_path = os.path.join(img_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            missing_count += 1
            continue
        data.append({"Entity ID": entity_id, "image_id": image_name})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Dataset: {args.dataset}")
    print(f"ent_ids_1: {len(ent_ids_1)}")
    print(f"ent_ids_2: {len(ent_ids_2)}")
    print(f"Entities with images: {len(data)}")
    print(f"Missing images: {missing_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity-image JSON for DBP15K")
    parser.add_argument("--dataset", type=str, default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--output_name", type=str, default="ent_img_mapping.json")
    args = parser.parse_args()
    main(args)
