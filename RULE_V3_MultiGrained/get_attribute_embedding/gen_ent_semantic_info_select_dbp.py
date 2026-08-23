"""Select concise semantic information for DBP15K entities before CLIP text embedding."""
import os
import json
import argparse

HIGH_PRIORITY_KEYS = {
    "name", "fullname", "nativeName", "originalName", "englishfullname", "playername",
    "occupation", "office", "title", "role", "type", "category", "status",
    "country", "nationality", "birthPlace", "placeofbirth", "deathPlace", "location",
    "headquarters", "cityname", "subdivisionName", "region",
    "team", "club", "clubname", "currentclub", "organization", "institution",
    "workInstitution", "work", "profession", "allegiance"
}

MID_PRIORITY_KEYS = {
    "birthDate", "dateofbirth", "deathDate", "founded", "foundation", "released",
    "introduced", "language", "script", "religion", "spouse", "partner",
    "awards", "members", "populationTotal", "areaTotal", "marketingTarget",
    "supportedPlatforms", "predecessor", "successor", "primaryUser"
}

NOISE_KEYWORDS = [
    "image", "img", "pic", "pixel", "width", "height", "size", "table",
    "website", "url", "http", "id", "code", "pattern", "color", "colour",
    "leftarm", "rightarm", "shorts", "socks", "logo", "signature",
    "caption", "map", "file", "filename", "number", "order",
    "termstart", "termend", "year", "rank", "ratio", "update"
]

def is_noise_attribute(attribute):
    key = attribute.strip()
    key_lower = key.lower()
    if key in HIGH_PRIORITY_KEYS or key in MID_PRIORITY_KEYS:
        return False
    return any(noise in key_lower for noise in NOISE_KEYWORDS)

def get_priority(attribute):
    if attribute in HIGH_PRIORITY_KEYS:
        return 0
    if attribute in MID_PRIORITY_KEYS:
        return 1
    return 2

def select_attributes(attributes, top_k):
    candidates = []
    for index, item in enumerate(attributes):
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()
        if not attribute:
            continue
        if is_noise_attribute(attribute):
            continue
        candidates.append({
            "attribute": attribute,
            "value": value,
            "_priority": get_priority(attribute),
            "_index": index
        })

    candidates.sort(key=lambda x: (x["_priority"], x["_index"]))

    selected = []
    seen_pairs = set()
    for item in candidates:
        pair = (item["attribute"], item["value"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected.append({
            "attribute": item["attribute"],
            "value": item["value"]
        })
        if top_k > 0 and len(selected) >= top_k:
            break
    return selected

def select_relation(relations):
    if not relations:
        return []
    return [relations[0]]

def main(args):
    data_file_path = os.path.join("./data", "DBP15K", args.data_split)
    input_path = os.path.join(data_file_path, args.input_name)
    output_path = os.path.join(data_file_path, args.output_name)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for entity_id, entity_info in data.items():
        result[entity_id] = {
            "name": entity_info.get("name", ""),
            "attributes": select_attributes(entity_info.get("attributes", []), args.top_k),
            "relations": select_relation(entity_info.get("relations", []))
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_entities = len(result)
    total_attributes = sum(len(item["attributes"]) for item in result.values())
    total_relations = sum(len(item["relations"]) for item in result.values())

    print(f"Dataset: {args.data_split}")
    print(f"Total entities: {total_entities}")
    print(f"Selected attributes: {total_attributes}")
    print(f"Selected 1-hop relations: {total_relations}")
    print(f"Top-k attributes per entity: {args.top_k}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select DBP15K semantic information for CLIP text embedding")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--input_name", default="ent_semantic_info.json", type=str)
    parser.add_argument("--output_name", default="ent_semantic_info_select.json", type=str)
    parser.add_argument("--top_k", default=5, type=int)
    args = parser.parse_args()
    main(args)
