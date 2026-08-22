"""Generate entity semantic information for DBP15K."""
import os
import re
import json
import argparse
from collections import defaultdict
from urllib.parse import unquote


def read_id_uri(file_path):
    id2uri, uri2id = {}, {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entity_id, uri = line.split("\t", 1)
            entity_id = int(entity_id)
            id2uri[entity_id] = uri
            uri2id[uri] = entity_id
    return id2uri, uri2id


def read_id_name(file_path):
    id2name = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 修改：rel_ids 文件不一定使用 Tab 分隔，统一按任意空白符切分，最多切成两部分
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"Invalid id-name line: {line}")
                continue
            item_id, uri = parts
            id2name[int(item_id)] = clean_uri_name(uri)
    return id2name


def clean_uri_name(uri):
    name = unquote(uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1])
    return name.replace("_", " ")


def resolve_att_file(dataset_dir, language):
    # 修改：zh_en 的 att_triples 带 .txt，fr_en / ja_en 可不带后缀；这里自动兼容两种形式
    candidates = [
        os.path.join(dataset_dir, f"{language}_att_triples.txt"),
        os.path.join(dataset_dir, f"{language}_att_triples"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Attribute triples not found: {candidates}")


def read_training_attrs(file_path):
    # 修改：保留 training_attrs 中属性的原始顺序，后续最多抽取 3 个实际存在属性值的属性
    attrs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                attrs[parts[0]] = parts[1:]
    return attrs


def parse_att_line(line):
    match = re.match(r'^<([^>]*)>\s+<([^>]*)>\s+(.+)\s+\.\s*$', line.strip())
    if not match:
        return None
    subject, predicate, obj = match.groups()
    literal_match = re.match(r'^"((?:\\.|[^"\\])*)"(?:@[^\s]+|\^\^<[^>]+>)?$', obj)
    if literal_match:
        value = literal_match.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\")
    elif obj.startswith("<") and obj.endswith(">"):
        value = clean_uri_name(obj[1:-1])
    else:
        value = obj
    return subject, predicate, value


def load_attributes(att_file, training_attrs, uri2id):
    # 修改：统一使用清洗后的属性名作为匹配 key，避免同一属性同时出现“有值”和空值
    # 规则：
    # 1) *_att_triples 中所有可匹配的 attribute-value 全部保留
    # 2) 同一个 attribute 有多个不同 value 时全部保留
    # 3) training_attrs 中存在、但 *_att_triples 中完全没有任何 value 的属性，才补一个 value=""
    selected = defaultdict(list)
    seen_pairs = defaultdict(set)
    attrs_with_value = defaultdict(set)

    with open(att_file, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_att_line(line)
            if not parsed:
                continue
            subject, predicate, value = parsed
            if subject not in uri2id:
                continue
            entity_id = uri2id[subject]
            attribute = clean_uri_name(predicate)
            pair = (attribute, value)
            if pair in seen_pairs[entity_id]:
                continue
            selected[entity_id].append({"attribute": attribute, "value": value})
            seen_pairs[entity_id].add(pair)
            if value != "":
                attrs_with_value[entity_id].add(attribute)

    # 修改：training_attrs 也先清洗属性名，再判断该属性是否已经有非空 value
    for subject, attr_list in training_attrs.items():
        if subject not in uri2id:
            continue
        entity_id = uri2id[subject]
        for predicate in attr_list:
            attribute = clean_uri_name(predicate)
            if attribute in attrs_with_value[entity_id]:
                continue
            pair = (attribute, "")
            if pair not in seen_pairs[entity_id]:
                selected[entity_id].append({"attribute": attribute, "value": ""})
                seen_pairs[entity_id].add(pair)

    return selected

def load_relations(triples_path, rel_ids_path, id2name):
    rel_id2name = read_id_name(rel_ids_path)
    relations = defaultdict(list)
    seen = defaultdict(set)

    # 修改：只构造 1-hop 关系，同时保留 outgoing 和 incoming；不继续扩展到 2-hop
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = map(int, parts)
            relation = rel_id2name.get(rel, str(rel))
            if head in id2name and tail in id2name:
                out_item = ("outgoing", relation, id2name[tail])
                in_item = ("incoming", relation, id2name[head])
                # 修改：每个实体只保留遇到的第 1 个一跳关系
                if not relations[head] and out_item not in seen[head]:
                    relations[head].append({"direction": "outgoing", "relation": relation, "neighbor": id2name[tail]})
                    seen[head].add(out_item)
                if not relations[tail] and in_item not in seen[tail]:
                    relations[tail].append({"direction": "incoming", "relation": relation, "neighbor": id2name[head]})
                    seen[tail].add(in_item)
    return relations


def main(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "DBP15K", args.data_split)
    lang_1 = args.data_split.split("_")[0]
    lang_2 = args.data_split.split("_")[1]

    ent_ids_1 = os.path.join(dataset_dir, "ent_ids_1")
    ent_ids_2 = os.path.join(dataset_dir, "ent_ids_2")
    training_attrs_1 = os.path.join(dataset_dir, "training_attrs_1")
    training_attrs_2 = os.path.join(dataset_dir, "training_attrs_2")
    triples_1 = os.path.join(dataset_dir, "triples_1")
    triples_2 = os.path.join(dataset_dir, "triples_2")
    rel_ids_1 = os.path.join(dataset_dir, "rel_ids_1")
    rel_ids_2 = os.path.join(dataset_dir, "rel_ids_2")
    att_file_1 = resolve_att_file(dataset_dir, lang_1)
    att_file_2 = resolve_att_file(dataset_dir, lang_2)
    output_path = os.path.join(dataset_dir, args.output_name)

    id2uri_1, uri2id_1 = read_id_uri(ent_ids_1)
    id2uri_2, uri2id_2 = read_id_uri(ent_ids_2)
    id2name_1 = {eid: clean_uri_name(uri) for eid, uri in id2uri_1.items()}
    id2name_2 = {eid: clean_uri_name(uri) for eid, uri in id2uri_2.items()}

    attrs_1 = load_attributes(att_file_1, read_training_attrs(training_attrs_1), uri2id_1)
    attrs_2 = load_attributes(att_file_2, read_training_attrs(training_attrs_2), uri2id_2)
    rels_1 = load_relations(triples_1, rel_ids_1, id2name_1)
    rels_2 = load_relations(triples_2, rel_ids_2, id2name_2)

    result = {}
    # 修改：最终 JSON 不保存 URI 和 neighbor_id，只保留对 Qwen 有直接语义价值的信息
    for entity_id in sorted(id2name_1):
        result[str(entity_id)] = {"name": id2name_1[entity_id], "attributes": attrs_1.get(entity_id, []), "relations": rels_1.get(entity_id, [])}
    for entity_id in sorted(id2name_2):
        result[str(entity_id)] = {"name": id2name_2[entity_id], "attributes": attrs_2.get(entity_id, []), "relations": rels_2.get(entity_id, [])}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 修改：统计最终 JSON 中保留的属性、属性值和 1-hop 关系数量
    attribute_count = sum(len(item["attributes"]) for item in result.values())
    attribute_value_count = sum(sum(1 for attr in item["attributes"] if attr.get("value", "") != "") for item in result.values())
    relation_count = sum(len(item["relations"]) for item in result.values())
    print(f"Dataset: {args.data_split}")
    print(f"Total entities: {len(result)}")
    print(f"Total attributes: {attribute_count}")
    print(f"Total attribute values: {attribute_value_count}")
    print(f"Total 1-hop relations: {relation_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DBP15K entity semantic information")
    parser.add_argument("--data_split", type=str, default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--output_name", type=str, default="ent_semantic_info.json")
    args = parser.parse_args()
    main(args)