from collections import defaultdict
from pathlib import Path


def load_entity_id_mapping(ent_ids_path):
    """
    读取实体 ID 映射文件 ent_ids_2。

    文件格式：
        entity_id<TAB>entity_uri

    返回：
        {
            entity_id: entity_uri
        }
    """
    entity_id_to_uri = {}

    with open(ent_ids_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"{ent_ids_path} 第 {line_number} 行格式错误：{line}"
                )

            entity_id_text, entity_uri = parts

            try:
                entity_id = int(entity_id_text)
            except ValueError as exc:
                raise ValueError(
                    f"{ent_ids_path} 第 {line_number} 行的实体 ID 不是整数："
                    f"{entity_id_text}"
                ) from exc

            entity_id_to_uri[entity_id] = entity_uri

    return entity_id_to_uri


def load_relation_triples(en_rel_triples_path):
    """
    读取 URI 形式的关系三元组 en_rel_triples。

    文件格式：
        head_uri<TAB>relation_uri<TAB>tail_uri

    同一个头尾实体可能对应多个关系，因此返回：

        {
            (head_uri, tail_uri): [
                relation_uri_1,
                relation_uri_2,
                ...
            ]
        }

    使用列表保存关系，是为了保留关系在原文件中的出现顺序。
    """
    entity_pair_to_relations = defaultdict(list)

    with open(en_rel_triples_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"{en_rel_triples_path} 第 {line_number} 行格式错误：{line}"
                )

            head_uri, relation_uri, tail_uri = parts
            entity_pair = (head_uri, tail_uri)

            # 防止同一个完全相同的三元组被重复保存
            if relation_uri not in entity_pair_to_relations[entity_pair]:
                entity_pair_to_relations[entity_pair].append(relation_uri)

    return entity_pair_to_relations


def build_relation_id_mapping(
    triples_path,
    entity_id_to_uri,
    entity_pair_to_relations
):
    """
    根据 triples_2 生成关系 ID 到关系 URI 的映射。

    triples_2 格式：
        head_id<TAB>relation_id<TAB>tail_id

    处理规则：
    1. 如果 relation_id 已经有映射，直接跳过。
    2. 将 head_id 和 tail_id 转换成 URI。
    3. 根据 (head_uri, tail_uri) 查找候选 relation_uri。
    4. 从候选关系中选择一个尚未被其他关系 ID 使用的关系。
    5. 建立 relation_id -> relation_uri 映射。

    返回：
        relation_id_to_uri:
            {
                relation_id: relation_uri
            }

        unresolved_relation_ids:
            没有找到映射的关系 ID 集合
    """
    relation_id_to_uri = {}

    # 用于保证一个关系 URI 只对应一个关系 ID
    used_relation_uris = set()

    # 保存暂时没有找到映射的关系 ID
    unresolved_relation_ids = set()

    with open(triples_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"{triples_path} 第 {line_number} 行格式错误：{line}"
                )

            try:
                head_id, relation_id, tail_id = map(int, parts)
            except ValueError as exc:
                raise ValueError(
                    f"{triples_path} 第 {line_number} 行包含非整数 ID：{line}"
                ) from exc

            # 该关系 ID 已经成功映射，后续三元组不再处理
            if relation_id in relation_id_to_uri:
                continue

            head_uri = entity_id_to_uri.get(head_id)
            tail_uri = entity_id_to_uri.get(tail_id)

            # 头实体或尾实体在 ent_ids_2 中不存在
            if head_uri is None or tail_uri is None:
                unresolved_relation_ids.add(relation_id)
                continue

            candidate_relations = entity_pair_to_relations.get(
                (head_uri, tail_uri)
            )

            # en_rel_triples 中不存在该头尾实体组合
            if not candidate_relations:
                unresolved_relation_ids.add(relation_id)
                continue

            selected_relation_uri = None

            # 选择尚未被其他关系 ID 使用的关系 URI
            for relation_uri in candidate_relations:
                if relation_uri not in used_relation_uris:
                    selected_relation_uri = relation_uri
                    break

            # 当前实体对中的所有关系都已经被使用
            # 暂时跳过，后续遇到该 relation_id 的其他实体对时继续尝试
            if selected_relation_uri is None:
                unresolved_relation_ids.add(relation_id)
                continue

            relation_id_to_uri[relation_id] = selected_relation_uri
            used_relation_uris.add(selected_relation_uri)

            # 已成功找到映射，从未解决集合中移除
            unresolved_relation_ids.discard(relation_id)

    return relation_id_to_uri, unresolved_relation_ids


def write_relation_id_mapping(relation_id_to_uri, output_path):
    """
    将关系映射写入 rel_ids_2。

    输出格式：
        relation_id<TAB>relation_uri
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for relation_id in sorted(relation_id_to_uri):
            relation_uri = relation_id_to_uri[relation_id]
            file.write(f"{relation_id}\t{relation_uri}\n")


def main():
    # 输入文件
    ent_ids_path = Path("D:/Other/Code/Datasets/MEAformer_Datasets/DBP_raw/DBP15k_raw_all/DBP15k/DBP15k_raw/zh_en/ent_ids_2")
    en_rel_triples_path = Path("D:/Other/Code/Datasets/MEAformer_Datasets/DBP_raw/DBP15k_raw_all/DBP15k/DBP15k_raw/zh_en/en_rel_triples.txt")
    triples_path = Path("D:/Other/Code/Datasets/MEAformer_Datasets/DBP_raw/DBP15k_raw_all/DBP15k/DBP15k_raw/zh_en/triples_2")

    # 输出文件
    output_path = Path("./zh/rel_ids_2")

    # 检查输入文件是否存在
    input_paths = [
        ent_ids_path,
        en_rel_triples_path,
        triples_path,
    ]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"找不到输入文件：{path}")

    # 1. 读取实体 ID 映射
    entity_id_to_uri = load_entity_id_mapping(ent_ids_path)

    # 2. 读取 URI 关系三元组
    entity_pair_to_relations = load_relation_triples(
        en_rel_triples_path
    )

    # 3. 构建关系 ID 映射
    relation_id_to_uri, unresolved_relation_ids = (
        build_relation_id_mapping(
            triples_path=triples_path,
            entity_id_to_uri=entity_id_to_uri,
            entity_pair_to_relations=entity_pair_to_relations,
        )
    )

    # 4. 写入 rel_ids_2
    write_relation_id_mapping(
        relation_id_to_uri=relation_id_to_uri,
        output_path=output_path,
    )

    print(f"实体数量：{len(entity_id_to_uri)}")
    print(f"成功生成的关系映射数量：{len(relation_id_to_uri)}")
    print(f"未找到映射的关系 ID 数量：{len(unresolved_relation_ids)}")
    if unresolved_relation_ids:
        print("\n未找到映射的关系 ID：")
        for relation_id in sorted(unresolved_relation_ids):
            print(relation_id)
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()