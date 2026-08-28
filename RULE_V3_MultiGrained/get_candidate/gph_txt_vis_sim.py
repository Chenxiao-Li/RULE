"""DBP15K entity alignment using graph, text, global visual, and local visual features."""
import os
import sys
import pickle
import argparse

import numpy as np
import torch


# 当前脚本位于 RULE/get_candidate/，将 RULE 根目录加入 Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utiles.evaluation import cosine_similarity_matrix, csls_similarity, evaluate_similarity, print_metrics


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def normalize_feature_dict_keys(feature_dict):
    """统一 Entity ID 为 int，并统一为 float32。"""
    return {int(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in feature_dict.items()}


def load_ref_pairs(file_path):
    """读取测试集 ref_pairs。"""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_id, right_id = line.split()[:2]
            pairs.append((int(left_id), int(right_id)))
    return pairs


def normalize_vector(x, eps=1e-12):
    """单个向量做 L2 normalization；0 向量保持为 0。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x)
    if norm <= eps:
        return x
    return x / norm


def get_feature_or_zero(entity_id, feature_dict, dim, modality_name):
    """读取模态特征；缺失时返回同维度 0 向量。"""
    if entity_id not in feature_dict:
        return np.zeros(dim, dtype=np.float32)
    feature = np.asarray(feature_dict[entity_id], dtype=np.float32).reshape(-1)
    if feature.shape[0] != dim:
        raise ValueError(f"Unexpected {modality_name} feature dim for Entity ID {entity_id}: {feature.shape[0]} != {dim}")
    return normalize_vector(feature)


def get_optional_feature(entity_id, feature_dict, dim, modality_name):
    """读取可缺失视觉模态；缺失时返回 None。"""
    if entity_id not in feature_dict:
        return None
    feature = np.asarray(feature_dict[entity_id], dtype=np.float32).reshape(-1)
    if feature.shape[0] != dim:
        raise ValueError(f"Unexpected {modality_name} feature dim for Entity ID {entity_id}: {feature.shape[0]} != {dim}")
    return normalize_vector(feature)


def get_feature_with_mask(entity_id, feature_dict, dim, modality_name):
    """返回归一化特征和存在标记；缺失时返回 0 向量和 False。"""
    if entity_id not in feature_dict:
        return np.zeros(dim, dtype=np.float32), False
    feature = np.asarray(feature_dict[entity_id], dtype=np.float32).reshape(-1)
    if feature.shape[0] != dim:
        raise ValueError(f"Unexpected {modality_name} feature dim for Entity ID {entity_id}: {feature.shape[0]} != {dim}")
    return normalize_vector(feature), True


def build_modality_matrices(entity_ids, feature_dict, dim, modality_name, device):
    """构造某一模态的特征矩阵和实体可用性 mask。"""
    features = []
    masks = []
    for entity_id in entity_ids:
        feature, exists = get_feature_with_mask(entity_id, feature_dict, dim, modality_name)
        features.append(feature)
        masks.append(exists)
    features = torch.from_numpy(np.stack(features)).float().to(device)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)
    return features, masks


def masked_multimodal_similarity(left_ids, right_ids, text_features, structure_features, global_features, local_features, args, device):
    """各模态独立算 cosine；仅双方都存在的模态参与，最后对有效模态取平均。"""
    modalities = [
        ("text", text_features),
        ("structure", structure_features),
        ("global", global_features),
        ("local", local_features),
    ]

    sim_sum = torch.zeros((len(left_ids), len(right_ids)), dtype=torch.float32, device=device)
    valid_count = torch.zeros_like(sim_sum)

    for modality_name, feature_dict in modalities:
        left_matrix, left_mask = build_modality_matrices(left_ids, feature_dict, args.feature_dim, modality_name, device)
        right_matrix, right_mask = build_modality_matrices(right_ids, feature_dict, args.feature_dim, modality_name, device)
        modality_sim = torch.matmul(left_matrix, right_matrix.transpose(0, 1))
        pair_mask = left_mask[:, None] & right_mask[None, :]
        sim_sum += modality_sim * pair_mask.float()
        valid_count += pair_mask.float()

    # 极端情况下双方没有任何共同模态，相似度设为 0。
    return torch.where(valid_count > 0, sim_sum / valid_count.clamp_min(1.0), torch.zeros_like(sim_sum))


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    dataset_dir = os.path.join(ROOT_DIR, "data", "DBP15K", args.data_split)
    pkl_dir = os.path.join(ROOT_DIR, "data", "pkls", "DBP15K", args.data_split)
    ref_pairs_path = os.path.join(dataset_dir, "ref_pairs")

    text_path = os.path.join(pkl_dir, args.text_name)
    structure_path = os.path.join(pkl_dir, args.structure_name)
    global_path = os.path.join(pkl_dir, args.global_name)
    local_path = os.path.join(pkl_dir, args.local_name)

    text_features = normalize_feature_dict_keys(load_pickle(text_path))
    structure_features = normalize_feature_dict_keys(load_pickle(structure_path))
    global_features = normalize_feature_dict_keys(load_pickle(global_path))
    local_features = normalize_feature_dict_keys(load_pickle(local_path))
    ref_pairs = load_ref_pairs(ref_pairs_path)

    if text_features:
        text_dim = len(next(iter(text_features.values())))
        if text_dim != args.feature_dim:
            raise ValueError(f"Text feature dim must be {args.feature_dim}, but got {text_dim}. This script requires text and graph to share the CLIP-space dimension.")

    if structure_features:
        structure_dim = len(next(iter(structure_features.values())))
        if structure_dim != args.feature_dim:
            raise ValueError(f"Structure feature dim must be {args.feature_dim}, but got {structure_dim}. Please use the regenerated 768-d structure feature.")

    left_ids = [left_id for left_id, _ in ref_pairs]
    right_ids = [right_id for _, right_id in ref_pairs]

    missing_text_left = sum(entity_id not in text_features for entity_id in left_ids)
    missing_text_right = sum(entity_id not in text_features for entity_id in right_ids)
    missing_structure_left = sum(entity_id not in structure_features for entity_id in left_ids)
    missing_structure_right = sum(entity_id not in structure_features for entity_id in right_ids)
    missing_global_left = sum(entity_id not in global_features for entity_id in left_ids)
    missing_global_right = sum(entity_id not in global_features for entity_id in right_ids)
    missing_local_left = sum(entity_id not in local_features for entity_id in left_ids)
    missing_local_right = sum(entity_id not in local_features for entity_id in right_ids)
    local_left_count = len(left_ids) - missing_local_left
    local_right_count = len(right_ids) - missing_local_right
    aligned_both_local = sum(left_id in local_features and right_id in local_features for left_id, right_id in ref_pairs)

    print(f"Device: {device}")
    print(f"Dataset: {args.data_split}")
    print(f"Text feature: {text_path}")
    print(f"Structure feature: {structure_path}")
    print(f"Global visual feature: {global_path}")
    print(f"Local visual feature: {local_path}")
    print(f"Feature dim per modality: {args.feature_dim}")
    print(f"Test pairs: {len(ref_pairs)}")
    print(f"Missing text in KG1: {missing_text_left}")
    print(f"Missing text in KG2: {missing_text_right}")
    print(f"Missing structure in KG1: {missing_structure_left}")
    print(f"Missing structure in KG2: {missing_structure_right}")
    print(f"Missing global in KG1: {missing_global_left}")
    print(f"Missing global in KG2: {missing_global_right}")
    print(f"Missing local in KG1: {missing_local_left}")
    print(f"Missing local in KG2: {missing_local_right}")
    print(f"KG1 entities with local: {local_left_count}")
    print(f"KG2 entities with local: {local_right_count}")
    print(f"Aligned pairs with local on both sides: {aligned_both_local}")

    sim_lr = masked_multimodal_similarity(left_ids, right_ids, text_features, structure_features, global_features, local_features, args, device)

    if args.csls:
        sim_lr = csls_similarity(sim_lr, k=args.csls_k, m=args.m_csls)

    metrics_lr, ranks_lr = evaluate_similarity(sim_lr)
    metrics_rl, ranks_rl = evaluate_similarity(sim_lr.transpose(0, 1))

    print()
    print_metrics("KG1 -> KG2", metrics_lr)

    print()
    print_metrics("KG2 -> KG1", metrics_rl)

    avg_metrics = {key: (metrics_lr[key] + metrics_rl[key]) / 2.0 for key in metrics_lr}

    print()
    print_metrics("Average", avg_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBP15K masked multimodal similarity fusion")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--text_name", default="txt_feature_without_surface.pkl", type=str)
    parser.add_argument("--structure_name", default="structure_feature.pkl", type=str)
    parser.add_argument("--global_name", default="img_feature_global.pkl", type=str)
    parser.add_argument("--local_name", default="img_feature_local_without_name.pkl", type=str)
    parser.add_argument("--feature_dim", default=768, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--csls", action="store_false", default=True)
    parser.add_argument("--csls_k", default=3, type=int)
    parser.add_argument("--m_csls", default=2, type=int)
    args = parser.parse_args()
    main(args)
