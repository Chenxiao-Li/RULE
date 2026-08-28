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


def get_required_feature(entity_id, feature_dict, dim, modality_name):
    """读取必须存在的模态，并检查维度。"""
    if entity_id not in feature_dict:
        raise KeyError(f"Missing {modality_name} feature for Entity ID {entity_id}")
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


def build_entity_feature(entity_id, text_features, structure_features, global_features, local_features, args):
    """
    构造最终实体表示。

    T, S, G, L 均为 768 维并分别 L2-normalize。

    K = normalize(T + S)

    当 G、L 都存在：
        SG = cosine(G, K)
        SL = cosine(L, K)
        V = [SG * G ; SL * L]

    只有 G：
        V = [G ; 0]

    只有 L：
        V = [0 ; L]

    G、L 都缺失：
        V = [0 ; 0]

    Final = [T ; S ; V]
    """
    text = get_required_feature(entity_id, text_features, args.feature_dim, "text")
    structure = get_required_feature(entity_id, structure_features, args.feature_dim, "structure")
    global_visual = get_optional_feature(entity_id, global_features, args.feature_dim, "global visual")
    local_visual = get_optional_feature(entity_id, local_features, args.feature_dim, "local visual")

    # text + graph 形成统一语义-结构参考表示。
    knowledge = normalize_vector(text + structure)
    zero_visual = np.zeros(args.feature_dim, dtype=np.float32)

    if global_visual is not None and local_visual is not None:
        sim_global = float(np.dot(global_visual, knowledge))
        sim_local = float(np.dot(local_visual, knowledge))
        weighted_global = sim_global * global_visual
        weighted_local = sim_local * local_visual
        visual = np.concatenate([weighted_global, weighted_local], axis=0)
        visual_case = "both"
    elif global_visual is not None:
        sim_global = None
        sim_local = None
        visual = np.concatenate([global_visual, zero_visual], axis=0)
        visual_case = "global_only"
    elif local_visual is not None:
        sim_global = None
        sim_local = None
        visual = np.concatenate([zero_visual, local_visual], axis=0)
        visual_case = "local_only"
    else:
        sim_global = None
        sim_local = None
        visual = np.concatenate([zero_visual, zero_visual], axis=0)
        visual_case = "none"

    final_feature = np.concatenate([text, structure, visual], axis=0).astype(np.float32)
    return final_feature, visual_case, sim_global, sim_local


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

    if not text_features:
        raise ValueError(f"Text feature file is empty: {text_path}")
    if not structure_features:
        raise ValueError(f"Structure feature file is empty: {structure_path}")

    text_dim = len(next(iter(text_features.values())))
    structure_dim = len(next(iter(structure_features.values())))

    if text_dim != args.feature_dim:
        raise ValueError(f"Text feature dim must be {args.feature_dim}, but got {text_dim}. This script requires text and graph to share the CLIP-space dimension.")
    if structure_dim != args.feature_dim:
        raise ValueError(f"Structure feature dim must be {args.feature_dim}, but got {structure_dim}. Please use the regenerated 768-d structure feature.")

    left_features = []
    right_features = []
    valid_pairs = []

    missing_text = 0
    missing_structure = 0
    visual_counts_left = {"both": 0, "global_only": 0, "local_only": 0, "none": 0}
    visual_counts_right = {"both": 0, "global_only": 0, "local_only": 0, "none": 0}
    sim_global_values = []
    sim_local_values = []

    for left_id, right_id in ref_pairs:
        if left_id not in text_features or right_id not in text_features:
            missing_text += 1
            continue

        if left_id not in structure_features or right_id not in structure_features:
            missing_structure += 1
            continue

        left_feature, left_case, left_sg, left_sl = build_entity_feature(left_id, text_features, structure_features, global_features, local_features, args)
        right_feature, right_case, right_sg, right_sl = build_entity_feature(right_id, text_features, structure_features, global_features, local_features, args)

        left_features.append(left_feature)
        right_features.append(right_feature)
        valid_pairs.append((left_id, right_id))

        visual_counts_left[left_case] += 1
        visual_counts_right[right_case] += 1

        if left_sg is not None:
            sim_global_values.append(left_sg)
            sim_local_values.append(left_sl)
        if right_sg is not None:
            sim_global_values.append(right_sg)
            sim_local_values.append(right_sl)

    if not valid_pairs:
        raise ValueError("No valid ref_pairs can be evaluated.")

    left_features = torch.from_numpy(np.stack(left_features)).float().to(device)
    right_features = torch.from_numpy(np.stack(right_features)).float().to(device)

    final_dim = args.feature_dim * 4

    print(f"Device: {device}")
    print(f"Dataset: {args.data_split}")
    print(f"Text feature: {text_path}")
    print(f"Structure feature: {structure_path}")
    print(f"Global visual feature: {global_path}")
    print(f"Local visual feature: {local_path}")
    print(f"Feature dim per modality: {args.feature_dim}")
    print(f"Final feature dim: {final_dim}")
    print(f"Test pairs: {len(ref_pairs)}")
    print(f"Valid test pairs: {len(valid_pairs)}")
    print(f"Skipped pairs due to missing text: {missing_text}")
    print(f"Skipped pairs due to missing structure: {missing_structure}")

    print("\nKG1 visual availability:")
    print(f"  both: {visual_counts_left['both']}")
    print(f"  global only: {visual_counts_left['global_only']}")
    print(f"  local only: {visual_counts_left['local_only']}")
    print(f"  none: {visual_counts_left['none']}")

    print("\nKG2 visual availability:")
    print(f"  both: {visual_counts_right['both']}")
    print(f"  global only: {visual_counts_right['global_only']}")
    print(f"  local only: {visual_counts_right['local_only']}")
    print(f"  none: {visual_counts_right['none']}")

    if sim_global_values:
        print(f"\nBoth-visual entities used for weighting: {len(sim_global_values)}")
        print(f"Mean SG: {np.mean(sim_global_values):.6f}")
        print(f"Mean SL: {np.mean(sim_local_values):.6f}")

    sim_lr = cosine_similarity_matrix(left_features, right_features)

    if args.csls:
        sim_lr = csls_similarity(sim_lr, k=args.csls_k, m=args.m_csls)

    metrics_lr, _ = evaluate_similarity(sim_lr)
    metrics_rl, _ = evaluate_similarity(sim_lr.transpose(0, 1))

    print()
    print_metrics("KG1 -> KG2", metrics_lr)

    print()
    print_metrics("KG2 -> KG1", metrics_rl)

    avg_metrics = {key: (metrics_lr[key] + metrics_rl[key]) / 2.0 for key in metrics_lr}

    print()
    print_metrics("Average", avg_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBP15K graph-text-visual similarity fusion")
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
