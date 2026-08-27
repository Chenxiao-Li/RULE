"""DBP15K entity alignment using text, global visual, and structure features."""
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
    """统一 Entity ID 为 int。"""
    return {int(k): np.asarray(v) for k, v in feature_dict.items()}


def load_ref_pairs(file_path):
    """读取测试集 ref_pairs。"""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_id, right_id = line.split()
            pairs.append((int(left_id), int(right_id)))
    return pairs


def normalize_vector(x, eps=1e-12):
    """单个模态先做 L2 normalization；0 向量保持为 0。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x)
    if norm <= eps:
        return x
    return x / norm


def get_feature_or_zero(entity_id, feature_dict, dim, modality_name, allow_missing=False):
    """
    读取并归一化单个模态。
    allow_missing=True 时，缺失特征用 0 向量占位。
    """
    if entity_id not in feature_dict:
        if allow_missing:
            return np.zeros(dim, dtype=np.float32)
        raise KeyError(f"Missing {modality_name} feature for Entity ID {entity_id}")

    feature = np.asarray(feature_dict[entity_id], dtype=np.float32).reshape(-1)

    if feature.shape[0] != dim:
        raise ValueError(
            f"Unexpected {modality_name} feature dim for Entity ID {entity_id}: "
            f"{feature.shape[0]} != {dim}"
        )

    return normalize_vector(feature)


def build_joint_feature(entity_id, text_features, image_features, structure_features, args):
    """
    根据三个模态超参数动态构造实体表示。
    每个启用的模态先独立 L2 normalize，然后 concat。

    - text:      自动从当前 PKL 检测维度
    - global:    768-d；缺图时用 0 向量
    - structure: 300-d
    """
    features = []

    if args.use_text:
        features.append(get_feature_or_zero(entity_id, text_features, args.text_dim, "text", allow_missing=False))

    if args.use_global:
        features.append(get_feature_or_zero( entity_id, image_features, args.img_dim, "global image", allow_missing=True))

    if args.use_structure:
        features.append(get_feature_or_zero(entity_id, structure_features, args.structure_dim, "structure", allow_missing=False))

    if not features:
        raise ValueError("At least one modality must be enabled.")

    return np.concatenate(features, axis=0)


def main(args):
    if not (args.use_text or args.use_global or args.use_structure):
        raise ValueError("At least one modality must be enabled: " "--use_text / --use_global / --use_structure")

    # GPU 超参数：默认使用 0 号卡
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    dataset_dir = os.path.join(ROOT_DIR, "data", "DBP15K", args.data_split)
    pkl_dir = os.path.join(ROOT_DIR, "data", "pkls", "DBP15K", args.data_split)

    ref_pairs_path = os.path.join(dataset_dir, "ref_pairs")

    # 仅在相应模态启用时读取对应 PKL
    text_features = {}
    image_features = {}
    structure_features = {}

    if args.use_text:
        text_name = ("txt_feature_with_surface.pkl" if args.use_surface else "txt_feature_without_surface.pkl")
        text_path = os.path.join(pkl_dir, text_name)
        text_features = normalize_feature_dict_keys(load_pickle(text_path))

        if not text_features:
            raise ValueError(f"Text feature file is empty: {text_path}")

        args.text_dim = len(next(iter(text_features.values())))
    else:
        args.text_dim = 0

    if args.use_global:
        image_path = os.path.join(pkl_dir, args.image_name)
        image_features = normalize_feature_dict_keys(load_pickle(image_path))

    if args.use_structure:
        structure_path = os.path.join(pkl_dir, args.structure_name)
        structure_features = normalize_feature_dict_keys(load_pickle(structure_path))

    ref_pairs = load_ref_pairs(ref_pairs_path)

    left_features = []
    right_features = []
    valid_pairs = []

    missing_text = 0
    missing_structure = 0
    left_missing_image = 0
    right_missing_image = 0

    for left_id, right_id in ref_pairs:
        # text / structure 是实体级特征，启用时若缺失则跳过 pair
        if args.use_text and (left_id not in text_features or right_id not in text_features):
            missing_text += 1
            continue

        if args.use_structure and (left_id not in structure_features or right_id not in structure_features):
            missing_structure += 1
            continue

        # global image 缺失时不跳过，用 0 向量占位
        if args.use_global:
            if left_id not in image_features:
                left_missing_image += 1
            if right_id not in image_features:
                right_missing_image += 1

        left_features.append(build_joint_feature(left_id, text_features, image_features, structure_features, args))
        right_features.append(build_joint_feature(right_id, text_features, image_features, structure_features, args))
        valid_pairs.append((left_id, right_id))

    if not valid_pairs:
        raise ValueError("No valid ref_pairs can be evaluated.")

    # 后续 cosine / CSLS / ranking 全部放 GPU
    left_features = torch.from_numpy(np.stack(left_features)).float().to(device)
    right_features = torch.from_numpy(np.stack(right_features)).float().to(device)
    text_dim = args.text_dim if args.use_text else 0
    joint_dim = text_dim + (args.img_dim if args.use_global else 0) + (args.structure_dim if args.use_structure else 0)

    print(f"Device: {device}")
    print(f"Dataset: {args.data_split}")
    print(f"Use text: {args.use_text}")
    print(f"Use global image: {args.use_global}")
    print(f"Use structure: {args.use_structure}")
    if args.use_text:
        print(f"Use surface: {args.use_surface}")
    print(f"CSLS: {args.csls}")

    if args.csls:
        print(f"CSLS k: {args.csls_k}")
        print(f"CSLS m: {args.m_csls}")

    print(f"Test pairs (ref_pairs): {len(ref_pairs)}")
    print(f"Valid test pairs: {len(valid_pairs)}")

    if args.use_text:
        print(f"Skipped pairs due to missing text: {missing_text}")
        print(f"Text feature dim: {args.text_dim}")

    if args.use_global:
        print(f"KG1 entities missing global image: {left_missing_image}")
        print(f"KG2 entities missing global image: {right_missing_image}")
        print(f"Global image feature dim: {args.img_dim}")

    if args.use_structure:
        print(f"Skipped pairs due to missing structure: {missing_structure}")
        print(f"Structure feature dim: {args.structure_dim}")

    print(f"Joint feature dim: {joint_dim}")

    # 各模态已经分别归一化；concat 后再次整体归一化并计算 cosine
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
    parser = argparse.ArgumentParser(description="DBP15K alignment with text, global image, and structure features")
    parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--use_text", action="store_false", default=True)
    parser.add_argument("--use_global", action="store_false", default=True)
    parser.add_argument("--use_structure", action="store_false", default=True)
    parser.add_argument("--use_surface", action="store_true", default=False)
    parser.add_argument("--img_dim", default=768, type=int)
    parser.add_argument("--structure_dim", default=300, type=int)
    parser.add_argument("--image_name", default="img_feature_global.pkl", type=str)
    parser.add_argument("--structure_name", default="structure_feature.pkl", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--csls", action="store_false", default=True)
    parser.add_argument("--csls_k", default=3, type=int)
    parser.add_argument("--m_csls", default=2, type=int)

    args = parser.parse_args()
    main(args)