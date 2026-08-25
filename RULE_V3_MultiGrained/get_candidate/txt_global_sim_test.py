"""Training-free DBP15K alignment using text + global visual features."""
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

from utiles.evaluation import (
    cosine_similarity_matrix,
    csls_similarity,
    evaluate_similarity,
    print_metrics,
)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def normalize_feature_dict_keys(feature_dict):
    """统一 Entity ID 为 int，避免文本 PKL 的 str key 与图像 PKL 的 int key 不匹配。"""
    return {int(k): np.asarray(v) for k, v in feature_dict.items()}


def load_ref_pairs(file_path):
    """读取 training-free 测试集 ref_pairs。"""
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


def build_joint_feature(entity_id, text_features, image_features, txt_dim, img_dim):
    """
    综合表示：
    1) text 和 global image 各自先 L2 normalize
    2) 有图像 -> [normalized_text ; normalized_global_image]
    3) 缺图像 -> [normalized_text ; zero_image]
    """
    if entity_id not in text_features:
        raise KeyError(f"Missing text feature for Entity ID {entity_id}")

    text_feature = np.asarray(text_features[entity_id], dtype=np.float32).reshape(-1)
    if text_feature.shape[0] != txt_dim:
        raise ValueError(
            f"Unexpected text feature dim for Entity ID {entity_id}: "
            f"{text_feature.shape[0]} != {txt_dim}"
        )
    text_feature = normalize_vector(text_feature)

    if entity_id in image_features:
        image_feature = np.asarray(image_features[entity_id], dtype=np.float32).reshape(-1)
        if image_feature.shape[0] != img_dim:
            raise ValueError(
                f"Unexpected image feature dim for Entity ID {entity_id}: "
                f"{image_feature.shape[0]} != {img_dim}"
            )
        image_feature = normalize_vector(image_feature)
    else:
        # 缺失图片时，视觉部分使用 0 向量占位
        image_feature = np.zeros(img_dim, dtype=np.float32)

    return np.concatenate([text_feature, image_feature], axis=0)


def main(args):
    # GPU 超参数：默认使用 0 号卡
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    dataset_dir = os.path.join(ROOT_DIR, "data", "DBP15K", args.data_split)
    pkl_dir = os.path.join(ROOT_DIR, "data", "pkls", "DBP15K", args.data_split)

    ref_pairs_path = os.path.join(dataset_dir, "ref_pairs")
    text_name = "txt_feature_with_name.pkl" if args.use_name else "txt_feature_without_name.pkl"
    text_path = os.path.join(pkl_dir, text_name)
    image_path = os.path.join(pkl_dir, args.image_name)

    ref_pairs = load_ref_pairs(ref_pairs_path)
    text_features = normalize_feature_dict_keys(load_pickle(text_path))
    image_features = normalize_feature_dict_keys(load_pickle(image_path))

    left_features = []
    right_features = []
    valid_pairs = []
    missing_text = 0
    left_missing_image = 0
    right_missing_image = 0

    for left_id, right_id in ref_pairs:
        if left_id not in text_features or right_id not in text_features:
            missing_text += 1
            continue

        if left_id not in image_features:
            left_missing_image += 1
        if right_id not in image_features:
            right_missing_image += 1

        left_features.append(
            build_joint_feature(
                left_id, text_features, image_features,
                args.txt_dim, args.img_dim
            )
        )
        right_features.append(
            build_joint_feature(
                right_id, text_features, image_features,
                args.txt_dim, args.img_dim
            )
        )
        valid_pairs.append((left_id, right_id))

    if not valid_pairs:
        raise ValueError("No valid ref_pairs can be evaluated.")

    # 综合特征放到指定 GPU；后续 cosine / CSLS / ranking 全部在 GPU 上运行
    left_features = torch.from_numpy(np.stack(left_features)).float().to(device)
    right_features = torch.from_numpy(np.stack(right_features)).float().to(device)

    print(f"Device: {device}")
    print(f"Dataset: {args.data_split}")
    print(f"Use name: {args.use_name}")
    print(f"CSLS: {args.csls}")
    if args.csls:
        print(f"CSLS k: {args.csls_k}")
        print(f"CSLS m: {args.m_csls}")
    print(f"Test pairs (ref_pairs): {len(ref_pairs)}")
    print(f"Valid test pairs: {len(valid_pairs)}")
    print(f"Skipped pairs due to missing text: {missing_text}")
    print(f"KG1 entities missing global image: {left_missing_image}")
    print(f"KG2 entities missing global image: {right_missing_image}")
    print(f"Text feature dim: {args.txt_dim}")
    print(f"Global image feature dim: {args.img_dim}")
    print(f"Joint feature dim: {args.txt_dim + args.img_dim}")

    # text/image 已分别归一化；concat 后再整体归一化并计算 cosine
    sim_lr = cosine_similarity_matrix(left_features, right_features)

    # CSLS 默认开启，参数与 RULE 一致：k=3, m=2
    if args.csls:
        sim_lr = csls_similarity(
            sim_lr,
            k=args.csls_k,
            m=args.m_csls
        )

    metrics_lr, _ = evaluate_similarity(sim_lr)
    metrics_rl, _ = evaluate_similarity(sim_lr.transpose(0, 1))

    print()
    print_metrics("KG1 -> KG2", metrics_lr)

    print()
    print_metrics("KG2 -> KG1", metrics_rl)

    avg_metrics = {
        key: (metrics_lr[key] + metrics_rl[key]) / 2.0
        for key in metrics_lr
    }

    print()
    print_metrics("Average", avg_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training-free DBP15K alignment with text + global image features"
    )
    parser.add_argument(
        "--data_split",
        default="zh_en",
        type=str,
        choices=["zh_en", "ja_en", "fr_en"]
    )
    parser.add_argument("--use_name", action="store_true")
    parser.add_argument("--txt_dim", default=768, type=int)
    parser.add_argument("--img_dim", default=768, type=int)
    parser.add_argument("--image_name", default="img_feature_global.pkl", type=str)
    parser.add_argument("--gpu", default=0, type=int)

    # 默认开启 CSLS；如果需要纯 cosine baseline，可传 --no-csls
    parser.add_argument(
        "--csls",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument("--csls_k", default=3, type=int)
    parser.add_argument("--m_csls", default=2, type=int)

    args = parser.parse_args()
    main(args)