"""Stack three ICEWS image feature PKLs."""
import argparse
import pickle
import numpy as np


def main(args):
    with open(args.pkl_1, "rb") as f:
        features_1 = pickle.load(f)
    with open(args.pkl_2, "rb") as f:
        features_2 = pickle.load(f)
    with open(args.pkl_3, "rb") as f:
        features_3 = pickle.load(f)

    # 修改：取三个 pkl 的并集；三张图都缺失的实体不会出现在任何 pkl 中，因此自然忽略
    all_ids = sorted(set(features_1.keys()) | set(features_2.keys()) | set(features_3.keys()))

    # 修改：少于三张图片时，对缺失位置使用 768 维 0 向量占位
    zero_feature = np.zeros(768, dtype=np.float16)
    features_dict = {}
    for entity_id in all_ids:
        feature_1 = features_1.get(entity_id, zero_feature)
        feature_2 = features_2.get(entity_id, zero_feature)
        feature_3 = features_3.get(entity_id, zero_feature)
        features_dict[entity_id] = np.stack([feature_1, feature_2, feature_3], axis=0)

    with open(args.output_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Complete. Total: {len(features_dict)}. Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack three ICEWS image feature PKLs")
    parser.add_argument("--pkl_1", type=str, required=True)
    parser.add_argument("--pkl_2", type=str, required=True)
    parser.add_argument("--pkl_3", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)