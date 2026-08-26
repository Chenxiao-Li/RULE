"""Generate pure GAT structure embeddings for DBP15K (V2)."""
import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 当前脚本位于 RULE/get_attribute_embedding/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utiles.tools import set_seed, read_entity_ids, read_pairs, read_triples, uild_edge_index, GATStructureEncoder, RobustAlignmentLoss, save_feature_pkl

def main(args):
    set_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    dataset_dir = os.path.join(ROOT_DIR, "data", "DBP15K", args.data_split)
    output_dir = os.path.join(ROOT_DIR, "data", "pkls", "DBP15K", args.data_split)

    ent_ids_1 = read_entity_ids(os.path.join(dataset_dir, "ent_ids_1"))
    ent_ids_2 = read_entity_ids(os.path.join(dataset_dir, "ent_ids_2"))

    entity_ids = sorted(set(ent_ids_1 + ent_ids_2))
    ent_num = max(entity_ids) + 1

    triples_1 = read_triples(os.path.join(dataset_dir, args.triples_1))
    triples_2 = read_triples(os.path.join(dataset_dir, args.triples_2))
    sup_pairs = read_pairs(os.path.join(dataset_dir, "sup_pairs"))

    edge_index = build_edge_index(triples_1, triples_2, ent_num, device)

    model = GATStructureEncoder(ent_num=ent_num, hidden_units=args.hidden_units, heads=args.heads, dropout=args.dropout, attn_dropout=args.attn_dropout).to(device)

    criterion = RobustAlignmentLoss(tau=args.tau, top_k=args.topk, lambda2=args.lambda2)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_tensor = torch.from_numpy(np.asarray(sup_pairs, dtype=np.int64)).long()

    loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True, drop_last=False)

    print(f"Device: {device}")
    print(f"Dataset: {args.data_split}")
    print(f"Entities: {len(entity_ids)}")
    print(f"Triples KG1: {len(triples_1)}")
    print(f"Triples KG2: {len(triples_2)}")
    print(f"Training pairs: {len(sup_pairs)}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Heads: {args.heads}")

    for epoch in range(args.epoch):
        model.train()

        total_loss = 0.0
        num_batches = 0

        for (pairs,) in loader:
            pairs = pairs.to(device)

            optimizer.zero_grad()

            structure_emb = model(edge_index)
            loss = criterion(structure_emb, pairs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        print(
            f"Epoch {epoch + 1:02d}/{args.epoch} | "
            f"structure loss: {avg_loss:.6f}"
        )

    # 不使用 ref_pairs 做模型选择，避免测试集泄漏
    model.eval()

    with torch.no_grad():
        structure_emb = model(edge_index)

    output_path = os.path.join(output_dir, args.output_name)

    save_feature_pkl(structure_emb, entity_ids, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pure GAT structure embeddings for DBP15K (V2)")
    parser.add_argument("--data_split", default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--triples_1", default="triples_1", type=str)
    parser.add_argument("--triples_2", default="triples_2", type=str)
    parser.add_argument("--output_name", default="structure_feature.pkl", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--clip", default=1.1, type=float)
    parser.add_argument("--random_seed", default=3408, type=int)
    parser.add_argument("--hidden_units", default="300,300,300", type=str)
    parser.add_argument("--heads", default="2,2", type=str)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--tau", default=0.07, type=float)
    parser.add_argument("--topk", default=50, type=int)
    parser.add_argument("--lambda2", default=0.0001, type=float)

    args = parser.parse_args()
    main(args)