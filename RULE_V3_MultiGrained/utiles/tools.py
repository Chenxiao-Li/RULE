"""Shared utilities for training pure GAT structure embeddings.

This file is self-contained and does NOT import or reuse RULE model code.
It reproduces only the necessary operations for the structure branch:
- reproducible seeding
- entity/pair/triple loading
- sparse graph construction
- trainable entity embeddings
- multi-head sparse GAT
- cosine-based robust alignment loss
- feature export
"""
import os
import math
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Reproducibility ====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== Data Loading ====================

def read_entity_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(line.split("\t", 1)[0]))
    return ids


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left_id, right_id = line.split()[:2]
            pairs.append((int(left_id), int(right_id)))
    return np.asarray(pairs, dtype=np.int64)


def read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            h, r, t = parts[:3]
            triples.append((int(h), int(r), int(t)))
    return triples


# ==================== Graph Construction ====================

def build_edge_index(triples_1, triples_2, ent_num, device):
    """
    Pure structural graph:
    - use only KG triples
    - add reverse edges
    - add self-loops
    """
    edges = set()

    for h, _, t in triples_1 + triples_2:
        edges.add((h, t))
        edges.add((t, h))

    for i in range(ent_num):
        edges.add((i, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long, device=device).t().contiguous()

    return edge_index


# ==================== Sparse GAT ====================

class SparseGraphAttentionLayer(nn.Module):
    """
    Sparse multi-head graph attention.

    For each edge i <- j:
        e_ij = LeakyReLU(a^T [W h_i || W h_j])
        alpha_ij = softmax_j(e_ij)
        h'_i = sum_j alpha_ij W h_j
    """

    def __init__(self, in_dim, out_dim, num_heads=1, attn_dropout=0.0, concat=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.attn_dropout = attn_dropout

        self.weight = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.att_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(num_heads, out_dim))

        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(-1))

    def forward(self, x, edge_index):
        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = x.size(0)

        head_outputs = []

        for head in range(self.num_heads):
            h = torch.matmul(x, self.weight[head])

            src_h = h[src]
            dst_h = h[dst]

            score = ((src_h * self.att_src[head]).sum(dim=-1) + (dst_h * self.att_dst[head]).sum(dim=-1))
            score = F.leaky_relu(score, negative_slope=0.2)

            # Sparse row-wise softmax over incoming neighbors of src.
            sparse_score = torch.sparse_coo_tensor(edge_index, score, size=(num_nodes, num_nodes), device=x.device).coalesce()

            att = torch.sparse.softmax(sparse_score, dim=1).coalesce()
            att_values = F.dropout(att.values(), p=self.attn_dropout, training=self.training)

            neigh = h[att.indices()[1]]

            out = torch.zeros(num_nodes, self.out_dim, dtype=x.dtype, device=x.device)

            out.index_add_(0, att.indices()[0], neigh * att_values.unsqueeze(-1))

            head_outputs.append(out)

        if self.concat:
            return torch.cat(head_outputs, dim=-1)

        return torch.stack(head_outputs, dim=0).mean(dim=0)


class GATStructureEncoder(nn.Module):
    """
    Pure structure encoder:
        trainable entity embedding + graph adjacency -> GAT -> structure embedding
    """

    def __init__(self, ent_num, hidden_units="300,300,300", heads="2,2", dropout=0.4, attn_dropout=0.0):
        super().__init__()

        dims = [int(x) for x in hidden_units.split(",")]
        num_heads = [int(x) for x in heads.split(",")]

        if len(dims) < 2:
            raise ValueError("hidden_units must contain at least two dimensions.")
        if len(num_heads) != len(dims) - 1:
            raise ValueError("The number of values in heads must equal len(hidden_units)-1.")

        self.ent_num = ent_num
        self.dropout = dropout

        # Same initialization principle used by RULE's structure branch.
        self.entity_emb = nn.Embedding(ent_num, dims[0])
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(ent_num))

        layers = []
        for i in range(len(dims) - 1):
            layers.append(SparseGraphAttentionLayer(in_dim=dims[i], out_dim=dims[i + 1], num_heads=num_heads[i], attn_dropout=attn_dropout, concat=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, edge_index):
        x = self.entity_emb.weight

        for i, layer in enumerate(self.layers):
            if i > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = layer(x, edge_index)

            if i < len(self.layers) - 1:
                x = F.elu(x)

        return x


# ==================== Robust Alignment Loss ====================

class RobustAlignmentLoss(nn.Module):
    """
    Structure-only alignment objective inspired by RULE's robust alignment idea.

    For a batch of aligned pairs:
    - L2 normalize embeddings
    - build bidirectional cosine-similarity matrix
    - use positive diagonal + top-k hard negatives
    - evidence-based robust loss with KL regularization
    """

    def __init__(self, tau=0.07, top_k=50, lambda2=1e-4):
        super().__init__()
        self.tau = tau
        self.top_k = top_k
        self.lambda2 = lambda2

    def _kl(self, alpha):
        c = alpha.size(1)
        beta = torch.ones_like(alpha)

        sum_alpha = alpha.sum(dim=1, keepdim=True)
        sum_beta = beta.sum(dim=1, keepdim=True)

        ln_b_alpha = (torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True))
        ln_b_beta = (torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta))

        dg0 = torch.digamma(sum_alpha)
        dg1 = torch.digamma(alpha)

        return (((alpha - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True) + ln_b_alpha + ln_b_beta)

    def _one_direction(self, sims):
        batch_size = sims.size(0)

        pos = torch.diag(sims).unsqueeze(1)

        mask = torch.eye(batch_size, device=sims.device, dtype=torch.bool)

        negatives = sims.masked_fill(mask, float("-inf"))

        effective_k = min(self.top_k - 1, max(batch_size - 1, 1))

        if batch_size > 1:
            neg = torch.topk(negatives, k=effective_k, dim=1, largest=True).values
            raw = torch.cat([pos, neg], dim=1)
        else:
            raw = pos

        evidence = torch.exp(torch.tanh(raw) / self.tau)
        alpha = evidence + 1.0

        label = torch.zeros_like(alpha)
        label[:, 0] = 1.0

        s = alpha.sum(dim=1, keepdim=True)
        mean = alpha / s

        mse = ((label - mean) ** 2).sum(dim=1, keepdim=True)
        variance = (alpha * (s - alpha) / (s * s * (s + 1.0))).sum(dim=1, keepdim=True)

        evidence_only = alpha - 1.0
        adjusted = evidence_only * (1.0 - label) + 1.0
        reg = self.lambda2 * self._kl(adjusted)

        return (mse + variance + reg).mean()

    def forward(self, embeddings, pairs):
        embeddings = F.normalize(embeddings, dim=1)

        left = embeddings[pairs[:, 0]]
        right = embeddings[pairs[:, 1]]

        sims = torch.matmul(left, right.t())

        loss_lr = self._one_direction(sims)
        loss_rl = self._one_direction(sims.t())

        return 0.5 * (loss_lr + loss_rl)


# ==================== Export ====================

def save_feature_pkl(embeddings, entity_ids, output_path):
    embeddings = F.normalize(embeddings, dim=1).detach().cpu().numpy().astype(np.float32)

    feature_dict = {int(entity_id): embeddings[int(entity_id)] for entity_id in entity_ids}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(feature_dict, f)

    print(f"Saved entities: {len(feature_dict)}")
    print(f"Feature dim: {embeddings.shape[1]}")
    print(f"Saved to: {output_path}")
