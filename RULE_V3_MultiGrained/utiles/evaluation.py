"""Reusable GPU evaluation utilities for entity alignment."""
import torch


def l2_normalize(x, eps=1e-12):
    """Row-wise L2 normalization."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def cosine_similarity_matrix(left_features, right_features):
    """Cosine similarity matrix."""
    left_features = l2_normalize(left_features)
    right_features = l2_normalize(right_features)
    return torch.matmul(left_features, right_features.transpose(0, 1))


def csls_similarity(similarity_matrix, k=3, m=2):
    """
    CSLS reweighting:
        CSLS(i,j) = m * sim(i,j) - r_left(i) - r_right(j)
    """
    sim = similarity_matrix.float()
    k_left = min(k, sim.size(1))
    k_right = min(k, sim.size(0))

    r_left = torch.topk(sim, k=k_left, dim=1, largest=True).values.mean(dim=1)
    r_right = torch.topk(sim, k=k_right, dim=0, largest=True).values.mean(dim=0)

    return m * sim - r_left.unsqueeze(1) - r_right.unsqueeze(0)


def get_ranks(similarity_matrix, ground_truth_indices=None):
    """
    Get 1-based ranks by descending sort, consistent with RULE-style evaluation.
    By default, row i is aligned with column i.
    """
    num_queries = similarity_matrix.size(0)
    device = similarity_matrix.device

    if ground_truth_indices is None:
        ground_truth_indices = torch.arange(num_queries, device=device)
    elif not torch.is_tensor(ground_truth_indices):
        ground_truth_indices = torch.tensor(
            ground_truth_indices, dtype=torch.long, device=device
        )
    else:
        ground_truth_indices = ground_truth_indices.to(device)

    indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    matches = indices.eq(ground_truth_indices.unsqueeze(1))
    ranks = torch.argmax(matches.to(torch.int64), dim=1) + 1
    return ranks


def hits_at_k(ranks, k):
    return (ranks <= k).float().mean().item()


def mean_reciprocal_rank(ranks):
    return (1.0 / ranks.float()).mean().item()


def evaluate_similarity(similarity_matrix, ground_truth_indices=None, hits=(1, 5, 10)):
    ranks = get_ranks(similarity_matrix, ground_truth_indices)
    metrics = {f"Hits@{k}": hits_at_k(ranks, k) for k in hits}
    metrics["MRR"] = mean_reciprocal_rank(ranks)
    return metrics, ranks


def print_metrics(title, metrics):
    print(title)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")