import torch
from torch import nn
import torch.nn.functional as F


class CustomMultiLossLayer(nn.Module):
    """保留 RULE 原始的可学习多损失加权。"""

    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num), requires_grad=True)

    def forward(self, loss_list):
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i, item in enumerate(loss_list):
            loss += precision[i] * item + self.log_vars[i]
        return loss


class RobustEvidentialAlignmentLoss(nn.Module):
    """
    保留 RULE 的 Robust Evidential Alignment Loss：
    Top-K hard negatives -> evidence -> Dirichlet alpha -> robust MSE + KL。

    删除：
    1. uncertainty estimation
    2. consensus
    3. TP/UFP/IFP sample division
    4. warmup-based robust reweighting
    """

    def __init__(self, tau=0.05, top_k=49, lambda2=0.0001):
        super().__init__()
        self.tau = tau
        self.topk = top_k - 1
        self.lambda2 = lambda2

    def KL(self, alpha, c):
        beta = torch.ones((1, c), device=alpha.device, dtype=alpha.dtype)
        s_alpha = torch.sum(alpha, dim=1, keepdim=True)
        s_beta = torch.sum(beta, dim=1, keepdim=True)
        ln_b = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        ln_b_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
        dg0 = torch.digamma(s_alpha)
        dg1 = torch.digamma(alpha)
        return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + ln_b + ln_b_uni

    def robust_mse_loss(self, alpha, label, lambda2=None):
        if lambda2 is None:
            lambda2 = self.lambda2

        s = torch.sum(alpha, dim=1, keepdim=True)
        mean = alpha / s
        error = torch.sum((label - mean) ** 2, dim=1, keepdim=True)
        variance = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)

        evidence = alpha - 1
        adjusted_alpha = evidence * (1 - label) + 1
        kl = lambda2 * self.KL(adjusted_alpha, label.size(1))
        return error + variance + kl

    def get_evidence(self, raw_sims):
        """只生成 evidence / alpha，不再估计 uncertainty。"""
        evidence = torch.exp(torch.tanh(raw_sims) / self.tau)
        return evidence + 1

    def forward(self, emb, train_links, mask=None, epoch=None, divide=False):
        if not torch.is_tensor(train_links):
            train_links = torch.as_tensor(train_links, dtype=torch.long, device=emb[0].device if divide else emb.device)
        else:
            train_links = train_links.to(device=emb[0].device if divide else emb.device, dtype=torch.long)

        if divide:
            zis, zjs = emb
        else:
            # 对缺失模态，只保留 GT 两侧都有效的训练 pair
            if mask is not None:
                mask = mask.to(train_links.device)
                valid = (mask[train_links[:, 0]] > 0) & (mask[train_links[:, 1]] > 0)
                train_links = train_links[valid]

            if train_links.size(0) < 2:
                return emb.sum() * 0.0

            emb = F.normalize(emb, dim=1)
            zis = emb[train_links[:, 0]]
            zjs = emb[train_links[:, 1]]

        if zis.size(0) < 2 or zjs.size(0) < 2:
            ref = zis if torch.is_tensor(zis) else emb
            return ref.sum() * 0.0

        cosine_sims = torch.mm(zis, zjs.t())
        positive_i = torch.diag(cosine_sims).unsqueeze(1)
        positive_j = torch.diag(cosine_sims.t()).unsqueeze(1)

        sim_mask = torch.eye(cosine_sims.size(0), device=cosine_sims.device, dtype=torch.bool)
        cosine_sims_no_diag = cosine_sims.masked_fill(sim_mask, -8.0)

        effective_k = min(self.topk, cosine_sims.size(1) - 1)
        if effective_k > 0:
            negative_i, _ = torch.topk(cosine_sims_no_diag, k=effective_k, dim=1, largest=True)
            negative_j, _ = torch.topk(cosine_sims_no_diag.t(), k=effective_k, dim=1, largest=True)
            raw_sims_i = torch.cat([positive_i, negative_i], dim=1)
            raw_sims_j = torch.cat([positive_j, negative_j], dim=1)
        else:
            raw_sims_i = positive_i
            raw_sims_j = positive_j

        alpha_i = self.get_evidence(raw_sims_i)
        alpha_j = self.get_evidence(raw_sims_j)

        label_i = F.one_hot(torch.zeros(raw_sims_i.size(0), dtype=torch.long, device=raw_sims_i.device), num_classes=raw_sims_i.size(1)).to(raw_sims_i.dtype)
        label_j = F.one_hot(torch.zeros(raw_sims_j.size(0), dtype=torch.long, device=raw_sims_j.device), num_classes=raw_sims_j.size(1)).to(raw_sims_j.dtype)

        loss_i = self.robust_mse_loss(alpha_i, label_i)
        loss_j = self.robust_mse_loss(alpha_j, label_j)

        return (loss_i.mean() + loss_j.mean()) / 2
