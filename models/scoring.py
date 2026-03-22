"""
不确定性感知匹配评分 & 损失计算。
- Cosine similarity + 可学习温度
- Free-bits KL 正则
- 方差惩罚已停用（代码保留）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def uncertainty_aware_score(s_T, mu, sigma_sq=None, variance_penalty: float = 0.1, temperature: float = 1.0):
    """
    Score(T, V_i) = cosine_sim(s_T, μ_i) / τ
    方差惩罚已停用（保留代码）。

    Args:
        s_T: (B_q, K) — 已 L2 归一化的查询激活向量
        mu: (B_v, K) — 已 L2 归一化的视频均值向量
        sigma_sq: (B_v, K) — 视频方差向量（当前未使用）
        variance_penalty: λ（当前未使用）
        temperature: τ — 温度缩放
    Returns:
        scores: (B_q, B_v)
    """
    # Cosine similarity（s_T 和 mu 已经 L2 归一化，点积即 cosine）
    match_score = torch.matmul(s_T, mu.T) / temperature  # (B_q, B_v)

    # 方差惩罚（停用，保留代码）
    # penalty = torch.matmul(s_T, sigma_sq.T)
    # return match_score - variance_penalty * penalty

    return match_score


def info_nce_loss(scores):
    """
    对称 InfoNCE loss（双向: text→video + video→text）。
    支持非方阵输入（Memory Bank 场景）。
    Args:
        scores: (B, B+Q) — 前 B 列对角线为正样本
    Returns:
        loss: scalar
    """
    B = scores.shape[0]
    labels = torch.arange(B, device=scores.device)
    # Text → Video
    loss_t2v = F.cross_entropy(scores, labels)
    # Video → Text（仅取 B×B 子矩阵）
    loss_v2t = F.cross_entropy(scores[:B, :B].T, labels)
    return (loss_t2v + loss_v2t) / 2


def kl_divergence(mu, sigma_sq, free_bits: float = 0.0):
    """
    KL(N(μ, Σ) || N(0, I))  闭式解 + Free-bits。
    Args:
        mu: (B, K)
        sigma_sq: (B, K) — 方差（非标准差）
        free_bits: 每维 KL 低于此阈值不惩罚
    Returns:
        kl: scalar (batch 均值)
    """
    # D_KL = 0.5 * Σ_k (σ²_k + μ²_k - 1 - ln(σ²_k))
    kl_per_dim = 0.5 * (sigma_sq + mu.pow(2) - 1 - torch.log(sigma_sq + 1e-8))  # (B, K)

    if free_bits > 0:
        # Free-bits: 每个维度的 KL 超过阈值才惩罚
        kl_per_dim = F.relu(kl_per_dim - free_bits)

    return kl_per_dim.sum(dim=-1).mean()
