"""
逐原型细粒度对比损失 & 推理评分。
- prototype_anchored_contrastive_loss: 训练用，逐原型对比 h vs q̃
- cosine_retrieval_score: 推理用，全局余弦 s_T vs μ
"""
from typing import Union
import torch
import torch.nn.functional as F


def prototype_anchored_contrastive_loss(
    h: torch.Tensor,
    q_tilde: torch.Tensor,
    s_T: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
):
    """
    逐原型细粒度对比损失（对称 InfoNCE）。

    在每个原型 k 的 R^d 空间中独立计算视频-查询相似度，
    然后以查询激活权重 s_T 加权求和得到最终对比分数。

    Args:
        h: (B, K, d) — 视频侧逐原型聚合特征（L2归一化前）
        q_tilde: (B, K, d) — 查询侧逐原型聚合语义（已 L2 归一化）
        s_T: (B, K) — 查询激活权重
        temperature: scalar
    Returns:
        loss: scalar — 对称 InfoNCE
    """
    B = h.shape[0]

    # L2 归一化
    h_norm = F.normalize(h, p=2, dim=-1)       # (B, K, d)
    q_norm = F.normalize(q_tilde, p=2, dim=-1)  # (B, K, d)

    # 逐原型相似度矩阵: (B, B, K)
    # sim[i, j, k] = cosine(h_i_k, q_j_k)
    sim_per_proto = torch.einsum("bkd, ckd -> bck", h_norm, q_norm)  # (B, B, K)

    # 加权求和: w_ik * sim[i, j, k] → (B, B)
    w = F.normalize(s_T, p=1, dim=-1)  # (B, K), L1 归一化
    scores = torch.einsum("bck, ck -> bc", sim_per_proto, w) / temperature  # (B, B)

    labels = torch.arange(B, device=scores.device)
    # Text → Video
    loss_t2v = F.cross_entropy(scores, labels)
    # Video → Text
    loss_v2t = F.cross_entropy(scores.T, labels)
    return (loss_t2v + loss_v2t) / 2


def cosine_retrieval_score(
    s_T: torch.Tensor,
    mu: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
):
    """
    推理用评分：余弦相似度 / 温度。

    Args:
        s_T: (B_q, K) — L2归一化的查询激活向量
        mu: (B_v, K) — L2归一化的视频均值向量
        temperature: scalar
    Returns:
        scores: (B_q, B_v)
    """
    return torch.matmul(s_T, mu.T) / temperature
