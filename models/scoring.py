"""
对比损失 & 推理评分函数。
- symmetric_infonce: 对称 InfoNCE 损失（训练用）
- coarse_prototype_score: 查询-原型相似度（推理粗筛用）
- cosine_retrieval_score: 通用余弦相似度评分
"""
from typing import Union
import torch
import torch.nn.functional as F


def symmetric_infonce(scores: torch.Tensor):
    """
    对称 InfoNCE 损失。
    Args:
        scores: (B, B) — 分数矩阵，对角线为正样本
    Returns:
        loss: scalar
    """
    B = scores.shape[0]
    labels = torch.arange(B, device=scores.device)
    loss_q2n = F.cross_entropy(scores, labels)
    loss_n2q = F.cross_entropy(scores.T, labels)
    return (loss_q2n + loss_n2q) / 2


def coarse_prototype_score(
    adapted_query: torch.Tensor,
    prototypes: torch.Tensor,
):
    """
    查询与原型的余弦相似度（推理粗筛用）。
    Args:
        adapted_query: (B_q, d) — L2归一化的适配后查询
        prototypes: (K, d) — L2归一化的原型
    Returns:
        scores: (B_q, K)
    """
    return torch.matmul(adapted_query, prototypes.T)


def cosine_retrieval_score(
    s_T: torch.Tensor,
    mu: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
):
    """
    推理用评分：余弦相似度 / 温度。
    Args:
        s_T: (B_q, d) — L2归一化的查询向量
        mu: (B_v, d) — L2归一化的视频/原型向量
        temperature: scalar
    Returns:
        scores: (B_q, B_v)
    """
    return torch.matmul(s_T, mu.T) / temperature
