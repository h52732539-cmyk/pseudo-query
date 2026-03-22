"""
Query-Driven Fine-Grained Semantic Assembly:
  1. 查询 token 特征与原型 Cross-Attention
  2. Max-Pooling 沿序列维度聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAssembly(nn.Module):
    """
    给定查询 token 特征和原型库，计算查询激活向量 s_T ∈ R^K。
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_tokens, prototypes, temperature, query_mask=None):
        """
        Args:
            query_tokens: (B, L, d) — 查询文本的 token-level 特征
            prototypes: (K, d) — 原型库
            temperature: scalar — 温度
            query_mask: (B, L) — 1=valid, 0=padding
        Returns:
            s_T: (B, K) — 查询对每个原型的最大激活强度
        """
        # Cross-Attention: query tokens attend to prototypes
        # A_{l,k} = softmax(E_{T,l} · P_k / τ)
        # (B, L, d) x (d, K) → (B, L, K)
        logits = torch.matmul(query_tokens, prototypes.T) / temperature
        A = F.softmax(logits, dim=-1)  # (B, L, K)

        # Mask padding tokens (set to -inf before max)
        if query_mask is not None:
            # (B, L, 1) — 将 padding 位置的激活设为 0
            mask = query_mask.unsqueeze(-1).float()
            A = A * mask  # padding 位置激活为0，不影响 max

        # Max-Pooling 沿序列维度 L
        s_T, _ = A.max(dim=1)  # (B, K)

        # L2 归一化，与 μ 对齐到同一空间
        s_T = F.normalize(s_T, p=2, dim=-1)
        return s_T
