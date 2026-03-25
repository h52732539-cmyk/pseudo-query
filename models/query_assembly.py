"""
Query-Driven Fine-Grained Semantic Assembly:
  1. 查询 token 特征与原型 Cross-Attention
  2. Max-Pooling 沿序列维度聚合 → s_T (B, K)
  3. 注意力加权聚合 → q_tilde (B, K, d)，用于逐原型对比损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAssembly(nn.Module):
    """
    给定查询 token 特征和原型库，计算：
    - s_T ∈ R^K — 查询对每个原型的最大激活强度（推理用）
    - q_tilde ∈ R^{K×d} — 查询 token 按注意力加权聚合到每个原型上的语义向量（训练用）
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
            q_tilde: (B, K, d) — 逐原型聚合的查询语义向量
        """
        # Cross-Attention: query tokens attend to prototypes
        # (B, L, d) x (d, K) → (B, L, K)
        logits = torch.matmul(query_tokens, prototypes.T) / temperature
        A = F.softmax(logits, dim=-1)  # (B, L, K)

        # Mask padding tokens
        if query_mask is not None:
            mask = query_mask.unsqueeze(-1).float()  # (B, L, 1)
            A = A * mask  # padding 位置激活为0

        # q_tilde: 注意力加权聚合查询 token 到每个原型
        # (B, K, L) @ (B, L, d) → (B, K, d)
        q_tilde = torch.bmm(A.transpose(1, 2), query_tokens)
        q_tilde = F.normalize(q_tilde, p=2, dim=-1)

        # Max-Pooling 沿序列维度 L → s_T
        s_T, _ = A.max(dim=1)  # (B, K)
        s_T = F.normalize(s_T, p=2, dim=-1)

        return s_T, q_tilde
