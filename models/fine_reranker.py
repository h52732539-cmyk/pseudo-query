"""
Token 级精排器：查询 token 通过 Cross-Attention 关注 narration token，产出匹配分数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedReranker(nn.Module):
    """
    FineGrainedReranker:
      1. MultiheadAttention: query tokens attend narration tokens
      2. LayerNorm + 残差连接
      3. Masked mean pooling → (B, d)
      4. Linear score head → 标量分数
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.score_head = nn.Linear(feature_dim, 1)

    def forward(self, query_tokens, narr_tokens, query_mask=None, narr_mask=None):
        """
        Args:
            query_tokens: (B, L_q, d) — 查询 token features
            narr_tokens:  (B, M, d)   — narration token features
            query_mask:   (B, L_q)    — 1=valid, 0=pad
            narr_mask:    (B, M)      — 1=valid, 0=pad
        Returns:
            score: (B,) — 匹配分数
        """
        narr_kpm = (~narr_mask.bool()) if narr_mask is not None else None

        attn_out, _ = self.cross_attn(
            query_tokens, narr_tokens, narr_tokens,
            key_padding_mask=narr_kpm,
        )
        # 残差 + LayerNorm
        out = self.norm(query_tokens + attn_out)  # (B, L_q, d)

        # Masked mean pooling
        if query_mask is not None:
            mask_expanded = query_mask.unsqueeze(-1)  # (B, L_q, 1)
            pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        score = self.score_head(pooled).squeeze(-1)  # (B,)
        return score
