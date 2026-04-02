"""
句子级查询适配器：轻量 MLP 将真实查询的 CLIP 句子 embedding 投影到伪查询特征空间。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAdapter(nn.Module):
    """
    QueryAdapter: LayerNorm → Linear(d, 4d) → GELU → Linear(4d, d) → 残差连接 → L2归一化。
    W2 零初始化确保初始化时近似恒等映射。
    """

    def __init__(self, feature_dim: int = 512, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = feature_dim * hidden_mult
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.act = nn.GELU()

        # W1 Xavier, W2 零初始化（残差初始化时为恒等）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Args:
            x: (B, d) — CLIP 句子 embedding
        Returns:
            out: (B, d) — 适配后 embedding, L2归一化
        """
        h = self.act(self.fc1(self.norm(x)))
        out = x + self.fc2(h)
        return F.normalize(out, p=2, dim=-1)
