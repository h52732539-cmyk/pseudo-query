"""
视频编码器 (Video Encoder):
  输入: 视频的所有密集token特征 {t_1, ..., t_M}
  Token Aggregation (Prototype-guided Attention) → (B, K, d) 每个原型独立聚合
  Per-prototype Mean Head  → μ_k ∈ R  (共 K 个)
  输出: h ∈ R^{K×d} (逐原型聚合特征, 训练用), μ ∈ R^K (标量投影, 推理用)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrototypeGuidedAttention(nn.Module):
    """
    用 K 个原型作为 query，attend 所有视频 token，产出 K 个聚合特征。
    返回 (B, K, d)，保留每个原型的独立信息。
    """

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, prototypes, token_features, token_mask=None):
        """
        Args:
            prototypes: (K, d) — 原型作为 query
            token_features: (B, M, d) — 视频的所有 token 特征
            token_mask: (B, M) — 1=valid, 0=padding
        Returns:
            out: (B, K, d) — 每个原型的独立聚合特征
        """
        B, M, d = token_features.shape
        K = prototypes.shape[0]

        Q = self.q_proj(prototypes.unsqueeze(0).expand(B, -1, -1))
        K_feat = self.k_proj(token_features)
        V = self.v_proj(token_features)

        Q = Q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        K_feat = K_feat.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K_feat.transpose(-2, -1)) / self.scale

        if token_mask is not None:
            mask = token_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, K, d)
        out = self.out_proj(out)

        return out


class MeanAggregation(nn.Module):
    """简单平均池化（backup选项），输出 (B, 1, d) 以兼容逐原型 head"""

    def forward(self, prototypes, token_features, token_mask=None):
        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).float()
            aggregated = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            aggregated = token_features.mean(dim=1)
        return aggregated.unsqueeze(1)


class VideoEncoder(nn.Module):
    """
    视频编码器：
      token_features → aggregation → (B, K, d) → 逐原型 μ_k
    返回:
      h: (B, K, d) — 逐原型聚合特征（训练时用于逐原型对比损失）
      mu: (B, K) — 标量投影（推理时用于全局匹配）
    """

    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        aggregation: str = "attention",
        num_heads: int = 8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes

        if aggregation == "attention":
            self.aggregator = PrototypeGuidedAttention(feature_dim, num_heads)
        else:
            self.aggregator = MeanAggregation()

        self.mean_head = nn.Linear(feature_dim, 1)

    def forward(self, prototypes, token_features, token_mask=None):
        """
        Args:
            prototypes: (K, d)
            token_features: (B, M, d)
            token_mask: (B, M) optional
        Returns:
            h: (B, K, d) — 逐原型聚合特征
            mu: (B, K) — L2归一化后的标量投影
        """
        h = self.aggregator(prototypes, token_features, token_mask)  # (B, K, d)
        mu = self.mean_head(h).squeeze(-1)  # (B, K)
        mu = F.normalize(mu, p=2, dim=-1)
        return h, mu
