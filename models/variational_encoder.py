"""
变分压缩网络 (Variational Compression Network):
  输入: 视频的所有密集token特征 {t_1, ..., t_M}
  Token Aggregation (Prototype-guided Attention) → (B, K, d) 每个原型独立聚合
  Per-prototype Mean Head  → μ_k ∈ R  (共 K 个)
  Per-prototype Var Head   → σ²_k ∈ R  (Softplus保证非负)
  输出: N(μ, diag(σ²)),  μ, σ² ∈ R^K
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrototypeGuidedAttention(nn.Module):
    """
    用 K 个原型作为 query，attend 所有视频 token，产出 K 个聚合特征。
    返回 (B, K, d)，不做 mean-pool，保留每个原型的独立信息。
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

        # 扩展 prototypes 为 query: (B, K, d)
        Q = self.q_proj(prototypes.unsqueeze(0).expand(B, -1, -1))
        K_feat = self.k_proj(token_features)
        V = self.v_proj(token_features)

        # Multi-head reshape: (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)         # (B, H, K, hd)
        K_feat = K_feat.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, hd)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)         # (B, H, M, hd)

        # Attention: (B, H, K, M)
        attn = torch.matmul(Q, K_feat.transpose(-2, -1)) / self.scale

        if token_mask is not None:
            # token_mask: (B, M) → (B, 1, 1, M)
            mask = token_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # (B, H, K, hd)
        out = torch.matmul(attn, V)
        # (B, K, d)
        out = out.transpose(1, 2).contiguous().view(B, K, d)
        out = self.out_proj(out)

        return out  # (B, K, d) — 保留每个原型的独立聚合特征


class MeanAggregation(nn.Module):
    """简单平均池化（backup选项），输出 (B, 1, d) 以兼容逐原型 head"""

    def forward(self, prototypes, token_features, token_mask=None):
        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).float()
            aggregated = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            aggregated = token_features.mean(dim=1)
        # (B, d) → (B, 1, d)，后续 mean_head 会广播到 K 维
        return aggregated.unsqueeze(1)


class VariationalCompressor(nn.Module):
    """
    变分压缩网络：
      token_features → aggregation → (B, K, d) → 逐原型 (μ_k, σ²_k)
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

        # 逐原型映射: (B, K, d) → (B, K, 1) → squeeze → (B, K)
        self.mean_head = nn.Linear(feature_dim, 1)
        self.var_head = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softplus(),
        )

    def forward(self, prototypes, token_features, token_mask=None):
        """
        Args:
            prototypes: (K, d)
            token_features: (B, M, d)
            token_mask: (B, M) optional
        Returns:
            mu: (B, K)
            sigma_sq: (B, K)
        """
        h = self.aggregator(prototypes, token_features, token_mask)  # (B, K, d)
        mu = self.mean_head(h).squeeze(-1)          # (B, K, 1) → (B, K)
        sigma_sq = self.var_head(h).squeeze(-1)     # (B, K, 1) → (B, K)
        return mu, sigma_sq

    def reparameterize(self, mu, sigma_sq):
        """重参数化采样（训练时用）"""
        std = torch.sqrt(sigma_sq + 1e-8)
        eps = torch.randn_like(std)
        return mu + eps * std
