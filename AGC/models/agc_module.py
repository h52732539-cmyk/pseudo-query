"""
AGC 核心模块: Q-Former 范式的伪查询视频表示压缩管线。

架构概览:
  1. 条件化元查询: M̃ = M + AttentionPool(X)
  2. 多层 Q-Former 交叉注意力: 伪查询从视频帧中直接聚合信息
  3. 输出 m 个连续表征作为视频的 multi-vector 表示, 送入 MaxSim 检索

(已移除全局密码本 E 和退火聚类机制, 解决路由崩塌/聚类坍缩问题)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── 基础组件 ───────────────


class AttentionPool(nn.Module):
    """注意力池化: 用可学习 query 对 token 序列做加权平均，得到单一向量。"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feature_dim) * (feature_dim ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, h)
        Returns:
            pooled: (B, h)
        """
        scores = torch.einsum("bnh,h->bn", x, self.query) / math.sqrt(x.size(-1))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnh->bh", weights, x)
        return pooled


# ─────────────── Q-Former 组件 ───────────────


class QFormerBlock(nn.Module):
    """
    Q-Former 单层: Self-Attention (伪查询间交互) + Cross-Attention (从视频帧聚合) + FFN。

    参考 BLIP-2 架构，伪查询通过自注意力保持多样性，通过交叉注意力从视频帧提取信息。
    """

    def __init__(self, feature_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Self-Attention: 伪查询间相互交互
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # Cross-Attention: Q=伪查询, K=V=视频帧
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, m, h) — 伪查询
            video_tokens: (B, n, h) — 视频帧特征
        Returns:
            queries: (B, m, h) — 更新后的伪查询
        """
        # Self-Attention
        residual = queries
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(residual + self.dropout(sa_out))

        # Cross-Attention
        residual = queries
        ca_out, _ = self.cross_attn(queries, video_tokens, video_tokens)
        queries = self.norm2(residual + self.dropout(ca_out))

        # FFN
        residual = queries
        queries = self.norm3(residual + self.ffn(queries))

        return queries


class PseudoQueryGenerator(nn.Module):
    """
    Q-Former 伪查询生成器:
      1. 视频注意力池化 → 全局摘要 x̄
      2. 条件化元查询: M̃ = M + x̄
      3. 多层 QFormerBlock: 伪查询从视频帧中直接聚合信息
      4. 输出 Z_Psi ∈ R^{m×h}
    """

    def __init__(
        self,
        feature_dim: int,
        num_pseudo_queries: int,
        num_qformer_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pool_type: str = "attention",
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # 元查询 M ∈ R^{m×h}
        self.meta_queries = nn.Parameter(
            torch.randn(num_pseudo_queries, feature_dim) * (feature_dim ** -0.5)
        )

        # 视频摘要池化 (用于条件化)
        if pool_type == "attention":
            self.pool = AttentionPool(feature_dim)
        else:
            self.pool = None

        # 视频帧输入 LayerNorm
        self.video_norm = nn.LayerNorm(feature_dim)

        # 多层 Q-Former
        self.layers = nn.ModuleList([
            QFormerBlock(feature_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_qformer_layers)
        ])

        # 输出归一化
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, h) — 视频帧特征
        Returns:
            Z_Psi: (B, m, h) — 伪查询表示
        """
        B = x.size(0)

        # Step 1: 视频摘要
        if self.pool is not None:
            x_bar = self.pool(x)  # (B, h)
        else:
            x_bar = x.mean(dim=1)  # (B, h)

        # Step 2: 条件化元查询
        M = self.meta_queries.unsqueeze(0).expand(B, -1, -1)  # (B, m, h)
        queries = M + x_bar.unsqueeze(1)  # (B, m, h)

        # Step 3: 归一化视频帧输入
        video_tokens = self.video_norm(x)  # (B, n, h)

        # Step 4: 多层 Q-Former 交叉注意力聚合
        for layer in self.layers:
            queries = layer(queries, video_tokens)  # (B, m, h)

        Z_Psi = self.output_norm(queries)
        return Z_Psi


# ─────────────── 完整模型 ───────────────


class AGCModel(nn.Module):
    """
    AGC 模型 (Q-Former 范式): 条件化元查询通过多层交叉注意力直接聚合视频帧特征,
    输出 m 个连续表征作为视频的 multi-vector 表示。
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_pseudo_queries: int = 32,
        num_qformer_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pool_type: str = "attention",
        temperature_init: float = 0.07,
        # 以下参数保留用于兼容旧配置加载，实际不使用
        codebook_size: int = 128,
        num_encoder_layers: int = 4,
        lambda_init: float = 0.1,
    ):
        super().__init__()

        # Q-Former 伪查询生成器
        self.pq_generator = PseudoQueryGenerator(
            feature_dim=feature_dim,
            num_pseudo_queries=num_pseudo_queries,
            num_qformer_layers=num_qformer_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            pool_type=pool_type,
        )

        # InfoNCE 可学习温度 (log scale, 保证正数)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def forward(self, video_tokens: torch.Tensor):
        """
        前向传播: 条件化元查询 → Q-Former 交叉注意力聚合 → multi-vector 视频表示。

        Args:
            video_tokens: (B, n, h) — 预提取 CLIP 视觉特征
        Returns:
            c: (B, m, h) — 压缩后视频表示
            aux: dict — 中间结果（用于监控）
        """
        # Q-Former 生成伪查询表示
        Z_Psi = self.pq_generator(video_tokens)  # (B, m, h)

        # 直接作为视频的 multi-vector 表示
        c = Z_Psi

        aux = {
            "Z_Psi": Z_Psi,
        }

        return c, aux

    def encode_video(self, video_tokens: torch.Tensor):
        """推理用: 编码视频为压缩表示。"""
        with torch.no_grad():
            c, _ = self.forward(video_tokens)
        return c


# ─────────────── 以下为旧版组件 (Deprecated, 保留用于加载旧 checkpoint 和对比实验) ───────────────


class _DeprecatedPseudoQueryGenerator(nn.Module):
    """[Deprecated] 旧版: 全局语义密码本 + 元查询 + 视频条件化交叉注意力 → 伪查询。"""

    def __init__(self, feature_dim, codebook_size, num_pseudo_queries, num_heads=8, pool_type="attention"):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, feature_dim) * (feature_dim ** -0.5))
        self.meta_queries = nn.Parameter(torch.randn(num_pseudo_queries, feature_dim) * (feature_dim ** -0.5))
        if pool_type == "attention":
            self.pool = AttentionPool(feature_dim)
        else:
            self.pool = None
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        B = x.size(0)
        x_bar = self.pool(x) if self.pool is not None else x.mean(dim=1)
        M = self.meta_queries.unsqueeze(0).expand(B, -1, -1)
        queries = M + x_bar.unsqueeze(1)
        E = self.codebook.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, E, E)
        return self.norm(attn_out + queries)


class TransformerBlock(nn.Module):
    """[Deprecated] Transformer 编码层，保留用于加载旧 checkpoint。"""

    def __init__(self, feature_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        residual = x
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.norm1(residual + self.dropout(attn_out))
        residual = x
        x = self.norm2(residual + self.ffn(x))
        return x, attn_weights


class JointEncoder(nn.Module):
    """[Deprecated] 联合编码器，保留用于加载旧 checkpoint。"""

    def __init__(self, feature_dim, num_layers, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, q_psi):
        n = x.size(1)
        combined = self.input_norm(torch.cat([x, q_psi], dim=1))
        attn_weights = None
        for i, layer in enumerate(self.layers):
            combined, attn_weights = layer(combined, need_weights=(i == len(self.layers) - 1))
        Z_X, Z_Psi = combined[:, :n], combined[:, n:]
        pq_to_video = attn_weights[:, :, n:, :n]
        alpha = pq_to_video.mean(dim=(1, 2))
        return Z_X, Z_Psi, alpha
