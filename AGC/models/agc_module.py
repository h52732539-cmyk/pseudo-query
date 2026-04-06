"""
AGC 核心模块: 伪查询作为聚类中心的完整管线。
Phase A: 数据感知伪查询生成（条件化交叉注意力路由）
Phase B: 联合编码 + 显著性计算
Phase C: Soft-to-Hard 退火聚类分配
Phase D: 显著性加权聚合 + 残差锚定
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── Phase A 组件 ───────────────


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


class PseudoQueryGenerator(nn.Module):
    """
    Phase A: 全局语义密码本 + 元查询 + 视频条件化交叉注意力 → 伪查询。
    Q_Ψ = CrossAttn(M + x̄, E, E)
    """

    def __init__(
        self,
        feature_dim: int,
        codebook_size: int,
        num_pseudo_queries: int,
        num_heads: int = 8,
        pool_type: str = "attention",
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # 全局语义密码本 E ∈ R^{K×h}
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, feature_dim) * (feature_dim ** -0.5)
        )
        # 元查询 M ∈ R^{m×h}
        self.meta_queries = nn.Parameter(
            torch.randn(num_pseudo_queries, feature_dim) * (feature_dim ** -0.5)
        )

        # 视频摘要池化
        if pool_type == "attention":
            self.pool = AttentionPool(feature_dim)
        else:
            self.pool = None

        # 交叉注意力: Query=M+x̄, Key=Value=E
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, h) — 视频 token
        Returns:
            Q_Psi: (B, m, h) — 伪查询
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

        # Step 3: 交叉注意力 — 从密码本中路由
        E = self.codebook.unsqueeze(0).expand(B, -1, -1)  # (B, K, h)
        attn_out, _ = self.cross_attn(queries, E, E)  # (B, m, h)
        Q_Psi = self.norm(attn_out + queries)  # 残差 + LayerNorm

        return Q_Psi


# ─────────────── Phase B 组件 ───────────────


class TransformerBlock(nn.Module):
    """Transformer 编码层，支持返回注意力权重（用于显著性计算）。"""

    def __init__(self, feature_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        """
        Args:
            x: (B, S, h)
            need_weights: 是否返回注意力权重
        Returns:
            x: (B, S, h)
            attn_weights: (B, H, S, S) 或 None
        """
        residual = x
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = self.norm1(residual + self.dropout(attn_out))

        residual = x
        x = self.norm2(residual + self.ffn(x))

        return x, attn_weights


class JointEncoder(nn.Module):
    """
    Phase B: 将 [视频 token, 伪查询] 联合编码，从最后一层注意力提取显著性。
    """

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, q_psi: torch.Tensor):
        """
        Args:
            x: (B, n, h) — 视频 token
            q_psi: (B, m, h) — 伪查询
        Returns:
            Z_X: (B, n, h) — 编码后视频 token
            Z_Psi: (B, m, h) — 编码后伪查询（= 聚类中心）
            alpha: (B, n) — 显著性分数
        """
        n = x.size(1)

        # 拼接 [video_tokens, pseudo_queries]
        combined = torch.cat([x, q_psi], dim=1)  # (B, n+m, h)
        combined = self.input_norm(combined)

        # 逐层编码，仅最后一层提取注意力权重
        attn_weights = None
        for i, layer in enumerate(self.layers):
            need_weights = (i == len(self.layers) - 1)
            combined, attn_weights = layer(combined, need_weights=need_weights)

        # 拆分
        Z_X = combined[:, :n]    # (B, n, h)
        Z_Psi = combined[:, n:]  # (B, m, h)

        # 显著性: 伪查询位置对视频 token 的平均注意力
        # attn_weights: (B, H, n+m, n+m)
        pq_to_video = attn_weights[:, :, n:, :n]  # (B, H, m, n)
        alpha = pq_to_video.mean(dim=(1, 2))       # (B, n)

        return Z_X, Z_Psi, alpha


# ─────────────── 完整模型 ───────────────


class AGCModel(nn.Module):
    """
    AGC 方案二完整模型: 伪查询作为聚类中心。
    Phases A → B → C → D，输出压缩后的视频表示 {c_k}。
    """

    def __init__(
        self,
        feature_dim: int = 512,
        codebook_size: int = 128,
        num_pseudo_queries: int = 32,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        lambda_init: float = 0.1,
        pool_type: str = "attention",
        temperature_init: float = 0.07,
    ):
        super().__init__()

        # Phase A: 伪查询生成器
        self.pq_generator = PseudoQueryGenerator(
            feature_dim=feature_dim,
            codebook_size=codebook_size,
            num_pseudo_queries=num_pseudo_queries,
            num_heads=num_heads,
            pool_type=pool_type,
        )

        # Phase B: 联合编码器
        self.joint_encoder = JointEncoder(
            feature_dim=feature_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Phase D: 残差锚定系数 λ (sigmoid 约束到 [0,1])
        _logit = math.log(lambda_init / (1.0 - lambda_init))
        self._lambda_raw = nn.Parameter(torch.tensor(_logit))

        # InfoNCE 可学习温度 (log scale, 保证正数)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    @property
    def lambda_val(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda_raw)

    @property
    def codebook(self) -> torch.Tensor:
        return self.pq_generator.codebook

    def forward(self, video_tokens: torch.Tensor, tau: float = 1.0):
        """
        完整前向传播 (Phases A-D)。

        Args:
            video_tokens: (B, n, h) — 预提取 CLIP 视觉特征
            tau: float — 聚类温度（由退火调度器控制）
        Returns:
            c: (B, m, h) — 压缩后视频表示
            aux: dict — 中间结果（用于损失计算和监控）
        """
        # Phase A: 生成伪查询
        Q_Psi = self.pq_generator(video_tokens)  # (B, m, h)

        # Phase B: 联合编码 + 显著性
        Z_X, Z_Psi, alpha = self.joint_encoder(video_tokens, Q_Psi)

        # Phase C: 软聚类分配
        Z_X_norm = F.normalize(Z_X, dim=-1)
        mu = F.normalize(Z_Psi, dim=-1)  # 聚类中心
        cos_sim = torch.bmm(Z_X_norm, mu.transpose(1, 2))  # (B, n, m)
        w = F.softmax(cos_sim / tau, dim=-1)  # (B, n, m) — 对聚类维度 softmax

        # Phase D: 显著性加权聚合 + 残差锚定
        w_T = w.transpose(1, 2)  # (B, m, n)
        alpha_exp = alpha.unsqueeze(1)  # (B, 1, n)
        combined_w = w_T * alpha_exp  # (B, m, n)

        denom = combined_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, m, 1)
        agg_pool = torch.bmm(combined_w, Z_X) / denom  # (B, m, h)

        lam = self.lambda_val
        c = (1.0 - lam) * agg_pool + lam * Z_Psi  # (B, m, h)

        aux = {
            "Z_X": Z_X,
            "Z_Psi": Z_Psi,
            "alpha": alpha,
            "w": w,
            "cos_sim": cos_sim,
            "lambda": lam,
            "Q_Psi": Q_Psi,
        }

        return c, aux

    def encode_video(self, video_tokens: torch.Tensor, tau: float = 0.01):
        """推理用: 编码视频为压缩表示（近硬分配）。"""
        with torch.no_grad():
            c, _ = self.forward(video_tokens, tau=tau)
        return c
