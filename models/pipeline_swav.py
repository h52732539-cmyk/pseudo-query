"""
方案B Pipeline: SwAV Sinkhorn 在线原型学习。
- 原型为 nn.Parameter，端到端梯度更新
- Sinkhorn 等分约束 + 多视图交叉预测
- 逐原型细粒度对比损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.prototype import PrototypeLibrary, SwAVPrototypeLoss
from models.video_encoder import VideoEncoder
from models.query_assembly import QueryAssembly
from models.scoring import prototype_anchored_contrastive_loss, cosine_retrieval_score


class SwAVPipelineModel(nn.Module):
    """
    SwAV-based Pseudo-Query Video Retrieval Model (方案B).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_prototypes: int = 512,
        aggregation: str = "attention",
        temperature_init: float = 0.07,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        swav_temperature: float = 0.1,
    ):
        super().__init__()

        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

        self.prototype_lib = PrototypeLibrary(num_prototypes, feature_dim)

        self.video_encoder = VideoEncoder(
            feature_dim, num_prototypes, aggregation=aggregation
        )

        self.query_assembly = QueryAssembly()

        self.swav_loss_fn = SwAVPrototypeLoss(
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iters=sinkhorn_iters,
            swav_temperature=swav_temperature,
        )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_video(self, token_features, token_mask=None):
        """
        编码视频 → (h, mu)
        Args:
            token_features: (B, M, d)
            token_mask: (B, M)
        Returns:
            h: (B, K, d) — 逐原型聚合特征
            mu: (B, K) — L2归一化标量投影
        """
        prototypes = self.prototype_lib()
        h, mu = self.video_encoder(prototypes, token_features, token_mask)
        return h, mu

    def encode_query(self, query_tokens, query_mask=None, return_raw=False):
        """
        编码查询 → (s_T, q_tilde[, s_T_raw])
        Args:
            query_tokens: (B, L, d)
            query_mask: (B, L)
            return_raw: bool — 是否额外返回归一化前的 s_T_raw
        Returns:
            s_T: (B, K) — 查询激活向量
            q_tilde: (B, K, d) — 逐原型查询语义
            s_T_raw: (B, K) — [仅 return_raw=True]
        """
        prototypes = self.prototype_lib()
        return self.query_assembly(query_tokens, prototypes, self.temperature,
                                   query_mask, return_raw=return_raw)

    def forward(self, v1_token_features, v1_token_mask,
                v2_token_features, v2_token_mask,
                query_token_features, query_token_mask):
        """
        训练前向传播（多视图）。
        Returns:
            h_avg: (B, K, d) — 两视图聚合特征均值
            mu1, mu2: (B, K) — 两视图的 μ
            s_T: (B, K) — 查询激活
            q_tilde: (B, K, d) — 查询逐原型语义
        """
        h1, mu1 = self.encode_video(v1_token_features, v1_token_mask)
        h2, mu2 = self.encode_video(v2_token_features, v2_token_mask)
        h_avg = (h1 + h2) / 2

        s_T, q_tilde = self.encode_query(query_token_features, query_token_mask)

        return h_avg, mu1, mu2, s_T, q_tilde

    def compute_loss(self, h_avg, mu1, mu2, s_T, q_tilde, swav_alpha=0.5):
        """
        总损失 = L_match + α · L_swav
        """
        loss_match = prototype_anchored_contrastive_loss(
            h_avg, q_tilde, s_T, temperature=self.temperature
        )
        loss_swav = self.swav_loss_fn(mu1, mu2)
        loss_total = loss_match + swav_alpha * loss_swav
        return loss_total, loss_match, loss_swav

    # ---- 推理方法 ----

    def get_video_repr(self, token_features, token_mask=None):
        """推理用：编码视频 → μ ∈ R^K"""
        _, mu = self.encode_video(token_features, token_mask)
        return mu

    def get_query_repr(self, query_tokens, query_mask=None, return_raw=False):
        """推理用：编码查询 → s_T ∈ R^K [可选返回 s_T_raw]"""
        result = self.encode_query(query_tokens, query_mask, return_raw=return_raw)
        if return_raw:
            s_T, _, s_T_raw = result
            return s_T, s_T_raw
        s_T, _ = result
        return s_T

    def retrieval_score(self, s_T, mu):
        """推理用：计算检索分数"""
        return cosine_retrieval_score(s_T, mu, temperature=self.temperature)
