"""
方案C Pipeline: 混合 EMA + Sinkhorn 原型学习。
- 原型为 nn.Parameter + EMA 影子副本
- Attention 中使用 EMA 原型（稳定），QueryAssembly 使用梯度原型（可微）
- 死原型检测与重初始化
- Sinkhorn 等分约束 + 多视图交叉预测
- 逐原型细粒度对比损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.prototype import EMAPrototypeLibrary, SwAVPrototypeLoss
from models.video_encoder import VideoEncoder
from models.query_assembly import QueryAssembly
from models.scoring import prototype_anchored_contrastive_loss, cosine_retrieval_score


class HybridPipelineModel(nn.Module):
    """
    Hybrid EMA+SwAV Pseudo-Query Video Retrieval Model (方案C).
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
        ema_decay: float = 0.999,
        dead_proto_threshold: int = 100,
    ):
        super().__init__()
        self.dead_proto_threshold = dead_proto_threshold

        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

        self.prototype_lib = EMAPrototypeLibrary(
            num_prototypes, feature_dim, ema_decay=ema_decay
        )

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

    def encode_video(self, token_features, token_mask=None, use_ema: bool = False):
        """
        编码视频 → (h, mu)
        Args:
            token_features: (B, M, d)
            token_mask: (B, M)
            use_ema: 是否使用 EMA 原型（训练时 True，推理时 False）
        """
        prototypes = self.prototype_lib(use_ema=use_ema)
        h, mu = self.video_encoder(prototypes, token_features, token_mask)
        return h, mu

    def encode_query(self, query_tokens, query_mask=None, use_ema: bool = False,
                     return_raw: bool = False):
        """
        编码查询 → (s_T, q_tilde[, s_T_raw])
        梯度原型用于 QueryAssembly 以保证梯度回传到原型参数。
        """
        prototypes = self.prototype_lib(use_ema=use_ema)
        return self.query_assembly(query_tokens, prototypes, self.temperature,
                                   query_mask, return_raw=return_raw)

    def forward(self, v1_token_features, v1_token_mask,
                v2_token_features, v2_token_mask,
                query_token_features, query_token_mask):
        """
        训练前向传播（多视图）。
        - 视频编码使用 EMA 原型（稳定），查询编码使用梯度原型（可微）
        """
        # 视频侧：EMA 原型（稳定目标）
        h1, mu1 = self.encode_video(v1_token_features, v1_token_mask, use_ema=True)
        h2, mu2 = self.encode_video(v2_token_features, v2_token_mask, use_ema=True)
        h_avg = (h1 + h2) / 2

        # 查询侧：梯度原型（保留梯度路径）
        s_T, q_tilde = self.encode_query(query_token_features, query_token_mask, use_ema=False)

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

    def post_step(self, s_T: torch.Tensor, token_features: torch.Tensor):
        """
        每训练步后调用：更新 EMA + 使用统计 + 死原型重初始化。
        Args:
            s_T: (B, K) — 当前 batch 的查询激活
            token_features: (B, M, d) — 当前 batch 的视频 token 特征
        Returns:
            n_dead: 重初始化的死原型数
        """
        self.prototype_lib.update_ema()
        self.prototype_lib.update_usage(s_T)
        n_dead = self.prototype_lib.reinit_dead_prototypes(
            token_features, dead_threshold=self.dead_proto_threshold
        )
        return n_dead

    # ---- 推理方法 ----

    def get_video_repr(self, token_features, token_mask=None):
        """推理用：编码视频 → μ ∈ R^K"""
        _, mu = self.encode_video(token_features, token_mask, use_ema=False)
        return mu

    def get_query_repr(self, query_tokens, query_mask=None, return_raw=False):
        """推理用：编码查询 → s_T ∈ R^K [可选返回 s_T_raw]"""
        result = self.encode_query(query_tokens, query_mask, use_ema=False,
                                   return_raw=return_raw)
        if return_raw:
            s_T, _, s_T_raw = result
            return s_T, s_T_raw
        s_T, _ = result
        return s_T

    def retrieval_score(self, s_T, mu):
        """推理用：计算检索分数"""
        return cosine_retrieval_score(s_T, mu, temperature=self.temperature)
