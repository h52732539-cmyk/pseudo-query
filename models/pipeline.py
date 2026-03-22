"""
完整 Pipeline 模型：将所有可训练模块组装为一个 nn.Module。
- 动态 num_prototypes（从加载的原型推断）
- μ 输出 L2 归一化
- Memory Bank 负样本增强
- Free-bits KL
- cosine similarity + 可学习温度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.prototype import PrototypeLibrary
from models.variational_encoder import VariationalCompressor
from models.query_assembly import QueryAssembly
from models.scoring import uncertainty_aware_score, info_nce_loss, kl_divergence
from models.memory_bank import MemoryBank


class VIBPseudoQueryModel(nn.Module):
    """
    VIB-based Pseudo-Query Video Retrieval Model.
    可训练模块：PrototypeLibrary, VariationalCompressor, QueryAssembly (无参数), temperature
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_prototypes: int = 512,
        aggregation: str = "attention",
        temperature_init: float = 0.07,
        variance_penalty: float = 0.1,
        prototype_init_weights: torch.Tensor = None,
        free_bits: float = 0.1,
        memory_bank_size: int = 0,
    ):
        super().__init__()
        self.variance_penalty = variance_penalty
        self.free_bits = free_bits

        # 可学习温度
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

        # 原型库
        self.prototype_lib = PrototypeLibrary(
            num_prototypes, feature_dim, init_weights=prototype_init_weights
        )

        # 变分压缩网络
        self.compressor = VariationalCompressor(
            feature_dim, num_prototypes, aggregation=aggregation
        )

        # 查询组装（无可学习参数）
        self.query_assembly = QueryAssembly()

        # Memory Bank（可选）
        self.memory_bank = None
        if memory_bank_size > 0:
            self.memory_bank = MemoryBank(memory_bank_size, num_prototypes)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_video(self, token_features, token_mask=None):
        """
        编码视频侧（伪查询token）→ (μ, σ²)
        μ 经过 L2 归一化以对齐 cosine similarity 空间。
        Args:
            token_features: (B, M, d)
            token_mask: (B, M)
        Returns:
            mu: (B, K), sigma_sq: (B, K)
        """
        prototypes = self.prototype_lib()  # (K, d)
        mu, sigma_sq = self.compressor(prototypes, token_features, token_mask)
        # L2 归一化 μ，与 s_T 对齐到同一空间
        mu = F.normalize(mu, p=2, dim=-1)
        return mu, sigma_sq

    def encode_query(self, query_tokens, query_mask=None):
        """
        编码查询侧 → s_T（内部已 L2 归一化）
        Args:
            query_tokens: (B, L, d)
            query_mask: (B, L)
        Returns:
            s_T: (B, K)
        """
        prototypes = self.prototype_lib()
        s_T = self.query_assembly(query_tokens, prototypes, self.temperature, query_mask)
        return s_T

    def forward(self, video_token_features, video_token_mask,
                query_token_features, query_token_mask):
        """
        完整前向传播（训练用）。
        Returns:
            scores: (B, B) 或 (B, B+Q) — batch 内相似度矩阵（可能含 memory bank 负样本）
            mu, sigma_sq: 用于 KL loss
        """
        mu, sigma_sq = self.encode_video(video_token_features, video_token_mask)
        s_T = self.encode_query(query_token_features, query_token_mask)

        # 训练时用重参数化采样替代 mu
        m = self.compressor.reparameterize(mu, sigma_sq)  # (B, K)
        # 采样后也做 L2 归一化
        m = F.normalize(m, p=2, dim=-1)

        # 计算 batch 内匹配分数
        scores = uncertainty_aware_score(
            s_T, m, sigma_sq, self.variance_penalty, temperature=self.temperature
        )  # (B, B)

        # Memory Bank 扩展负样本
        if self.training and self.memory_bank is not None:
            negatives = self.memory_bank.get_negatives()  # (Q, K) or None
            if negatives is not None:
                neg_scores = uncertainty_aware_score(
                    s_T, negatives, None, 0.0, temperature=self.temperature
                )  # (B, Q)
                scores = torch.cat([scores, neg_scores], dim=1)  # (B, B+Q)
            # 入队当前 batch 的 μ（detached）
            self.memory_bank.enqueue(mu.detach())

        return scores, mu, sigma_sq

    def compute_loss(self, scores, mu, sigma_sq, beta=1e-3):
        """
        总损失 = InfoNCE + β * KL (with free-bits)
        """
        loss_task = info_nce_loss(scores)
        loss_kl = kl_divergence(mu, sigma_sq, free_bits=self.free_bits)
        loss_total = loss_task + beta * loss_kl
        return loss_total, loss_task, loss_kl
