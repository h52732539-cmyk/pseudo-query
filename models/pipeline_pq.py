"""
PseudoQueryPipeline — 新 Pipeline，整合 QueryAdapter + FineGrainedReranker。
不包含 CLIP encoder（冻结，在外部使用）和 NucleusFilter（单独管理）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.query_adapter import QueryAdapter
from models.fine_reranker import FineGrainedReranker


class PseudoQueryPipeline(nn.Module):
    """
    训练接口:
        forward() → adapted_query, fine_score_matrix
        compute_loss() → L_total, L_coarse, L_fine
    推理接口:
        adapt_query() → adapted embedding
        rerank() → 精排分数
    """

    def __init__(
        self,
        feature_dim: int = 512,
        adapter_hidden_mult: int = 4,
        reranker_num_heads: int = 8,
        temperature_init: float = 0.07,
        fine_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.query_adapter = QueryAdapter(feature_dim, adapter_hidden_mult)
        self.fine_reranker = FineGrainedReranker(feature_dim, reranker_num_heads)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))
        self.fine_loss_weight = fine_loss_weight

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, q_sent, q_tokens, q_mask, narr_tokens, narr_mask):
        """
        训练时前向传播。
        Args:
            q_sent:      (B, d) — 查询句子 embedding
            q_tokens:    (B, L_q, d) — 查询 token features
            q_mask:      (B, L_q) — 查询 attention mask
            narr_tokens: (B, M, d) — narration token features
            narr_mask:   (B, M) — narration attention mask
        Returns:
            adapted_query: (B, d) — 适配后查询 embedding
            fine_score_matrix: (B, B) — B×B 精排分数矩阵
        """
        adapted_query = self.query_adapter(q_sent)  # (B, d)

        # 构建 B×B 精排分数矩阵
        B = q_tokens.shape[0]
        fine_score_matrix = self._build_score_matrix(
            q_tokens, q_mask, narr_tokens, narr_mask, B
        )

        return adapted_query, fine_score_matrix

    def _build_score_matrix(self, q_tokens, q_mask, narr_tokens, narr_mask, B):
        """构建 B×B 精排分数矩阵。"""
        # 展开为 B² 对
        q_expand = q_tokens.unsqueeze(1).expand(-1, B, -1, -1).reshape(B * B, -1, q_tokens.shape[-1])
        qm_expand = q_mask.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
        n_expand = narr_tokens.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * B, -1, narr_tokens.shape[-1])
        nm_expand = narr_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)

        scores = self.fine_reranker(q_expand, n_expand, qm_expand, nm_expand)  # (B²,)
        return scores.reshape(B, B)

    def compute_loss(self, adapted_query, filtered_centroids, fine_score_matrix):
        """
        计算总损失。
        Args:
            adapted_query:     (B, d) — 适配后查询
            filtered_centroids: (B, d) — 核过滤后的 narration centroid
            fine_score_matrix:  (B, B) — 精排分数矩阵
        Returns:
            loss_total, loss_coarse, loss_fine
        """
        tau = self.temperature

        # 粗筛对称 InfoNCE
        coarse_scores = adapted_query @ filtered_centroids.T / tau  # (B, B)
        loss_coarse = symmetric_infonce(coarse_scores)

        # 精排对称 InfoNCE
        fine_scores = fine_score_matrix / tau
        loss_fine = symmetric_infonce(fine_scores)

        loss_total = loss_coarse + self.fine_loss_weight * loss_fine
        return loss_total, loss_coarse, loss_fine

    def adapt_query(self, q_sent):
        """推理用：适配查询。"""
        return self.query_adapter(q_sent)

    def rerank(self, q_tokens, q_mask, narr_tokens, narr_mask):
        """推理用：精排单个 query-video 对。"""
        return self.fine_reranker(q_tokens, narr_tokens, q_mask, narr_mask)


def symmetric_infonce(scores):
    """
    对称 InfoNCE 损失。
    Args:
        scores: (B, B) — 分数矩阵，对角线为正样本
    Returns:
        loss: scalar
    """
    B = scores.shape[0]
    labels = torch.arange(B, device=scores.device)
    loss_q2n = F.cross_entropy(scores, labels)
    loss_n2q = F.cross_entropy(scores.T, labels)
    return (loss_q2n + loss_n2q) / 2
