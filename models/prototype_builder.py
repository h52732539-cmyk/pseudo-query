"""
推理时原型构建：Sinkhorn-Knopp 聚类 + 倒排索引。
sinkhorn() 从旧 prototype.py 迁移而来。
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, niters: int = 3):
    """
    Sinkhorn-Knopp 最优传输，产出满足等分约束的软分配矩阵。
    Args:
        scores: (N, K) — 样本与原型的相似度
        eps: 温度参数
        niters: 迭代次数
    Returns:
        Q: (N, K) — 软分配矩阵
    """
    Q = torch.exp(scores / eps).T  # (K, N)
    K, N = Q.shape
    Q /= Q.sum()

    for _ in range(niters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q *= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q *= N

    return (Q / N).T  # (N, K)


@torch.no_grad()
def sinkhorn_cluster(
    embeddings: torch.Tensor,
    num_prototypes: int = 256,
    eps: float = 0.05,
    niters: int = 3,
    n_rounds: int = 10,
):
    """
    Sinkhorn-Knopp 聚类：迭代质心-分配交替更新。
    Args:
        embeddings: (N, d) — L2 归一化的 narration sentence embeddings
        num_prototypes: K
        eps: Sinkhorn 温度
        niters: 每轮 Sinkhorn 迭代次数
        n_rounds: 质心-分配交替更新轮数
    Returns:
        prototypes: (K, d) — L2 归一化
        assignments: (N, K) — 软分配矩阵
    """
    N, d = embeddings.shape
    K = min(num_prototypes, N)

    # 随机初始化
    indices = torch.randperm(N, device=embeddings.device)[:K]
    prototypes = F.normalize(embeddings[indices].clone(), p=2, dim=-1)

    for _ in range(n_rounds):
        # 相似度
        sim = embeddings @ prototypes.T  # (N, K)
        # Sinkhorn 软分配
        assignments = sinkhorn(sim, eps, niters)  # (N, K)
        # 加权更新质心
        for k in range(K):
            w = assignments[:, k]  # (N,)
            if w.sum() > 0:
                prototypes[k] = F.normalize((w.unsqueeze(-1) * embeddings).sum(dim=0), p=2, dim=-1)

    return prototypes, assignments


@torch.no_grad()
def build_inverted_index(
    assignments: torch.Tensor,
    narration_metadata: List[Tuple[str, int]],
):
    """
    从聚类分配矩阵构建倒排索引。
    Args:
        assignments: (N, K) — 软分配矩阵
        narration_metadata: [(video_id, narr_idx), ...] — 每条 narration 的归属信息
    Returns:
        inverted_index: {prototype_id: set(video_ids)}
        video_narr_indices: {video_id: [global_narr_idx, ...]}
    """
    hard_assign = assignments.argmax(dim=1)  # (N,)

    inverted_index: Dict[int, set] = defaultdict(set)
    video_narr_indices: Dict[str, List[int]] = defaultdict(list)

    for global_idx, (vid, _narr_idx) in enumerate(narration_metadata):
        cluster_id = hard_assign[global_idx].item()
        inverted_index[cluster_id].add(vid)
        video_narr_indices[vid].append(global_idx)

    return dict(inverted_index), dict(video_narr_indices)


class InferenceIndex:
    """封装原型 + 倒排索引 + narration 缓存用于推理。"""

    def __init__(
        self,
        prototypes: torch.Tensor,
        inverted_index: Dict[int, set],
        video_narr_indices: Dict[str, List[int]],
        narr_sent_embs: torch.Tensor,
        narr_token_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Args:
            prototypes: (K, d)
            inverted_index: {prototype_id: set(video_ids)}
            video_narr_indices: {video_id: [global_narr_idx, ...]}
            narr_sent_embs: (N_total, d) — 所有过滤后 narration 的句子 embedding
            narr_token_cache: {video_id: (token_feats, token_mask)} — 精排用
        """
        self.prototypes = prototypes
        self.inverted_index = inverted_index
        self.video_narr_indices = video_narr_indices
        self.narr_sent_embs = narr_sent_embs
        self.narr_token_cache = narr_token_cache or {}

    def coarse_retrieve(self, query_emb: torch.Tensor, top_m: int = 10,
                        max_candidates: int = 200) -> List[str]:
        """
        粗筛：query vs 原型 → top-M → 合并倒排索引 → 候选视频集。
        Args:
            query_emb: (d,) — adapted query embedding
            top_m: 选取 top-M 个原型
            max_candidates: 最大候选数
        Returns:
            candidate_vids: List[str]
        """
        sim = query_emb @ self.prototypes.T  # (K,)
        _, top_indices = sim.topk(min(top_m, self.prototypes.shape[0]))

        candidates = set()
        for idx in top_indices.tolist():
            if idx in self.inverted_index:
                candidates.update(self.inverted_index[idx])
            if len(candidates) >= max_candidates:
                break

        return list(candidates)[:max_candidates]

    def get_video_narr_tokens(self, video_id: str):
        """获取缓存的 narration token features。"""
        return self.narr_token_cache.get(video_id, (None, None))
