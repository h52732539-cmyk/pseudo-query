"""
原型库 P（K × d）：
- 离线 K-Means 或 GMM+BIC 初始化
- 训练时作为 nn.Parameter 可学习微调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def build_prototypes_kmeans(
    token_features: np.ndarray,
    num_prototypes: int = 512,
    batch_size: int = 100000,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    对全局 token pool 运行 Mini-Batch K-Means，返回聚类中心。
    Args:
        token_features: (N, d) numpy array — 所有 token 的特征
        num_prototypes: K
    Returns:
        centroids: (K, d) numpy array
    """
    print(f"[K-Means] Running MiniBatchKMeans: N={token_features.shape[0]}, K={num_prototypes} ...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_prototypes,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    kmeans.fit(token_features)
    print(f"[K-Means] Done. Inertia={kmeans.inertia_:.4f}")
    return kmeans.cluster_centers_.astype(np.float32)


def build_prototypes_gmm(
    features: np.ndarray,
    candidate_ks: list = None,
    random_state: int = 42,
    max_samples: int = 50000,
) -> tuple:
    """
    对特征运行 GMM + BIC 自适应选择最优 K，返回聚类中心。
    Args:
        features: (N, d) numpy array — sentence-level 或 token-level 特征
        candidate_ks: 候选 K 列表
        max_samples: 超过此数量时随机子采样（GMM 计算量大）
    Returns:
        (centroids, best_k):
            centroids: (K, d) numpy array
            best_k: int — BIC 选出的最优 K
    """
    if candidate_ks is None:
        candidate_ks = [32, 64, 128, 256, 512]

    N, d = features.shape
    # 子采样以控制 GMM 计算量
    if N > max_samples:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(N, max_samples, replace=False)
        fit_features = features[indices]
        print(f"[GMM] Subsampled {N} → {max_samples} for fitting")
    else:
        fit_features = features

    best_bic = float("inf")
    best_k = candidate_ks[0]
    best_gmm = None

    print(f"[GMM+BIC] Fitting candidates K={candidate_ks} on {fit_features.shape[0]} samples, d={d} ...")
    for k in candidate_ks:
        if k > fit_features.shape[0]:
            print(f"  K={k}: skipped (K > N_samples)")
            continue
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            max_iter=100,
            n_init=1,
            random_state=random_state,
            verbose=0,
        )
        gmm.fit(fit_features)
        bic = gmm.bic(fit_features)
        print(f"  K={k}: BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    print(f"[GMM+BIC] Best K={best_k}, BIC={best_bic:.2f}")

    centroids = best_gmm.means_.astype(np.float32)  # (K, d)

    # 打印聚类大小分布
    labels = best_gmm.predict(fit_features)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[GMM] Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, non-empty={len(unique)}/{best_k}")

    return centroids, best_k


class PrototypeLibrary(nn.Module):
    """
    可学习原型库 P ∈ R^{K × d}。
    可从 K-Means 中心初始化，也可随机初始化。
    """

    def __init__(self, num_prototypes: int, feature_dim: int, init_weights: torch.Tensor = None):
        super().__init__()
        if init_weights is not None:
            assert init_weights.shape == (num_prototypes, feature_dim)
            self.prototypes = nn.Parameter(init_weights.clone())
        else:
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.02)

    @property
    def K(self):
        return self.prototypes.shape[0]

    @property
    def d(self):
        return self.prototypes.shape[1]

    def forward(self):
        """返回 L2 归一化的原型矩阵 (K, d)"""
        return F.normalize(self.prototypes, p=2, dim=-1)
