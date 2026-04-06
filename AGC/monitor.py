"""
监控工具: 密码本健康度 + 聚类均衡度指标。
用于决定是否需要引入 L_div 和 L_bal 辅助损失。
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_codebook_health(codebook: torch.Tensor, tau_div: float = 0.5) -> dict:
    """
    计算密码本健康度指标。

    Args:
        codebook: (K, h) — 密码本条目
        tau_div: 相似度阈值

    Returns:
        dict:
            cos_mean: 两两余弦相似度均值（越低越好）
            cos_max: 两两余弦最大值（越低越好）
            effective_rank: 有效秩（越高越好，接近 K 最佳）
            high_sim_ratio: cos > tau_div 的 pair 占比（越低越好）
    """
    K = codebook.size(0)
    E_norm = F.normalize(codebook, dim=-1)
    cos_matrix = E_norm @ E_norm.T  # (K, K)

    # 去除对角线
    mask = ~torch.eye(K, dtype=torch.bool, device=codebook.device)
    off_diag = cos_matrix.masked_select(mask)

    cos_mean = off_diag.mean().item()
    cos_max = off_diag.max().item()
    high_sim_ratio = (off_diag > tau_div).float().mean().item()

    # 有效秩: exp(-Σ σ̂_i log σ̂_i)，σ̂ 为归一化奇异值
    _, S, _ = torch.svd(E_norm)
    S_norm = S / S.sum()
    S_norm = S_norm.clamp(min=1e-10)
    entropy = -(S_norm * S_norm.log()).sum().item()
    effective_rank = np.exp(entropy)

    return {
        "cos_mean": cos_mean,
        "cos_max": cos_max,
        "effective_rank": effective_rank,
        "high_sim_ratio": high_sim_ratio,
    }


def compute_cluster_balance(w: torch.Tensor) -> dict:
    """
    计算聚类均衡度指标。

    Args:
        w: (B, n, m) — 软分配权重

    Returns:
        dict:
            gini: Gini 系数（0=均匀，1=集中）
            cv: 变异系数 std/mean
            dead_ratio: 死聚类占比（无 token 被分配）
            max_load_ratio: 最大聚类负载占比
            load_distribution: 各聚类 token 数量 (list)
    """
    B, n, m = w.shape

    # 硬分配
    hard = w.argmax(dim=-1).view(-1)  # (B*n,)
    counts = torch.zeros(m, device=w.device)
    counts.scatter_add_(0, hard, torch.ones_like(hard, dtype=torch.float))

    total = B * n
    load = counts.cpu().numpy()

    # Gini 系数
    sorted_load = np.sort(load)
    index = np.arange(1, m + 1)
    gini = (2 * (index * sorted_load).sum() / (m * sorted_load.sum()) - (m + 1) / m)
    gini = max(0.0, gini)  # 数值安全

    # 变异系数
    mean_load = load.mean()
    std_load = load.std()
    cv = std_load / max(mean_load, 1e-8)

    # 死聚类
    dead = (load == 0).sum()
    dead_ratio = dead / m

    # 最大负载
    max_load_ratio = load.max() / max(total, 1)

    return {
        "gini": float(gini),
        "cv": float(cv),
        "dead_ratio": float(dead_ratio),
        "max_load_ratio": float(max_load_ratio),
        "load_distribution": load.tolist(),
    }


def health_report(codebook: torch.Tensor, w: torch.Tensor, cfg: dict) -> dict:
    """
    综合健康报告，判断是否需要开启辅助损失。

    Args:
        codebook: (K, h)
        w: (B, n, m)
        cfg: monitor 配置

    Returns:
        dict 包含指标 + 建议
    """
    mon_cfg = cfg.get("monitor", {})
    cb = compute_codebook_health(codebook)
    cl = compute_cluster_balance(w)

    recommendations = []

    # 检查密码本
    if cb["cos_max"] > mon_cfg.get("codebook_cos_max_alarm", 0.8):
        recommendations.append(
            f"⚠ Codebook cos_max={cb['cos_max']:.3f} > {mon_cfg.get('codebook_cos_max_alarm', 0.8)}: "
            f"建议开启 L_div (beta_div=0.1)"
        )
    if cb["cos_mean"] > mon_cfg.get("codebook_cos_mean_alarm", 0.5):
        recommendations.append(
            f"⚠ Codebook cos_mean={cb['cos_mean']:.3f} > {mon_cfg.get('codebook_cos_mean_alarm', 0.5)}: "
            f"密码本多样性偏低"
        )

    # 检查聚类
    if cl["gini"] > mon_cfg.get("cluster_gini_alarm", 0.3):
        recommendations.append(
            f"⚠ Cluster gini={cl['gini']:.3f} > {mon_cfg.get('cluster_gini_alarm', 0.3)}: "
            f"建议开启 L_bal (beta_bal=0.01)"
        )
    if cl["dead_ratio"] > mon_cfg.get("cluster_dead_ratio_alarm", 0.1):
        recommendations.append(
            f"⚠ Dead clusters={cl['dead_ratio']:.1%}: "
            f"建议开启 L_bal 或降低 m"
        )

    if not recommendations:
        recommendations.append("✓ 所有指标正常，无需开启辅助损失")

    return {
        "codebook": cb,
        "cluster": cl,
        "recommendations": recommendations,
    }
