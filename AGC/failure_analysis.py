"""
失效分析模块: 诊断 R@1 miss 的根因，重点检测类间区分度不足。

分析维度:
  1. GT/Pred 伪查询是否落入同一聚类（聚类重叠度）
  2. Query→GT vs Query→Pred 的 MaxSim 分数对比（得分间距分析）
  3. 死聚类占比（全局 + per-video）
  4. 综合诊断: 区分度不足指标汇总
"""
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("agc_failure")


# ─────────────── 核心分析函数 ───────────────


def compute_per_video_cluster_stats(all_w: dict):
    """
    计算每个视频的聚类统计。

    Args:
        all_w: dict vid → (n, m) 软分配权重
    Returns:
        dict vid → {
            'hard_assign': (n,) 硬分配标签,
            'active_clusters': set of active cluster indices,
            'dominant_cluster': int,  # 最多 token 被分配到的聚类
            'cluster_load': (m,) 各聚类 token 数量,
        }
    """
    stats = {}
    for vid, w in all_w.items():
        # w: (n, m)
        hard = w.argmax(dim=-1)  # (n,)
        m = w.size(1)
        counts = torch.zeros(m)
        counts.scatter_add_(0, hard.cpu(), torch.ones_like(hard, dtype=torch.float).cpu())

        active = set(hard.unique().cpu().tolist())
        dominant = counts.argmax().item()

        stats[vid] = {
            "hard_assign": hard.cpu(),
            "active_clusters": active,
            "dominant_cluster": dominant,
            "cluster_load": counts,
        }
    return stats


def analyze_cluster_overlap(gt_stats, pred_stats):
    """
    分析 GT 视频和 Pred 视频的聚类重叠度。

    Returns:
        overlap_info: dict with IoU, same_dominant, shared_ratio
    """
    gt_active = gt_stats["active_clusters"]
    pred_active = pred_stats["active_clusters"]

    intersection = gt_active & pred_active
    union = gt_active | pred_active

    iou = len(intersection) / max(len(union), 1)
    same_dominant = gt_stats["dominant_cluster"] == pred_stats["dominant_cluster"]

    # 加权重叠: 考虑 token 数量
    gt_load = gt_stats["cluster_load"]
    pred_load = pred_stats["cluster_load"]
    gt_norm = gt_load / gt_load.sum().clamp(min=1e-8)
    pred_norm = pred_load / pred_load.sum().clamp(min=1e-8)
    weighted_overlap = torch.min(gt_norm, pred_norm).sum().item()

    return {
        "iou": iou,
        "same_dominant": same_dominant,
        "shared_active_count": len(intersection),
        "gt_active_count": len(gt_active),
        "pred_active_count": len(pred_active),
        "weighted_overlap": weighted_overlap,
    }


def compute_dead_cluster_ratio_global(all_w: dict, m: int):
    """
    计算全局死聚类占比: 在整个数据集中从未被任何视频 token 选为硬分配的聚类。

    Args:
        all_w: dict vid → (n, m) 软分配权重
        m: 聚类总数
    Returns:
        dead_ratio, dead_indices, global_counts
    """
    global_counts = torch.zeros(m)
    for vid, w in all_w.items():
        hard = w.argmax(dim=-1)  # (n,)
        counts = torch.zeros(m)
        counts.scatter_add_(0, hard.cpu(), torch.ones_like(hard, dtype=torch.float).cpu())
        global_counts += counts

    dead_mask = global_counts == 0
    dead_ratio = dead_mask.float().mean().item()
    dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()

    return dead_ratio, dead_indices, global_counts


def analyze_maxsim_margins(all_scores, query_vid_map, video_ids):
    """
    分析 Query→GT 和 Query→Pred 的 MaxSim 得分差距。

    Args:
        all_scores: (N_q, N_v) 得分矩阵
        query_vid_map: list[str] - 每个 query 对应的 GT video_id
        video_ids: list[str]
    Returns:
        analysis: dict with margin statistics for R@1 misses
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = all_scores.shape[0]

    if isinstance(all_scores, np.ndarray):
        all_scores = torch.from_numpy(all_scores)

    miss_margins = []       # pred_score - gt_score for R@1 misses
    miss_gt_scores = []     # GT score for misses
    miss_pred_scores = []   # Pred score for misses
    miss_ranks = []         # GT rank for misses
    miss_pred_vids = []     # Pred video id for misses
    miss_gt_vids = []       # GT video id for misses
    hit_gt_scores = []      # GT score for hits

    for q_idx in range(N_q):
        gt_vid = query_vid_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        row = all_scores[q_idx]

        sorted_idx = torch.argsort(row, descending=True)
        pred_idx = sorted_idx[0].item()
        rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1

        gt_score = row[gt_idx].item()
        pred_score = row[pred_idx].item()

        if rank == 1:
            hit_gt_scores.append(gt_score)
        else:
            margin = pred_score - gt_score
            miss_margins.append(margin)
            miss_gt_scores.append(gt_score)
            miss_pred_scores.append(pred_score)
            miss_ranks.append(rank)
            miss_pred_vids.append(video_ids[pred_idx])
            miss_gt_vids.append(gt_vid)

    miss_margins = np.array(miss_margins) if miss_margins else np.array([0.0])
    miss_gt_scores = np.array(miss_gt_scores) if miss_gt_scores else np.array([0.0])
    miss_pred_scores = np.array(miss_pred_scores) if miss_pred_scores else np.array([0.0])
    miss_ranks = np.array(miss_ranks) if miss_ranks else np.array([1])
    hit_gt_scores = np.array(hit_gt_scores) if hit_gt_scores else np.array([0.0])

    # 间距分布: 有多少 miss 的 margin 很小 (→ 区分度不足)
    total_miss = len(miss_margins)
    margin_bins = {
        "margin<0.5": (miss_margins < 0.5).sum() / max(total_miss, 1),
        "margin<1.0": (miss_margins < 1.0).sum() / max(total_miss, 1),
        "margin<2.0": (miss_margins < 2.0).sum() / max(total_miss, 1),
        "margin<5.0": (miss_margins < 5.0).sum() / max(total_miss, 1),
    }

    return {
        "total_queries": N_q,
        "total_misses": total_miss,
        "miss_rate": total_miss / max(N_q, 1),
        "margin_mean": float(miss_margins.mean()),
        "margin_median": float(np.median(miss_margins)),
        "margin_std": float(miss_margins.std()),
        "margin_p90": float(np.percentile(miss_margins, 90)),
        "margin_distribution": {k: float(v) for k, v in margin_bins.items()},
        "miss_gt_score_mean": float(miss_gt_scores.mean()),
        "miss_pred_score_mean": float(miss_pred_scores.mean()),
        "hit_gt_score_mean": float(hit_gt_scores.mean()),
        "miss_gt_rank_mean": float(miss_ranks.mean()),
        "miss_gt_rank_median": float(np.median(miss_ranks)),
        # 原始数据（用于聚类分析联动）
        "_miss_gt_vids": miss_gt_vids,
        "_miss_pred_vids": miss_pred_vids,
        "_miss_margins": miss_margins.tolist(),
        "_miss_ranks": miss_ranks.tolist(),
    }


# ─────────────── 综合失效分析 ───────────────


def run_failure_analysis(
    all_scores,
    query_vid_map,
    video_ids,
    all_w,
    codebook,
    logger_fn=None,
):
    """
    运行完整失效分析。

    Args:
        all_scores: (N_q, N_v) 得分矩阵
        query_vid_map: list[str] — 第 i 个 query 对应的 GT video_id
        video_ids: list[str] — 候选视频列表
        all_w: dict vid → (n, m) 软分配权重
        codebook: (K, h) 密码本张量 (detached)
        logger_fn: 可选的 log 函数，默认用 module logger

    Returns:
        report: dict 包含所有分析结果
    """
    log = logger_fn or logger.info

    m = codebook.size(0) if codebook is not None else 0

    # ──── 1. MaxSim 间距分析 ────
    margin_analysis = analyze_maxsim_margins(all_scores, query_vid_map, video_ids)

    log(f"  [失效分析] === MaxSim 间距分析 ===")
    log(f"    总查询: {margin_analysis['total_queries']}, "
        f"R@1 Miss: {margin_analysis['total_misses']} "
        f"({margin_analysis['miss_rate']:.1%})")
    log(f"    Pred-GT 间距: mean={margin_analysis['margin_mean']:.3f}, "
        f"median={margin_analysis['margin_median']:.3f}, "
        f"std={margin_analysis['margin_std']:.3f}, "
        f"p90={margin_analysis['margin_p90']:.3f}")
    log(f"    间距分布: {margin_analysis['margin_distribution']}")
    log(f"    Miss GT score mean: {margin_analysis['miss_gt_score_mean']:.3f}, "
        f"Miss Pred score mean: {margin_analysis['miss_pred_score_mean']:.3f}, "
        f"Hit GT score mean: {margin_analysis['hit_gt_score_mean']:.3f}")
    log(f"    Miss GT rank: mean={margin_analysis['miss_gt_rank_mean']:.1f}, "
        f"median={margin_analysis['miss_gt_rank_median']:.1f}")

    # ──── 2. GT/Pred 聚类重叠分析 ────
    cluster_overlap_report = {"analyzed": False}
    if all_w and margin_analysis["total_misses"] > 0:
        per_video_stats = compute_per_video_cluster_stats(all_w)
        miss_gt_vids = margin_analysis["_miss_gt_vids"]
        miss_pred_vids = margin_analysis["_miss_pred_vids"]

        overlap_ious = []
        same_dominant_count = 0
        weighted_overlaps = []
        analyzable = 0

        for gt_vid, pred_vid in zip(miss_gt_vids, miss_pred_vids):
            if gt_vid in per_video_stats and pred_vid in per_video_stats:
                overlap = analyze_cluster_overlap(
                    per_video_stats[gt_vid], per_video_stats[pred_vid]
                )
                overlap_ious.append(overlap["iou"])
                weighted_overlaps.append(overlap["weighted_overlap"])
                if overlap["same_dominant"]:
                    same_dominant_count += 1
                analyzable += 1

        if analyzable > 0:
            overlap_ious = np.array(overlap_ious)
            weighted_overlaps = np.array(weighted_overlaps)

            cluster_overlap_report = {
                "analyzed": True,
                "analyzable_misses": analyzable,
                "cluster_iou_mean": float(overlap_ious.mean()),
                "cluster_iou_median": float(np.median(overlap_ious)),
                "cluster_iou_p75": float(np.percentile(overlap_ious, 75)),
                "same_dominant_ratio": same_dominant_count / analyzable,
                "weighted_overlap_mean": float(weighted_overlaps.mean()),
                "high_overlap_ratio": float((overlap_ious > 0.5).mean()),
            }

            log(f"  [失效分析] === GT/Pred 聚类重叠分析 ===")
            log(f"    可分析 miss 样本: {analyzable}")
            log(f"    聚类 IoU: mean={cluster_overlap_report['cluster_iou_mean']:.3f}, "
                f"median={cluster_overlap_report['cluster_iou_median']:.3f}, "
                f"p75={cluster_overlap_report['cluster_iou_p75']:.3f}")
            log(f"    同主聚类占比: {cluster_overlap_report['same_dominant_ratio']:.1%}")
            log(f"    加权重叠均值: {cluster_overlap_report['weighted_overlap_mean']:.3f}")
            log(f"    高重叠(IoU>0.5)占比: {cluster_overlap_report['high_overlap_ratio']:.1%}")

    # ──── 3. 死聚类分析 ────
    dead_report = {"analyzed": False}
    if all_w and m > 0:
        dead_ratio, dead_indices, global_counts = compute_dead_cluster_ratio_global(all_w, m)

        # 每视频平均活跃聚类数
        active_per_video = []
        for vid, w in all_w.items():
            hard = w.argmax(dim=-1)
            active_per_video.append(len(hard.unique()))
        active_per_video = np.array(active_per_video)

        # 聚类负载 Gini 系数
        load = global_counts.numpy()
        sorted_load = np.sort(load)
        idx = np.arange(1, m + 1)
        gini = (2 * (idx * sorted_load).sum() / (m * sorted_load.sum() + 1e-8) - (m + 1) / m)
        gini = max(0.0, gini)

        dead_report = {
            "analyzed": True,
            "dead_ratio": dead_ratio,
            "dead_count": len(dead_indices),
            "total_clusters": m,
            "dead_indices": dead_indices[:20],  # 最多展示 20 个
            "global_load_gini": float(gini),
            "active_per_video_mean": float(active_per_video.mean()),
            "active_per_video_std": float(active_per_video.std()),
            "active_per_video_min": int(active_per_video.min()),
            "top5_cluster_load_ratio": float(sorted_load[-5:].sum() / sorted_load.sum()) if sorted_load.sum() > 0 else 0.0,
        }

        log(f"  [失效分析] === 死聚类分析 ===")
        log(f"    死聚类: {dead_report['dead_count']}/{m} ({dead_ratio:.1%})")
        log(f"    全局负载 Gini: {gini:.3f}")
        log(f"    每视频活跃聚类: mean={dead_report['active_per_video_mean']:.1f}, "
            f"std={dead_report['active_per_video_std']:.1f}, "
            f"min={dead_report['active_per_video_min']}")
        log(f"    Top-5 聚类负载占比: {dead_report['top5_cluster_load_ratio']:.1%}")

    # ──── 4. 综合诊断: 类间区分度 ────
    diagnosis = diagnose_discrimination(margin_analysis, cluster_overlap_report, dead_report)
    log(f"  [失效分析] === 综合诊断 ===")
    for line in diagnosis["findings"]:
        log(f"    {line}")
    log(f"    严重程度: {diagnosis['severity']}")

    report = {
        "margin_analysis": {k: v for k, v in margin_analysis.items() if not k.startswith("_")},
        "cluster_overlap": cluster_overlap_report,
        "dead_clusters": dead_report,
        "diagnosis": diagnosis,
    }
    return report


def diagnose_discrimination(margin_analysis, cluster_overlap, dead_report):
    """
    综合诊断: 判断 R@1 低是否由类间区分度不足导致。

    Returns:
        dict with findings list and severity level
    """
    findings = []
    severity_score = 0  # 0-10, 越高问题越严重

    miss_rate = margin_analysis["miss_rate"]

    # 诊断 1: 间距过小 → 区分度不足的直接证据
    if margin_analysis["total_misses"] > 0:
        small_margin_ratio = margin_analysis["margin_distribution"].get("margin<1.0", 0)
        if small_margin_ratio > 0.7:
            findings.append(
                f"⚠ {small_margin_ratio:.0%} 的 R@1 miss 间距 < 1.0 — "
                f"模型对 GT 和 Pred 的区分能力严重不足"
            )
            severity_score += 3
        elif small_margin_ratio > 0.4:
            findings.append(
                f"△ {small_margin_ratio:.0%} 的 R@1 miss 间距 < 1.0 — "
                f"存在一定区分度不足"
            )
            severity_score += 2

        very_small = margin_analysis["margin_distribution"].get("margin<0.5", 0)
        if very_small > 0.5:
            findings.append(
                f"⚠ {very_small:.0%} 的 miss 间距 < 0.5 — GT 和 Pred 几乎不可分"
            )
            severity_score += 2

    # 诊断 2: hit vs miss 的 GT 得分差异
    if margin_analysis["total_misses"] > 0:
        score_gap = margin_analysis["hit_gt_score_mean"] - margin_analysis["miss_gt_score_mean"]
        if score_gap > 5.0:
            findings.append(
                f"△ Hit GT score ({margin_analysis['hit_gt_score_mean']:.1f}) 显著高于 "
                f"Miss GT score ({margin_analysis['miss_gt_score_mean']:.1f})，"
                f"说明部分视频本身表示质量低"
            )

    # 诊断 3: 聚类重叠
    if cluster_overlap.get("analyzed"):
        iou_mean = cluster_overlap["cluster_iou_mean"]
        same_dom = cluster_overlap["same_dominant_ratio"]
        high_overlap = cluster_overlap["high_overlap_ratio"]

        if same_dom > 0.5:
            findings.append(
                f"⚠ {same_dom:.0%} 的 miss 中 GT 和 Pred 主聚类相同 — "
                f"伪查询聚类未能区分不同视频"
            )
            severity_score += 3

        if high_overlap > 0.6:
            findings.append(
                f"⚠ {high_overlap:.0%} 的 miss 聚类 IoU > 0.5 — "
                f"不同视频的聚类模式高度相似，类间区分度严重不足"
            )
            severity_score += 2

        if iou_mean > 0.4:
            findings.append(
                f"△ 平均聚类 IoU = {iou_mean:.3f}，不同视频共享过多聚类"
            )
            severity_score += 1

    # 诊断 4: 死聚类
    if dead_report.get("analyzed"):
        dead_ratio = dead_report["dead_ratio"]
        if dead_ratio > 0.3:
            findings.append(
                f"⚠ 死聚类高达 {dead_ratio:.0%} — "
                f"大量聚类未被使用，有效表达能力受限"
            )
            severity_score += 2
        elif dead_ratio > 0.1:
            findings.append(
                f"△ 死聚类 {dead_ratio:.0%}，部分聚类浪费"
            )
            severity_score += 1

        if dead_report["global_load_gini"] > 0.5:
            findings.append(
                f"△ 聚类负载 Gini = {dead_report['global_load_gini']:.3f}，聚类使用极不均衡"
            )
            severity_score += 1

    # 总结
    if not findings:
        findings.append("✓ 未检测到明显的类间区分度问题")

    severity_score = min(severity_score, 10)
    if severity_score >= 7:
        severity = "严重 — 强烈建议开启 L_div + L_bal，或增大 codebook_size/num_pseudo_queries"
    elif severity_score >= 4:
        severity = "中等 — 建议开启辅助损失调节"
    elif severity_score >= 2:
        severity = "轻微 — 可观察后续训练趋势"
    else:
        severity = "正常"

    return {
        "findings": findings,
        "severity": severity,
        "severity_score": severity_score,
    }
