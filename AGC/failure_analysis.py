"""
失效分析模块: 诊断 AGC 模型 R@1 不足的根本原因。

分析维度:
  1. 得分间距分析    — R@1 miss 中 GT vs Pred 分数差距及排名深度分布
  2. 得分空间方差分析 — 模型对整个候选库的区分能力是否退化
  3. 聚类利用率分析  — m 个伪查询 token 的实际有效使用情况
  4. 视频表示区分度  — 压缩后不同视频表示之间的相似程度
"""
import logging

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("agc_failure")


# ─────────────── 1. 得分间距分析 ───────────────


def analyze_score_margins(all_scores, query_vid_map, video_ids):
    """
    分析 R@1 miss 中 Pred-GT 得分间距与排名深度分布。

    Returns:
        dict with margin stats, rank distribution, near-hit analysis
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    if isinstance(all_scores, np.ndarray):
        all_scores = torch.from_numpy(all_scores)

    N_q = all_scores.shape[0]
    miss_margins = []
    miss_gt_scores = []
    miss_pred_scores = []
    miss_ranks = []
    miss_gt_vids = []
    miss_pred_vids = []
    hit_gt_scores = []
    rank_buckets = {"rank1": 0, "rank2_5": 0, "rank6_20": 0, "rank21_100": 0, "rank_over100": 0}

    for q_idx in range(N_q):
        gt_vid = query_vid_map[q_idx]
        if gt_vid not in vid_to_idx:
            continue
        gt_idx = vid_to_idx[gt_vid]
        row = all_scores[q_idx]

        sorted_idx = torch.argsort(row, descending=True)
        pred_idx = sorted_idx[0].item()
        rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1

        gt_score = row[gt_idx].item()
        pred_score = row[pred_idx].item()

        if rank == 1:
            hit_gt_scores.append(gt_score)
            rank_buckets["rank1"] += 1
        else:
            miss_margins.append(pred_score - gt_score)
            miss_gt_scores.append(gt_score)
            miss_pred_scores.append(pred_score)
            miss_ranks.append(rank)
            miss_gt_vids.append(gt_vid)
            miss_pred_vids.append(video_ids[pred_idx])

            if rank <= 5:
                rank_buckets["rank2_5"] += 1
            elif rank <= 20:
                rank_buckets["rank6_20"] += 1
            elif rank <= 100:
                rank_buckets["rank21_100"] += 1
            else:
                rank_buckets["rank_over100"] += 1

    miss_margins = np.array(miss_margins) if miss_margins else np.array([0.0])
    miss_gt_scores = np.array(miss_gt_scores) if miss_gt_scores else np.array([0.0])
    miss_pred_scores = np.array(miss_pred_scores) if miss_pred_scores else np.array([0.0])
    miss_ranks = np.array(miss_ranks) if miss_ranks else np.array([1])
    hit_gt_scores = np.array(hit_gt_scores) if hit_gt_scores else np.array([0.0])
    total_miss = len(miss_gt_vids)

    return {
        "total_queries": N_q,
        "total_misses": total_miss,
        "miss_rate": total_miss / max(N_q, 1),
        "rank_distribution": {k: v / max(N_q, 1) for k, v in rank_buckets.items()},
        "near_hit_ratio": rank_buckets["rank2_5"] / max(total_miss, 1),
        "deep_miss_ratio": rank_buckets["rank_over100"] / max(total_miss, 1),
        "margin_mean": float(miss_margins.mean()),
        "margin_median": float(np.median(miss_margins)),
        "margin_p10": float(np.percentile(miss_margins, 10)),
        "margin_p90": float(np.percentile(miss_margins, 90)),
        "margin_lt0_5_ratio": float((miss_margins < 0.5).mean()),
        "margin_lt1_ratio": float((miss_margins < 1.0).mean()),
        "miss_gt_score_mean": float(miss_gt_scores.mean()),
        "miss_pred_score_mean": float(miss_pred_scores.mean()),
        "hit_gt_score_mean": float(hit_gt_scores.mean()),
        "miss_gt_rank_mean": float(miss_ranks.mean()),
        "miss_gt_rank_median": float(np.median(miss_ranks)),
        # 内部字段供后续分析调用，不写入 JSON
        "_miss_gt_vids": miss_gt_vids,
        "_miss_pred_vids": miss_pred_vids,
        "_miss_ranks": miss_ranks.tolist(),
    }


# ─────────────── 2. 得分空间方差分析 ───────────────


def analyze_score_variance(all_scores, query_vid_map, video_ids):
    """
    分析模型对候选视频库的区分能力。
    若所有候选视频得分相近(方差极低)，模型无法排序，排名将退化为随机。

    Returns:
        dict with per-query score std, GT percentile, flat-query ratio
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    if isinstance(all_scores, np.ndarray):
        all_scores = torch.from_numpy(all_scores)

    N_q = all_scores.shape[0]
    hit_stds, miss_stds = [], []
    hit_gt_pct, miss_gt_pct = [], []
    flat_count = 0

    for q_idx in range(N_q):
        gt_vid = query_vid_map[q_idx]
        if gt_vid not in vid_to_idx:
            continue
        gt_idx = vid_to_idx[gt_vid]
        row = all_scores[q_idx]

        row_std = row.std().item()
        row_mean_abs = row.abs().mean().item()

        gt_score = row[gt_idx].item()
        # GT 得分在所有候选中的分位数 (越高越好)
        gt_pct = (row < gt_score).float().mean().item()

        sorted_idx = torch.argsort(row, descending=True)
        rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1

        # 极平坦得分: std < 5% × 均值绝对值
        if row_mean_abs > 1e-6 and row_std < 0.05 * row_mean_abs:
            flat_count += 1

        if rank == 1:
            hit_stds.append(row_std)
            hit_gt_pct.append(gt_pct)
        else:
            miss_stds.append(row_std)
            miss_gt_pct.append(gt_pct)

    hit_stds = np.array(hit_stds) if hit_stds else np.array([0.0])
    miss_stds = np.array(miss_stds) if miss_stds else np.array([0.0])
    hit_gt_pct = np.array(hit_gt_pct) if hit_gt_pct else np.array([1.0])
    miss_gt_pct = np.array(miss_gt_pct) if miss_gt_pct else np.array([0.0])

    return {
        "flat_query_ratio": flat_count / max(N_q, 1),
        "hit_score_std_mean": float(hit_stds.mean()),
        "miss_score_std_mean": float(miss_stds.mean()),
        # std_gap > 0: hit 查询分数更分散 (正常); < 0: miss 查询更分散 (异常)
        "std_gap_hit_minus_miss": float(hit_stds.mean() - miss_stds.mean()),
        "hit_gt_percentile_mean": float(hit_gt_pct.mean()),
        "miss_gt_percentile_mean": float(miss_gt_pct.mean()),
        "miss_gt_percentile_median": float(np.median(miss_gt_pct)),
        # miss 中 GT 得分低于候选中位数的比例 → GT 被系统性低估
        "miss_gt_pct_lt50_ratio": float((miss_gt_pct < 0.5).mean()),
    }


# ─────────────── 3. 聚类利用率分析 ───────────────


def analyze_cluster_utilization(all_w):
    """
    分析 m 个伪查询聚类的实际有效利用率。

    注意: m 从 w.size(-1) 读取，不使用 codebook.size(0) (K)，避免混淆。

    Args:
        all_w: dict vid -> (n_frames, m) 软分配权重
    Returns:
        dict with cluster utilization stats
    """
    if not all_w:
        return {"analyzed": False}

    sample_w = next(iter(all_w.values()))
    m = sample_w.size(-1)   # 真实伪查询数量

    global_counts = torch.zeros(m)
    active_per_video = []
    # 每个聚类被多少视频使用
    cluster_video_usage = torch.zeros(m)

    for vid, w in all_w.items():
        hard = w.argmax(dim=-1)   # (n,)
        counts = torch.zeros(m)
        counts.scatter_add_(0, hard.cpu(), torch.ones_like(hard, dtype=torch.float).cpu())
        global_counts += counts

        active = hard.unique()
        active_per_video.append(len(active))

        used_mask = (counts > 0).float()
        cluster_video_usage += used_mask

    N_vids = len(all_w)
    active_per_video = np.array(active_per_video)

    # 崩塌聚类: 被 >80% 视频共用 → 对区分视频无贡献
    collapsed_thresh = 0.8 * N_vids
    collapsed_count = int((cluster_video_usage >= collapsed_thresh).sum().item())

    # 死聚类: 全局完全未使用
    dead_count = int((global_counts == 0).sum().item())

    # 聚类负载 Gini 系数
    load = global_counts.numpy()
    sorted_load = np.sort(load)
    idx = np.arange(1, m + 1)
    gini = float(max(0.0,
        2 * (idx * sorted_load).sum() / (m * max(sorted_load.sum(), 1e-8)) - (m + 1) / m
    ))
    top5_ratio = float(sorted_load[-5:].sum() / max(sorted_load.sum(), 1e-8))

    return {
        "analyzed": True,
        "m": m,
        "dead_clusters": dead_count,
        "dead_ratio": dead_count / m,
        "collapsed_clusters": collapsed_count,
        "collapsed_ratio": collapsed_count / m,
        "active_per_video_mean": float(active_per_video.mean()),
        "active_per_video_std": float(active_per_video.std()),
        "active_per_video_min": int(active_per_video.min()),
        "active_per_video_p25": float(np.percentile(active_per_video, 25)),
        "load_gini": gini,
        "top5_load_ratio": top5_ratio,
        "cluster_video_usage": cluster_video_usage.tolist(),
    }


# ─────────────── 4. 视频表示区分度分析 ───────────────


def analyze_representation_discrimination(all_c, miss_gt_vids, miss_pred_vids, video_ids,
                                           sample_size=300):
    """
    分析压缩后视频表示之间的相似度。
    相似度过高 -> 不同视频的压缩表示几乎相同 -> 模型无法区分。

    使用 mean-pool 后的余弦相似度作为视频级相似度代理指标。

    Args:
        all_c:          dict vid -> (m, h) 压缩视频表示
        miss_gt_vids:   list[str] R@1 miss 对应的 GT 视频 ID
        miss_pred_vids: list[str] R@1 miss 对应的 Pred 视频 ID
        video_ids:      list[str] 所有测试视频 ID
        sample_size:    全局成对相似度估计的视频采样数
    Returns:
        dict with inter-video similarity stats and miss-specific similarity
    """
    if not all_c:
        return {"analyzed": False, "reason": "all_c 未提供"}

    # ── 全局视频间相似度 (采样估计) ──
    sampled_vids = list(all_c.keys())
    if len(sampled_vids) > sample_size:
        rng = np.random.RandomState(42)
        sampled_vids = rng.choice(sampled_vids, sample_size, replace=False).tolist()

    sampled_reps = []
    for vid in sampled_vids:
        c = all_c[vid].float()
        rep = F.normalize(c.mean(dim=0), dim=0)   # (h,)
        sampled_reps.append(rep)
    reps_mat = torch.stack(sampled_reps)           # (S, h)

    sim_mat = reps_mat @ reps_mat.T                # (S, S)
    off_diag = ~torch.eye(len(sampled_vids), dtype=torch.bool)
    inter_sim = sim_mat[off_diag].cpu().numpy()

    # ── Miss 样本: GT 视频 vs Pred 视频相似度 ──
    miss_sims = []
    for gt_vid, pred_vid in zip(miss_gt_vids, miss_pred_vids):
        if gt_vid in all_c and pred_vid in all_c:
            gt_rep = F.normalize(all_c[gt_vid].float().mean(dim=0), dim=0)
            pred_rep = F.normalize(all_c[pred_vid].float().mean(dim=0), dim=0)
            miss_sims.append((gt_rep * pred_rep).sum().item())

    miss_sims = np.array(miss_sims) if miss_sims else np.array([0.0])

    return {
        "analyzed": True,
        "sample_size": len(sampled_vids),
        "inter_video_sim_mean": float(inter_sim.mean()),
        "inter_video_sim_median": float(np.median(inter_sim)),
        "inter_video_sim_p90": float(np.percentile(inter_sim, 90)),
        # 近似相同对比例 (>0.9 -> 压缩后几乎无差异)
        "high_sim_pair_ratio": float((inter_sim > 0.9).mean()),
        "miss_gt_pred_sim_mean": float(miss_sims.mean()),
        "miss_gt_pred_sim_median": float(np.median(miss_sims)),
        # miss 中 GT 与 Pred 相似度 >0.8 -> 确认压缩层无法区分这些视频
        "miss_high_sim_ratio": float((miss_sims > 0.8).mean()),
    }


# ─────────────── 综合诊断 ───────────────


def diagnose(margin, score_var, cluster_util, repr_disc):
    """综合四个分析维度，生成可解释的诊断报告。"""
    findings = []
    severity = 0

    # ── 维度 1: 分数间距 ──
    if margin["total_misses"] > 0:
        deep_miss = margin["deep_miss_ratio"]
        if deep_miss > 0.3:
            findings.append(
                f"[x] {deep_miss:.0%} 的 miss GT 排名超过 100 位 — "
                f"模型对这些查询完全失去定位能力"
            )
            severity += 3

        small_margin = margin["margin_lt1_ratio"]
        if small_margin > 0.6:
            findings.append(
                f"[x] {small_margin:.0%} 的 miss 分数间距 < 1.0 — "
                f"Pred 与 GT 得分几乎无差异，模型区分能力严重不足"
            )
            severity += 2
        elif small_margin > 0.3:
            findings.append(
                f"[!] {small_margin:.0%} 的 miss 分数间距 < 1.0 — 区分度存在不足"
            )
            severity += 1

        near_hit = margin["near_hit_ratio"]
        if near_hit > 0.25:
            findings.append(
                f"[i] {near_hit:.0%} 的 miss GT 排名在 2-5 位 — "
                f"存在较多近命中样本，小幅改进可显著提升 R@1"
            )

    # ── 维度 2: 分数方差 ──
    flat_r = score_var["flat_query_ratio"]
    if flat_r > 0.2:
        findings.append(
            f"[x] {flat_r:.0%} 的查询对候选库得分极平坦 — "
            f"模型输出近似均匀分布，排名退化为随机"
        )
        severity += 3

    pct_lt50 = score_var["miss_gt_pct_lt50_ratio"]
    if pct_lt50 > 0.5:
        findings.append(
            f"[x] {pct_lt50:.0%} 的 miss 中 GT 得分低于候选中位数 — "
            f"GT 视频被系统性低分，非噪声问题"
        )
        severity += 2

    std_gap = score_var["std_gap_hit_minus_miss"]
    if std_gap < -0.01:
        findings.append(
            f"[!] Miss 查询得分标准差高于 Hit 查询 — "
            f"失败原因是错误视频得分异常偏高，非全局平坦"
        )
        severity += 1

    # ── 维度 3: 聚类利用率 ──
    if cluster_util.get("analyzed"):
        m = cluster_util["m"]
        collapsed_r = cluster_util["collapsed_ratio"]
        if collapsed_r > 0.4:
            findings.append(
                f"[x] {collapsed_r:.0%} 的聚类被 >80% 视频共用 (严重聚类崩塌) — "
                f"压缩表示丧失跨视频区分性"
            )
            severity += 3

        active_mean = cluster_util["active_per_video_mean"]
        if active_mean < m * 0.5:
            findings.append(
                f"[!] 每视频平均仅 {active_mean:.1f}/{m} 个聚类活跃 — "
                f"超过半数的伪查询 token 未发挥作用"
            )
            severity += 2

        gini = cluster_util["load_gini"]
        if gini > 0.6:
            findings.append(
                f"[!] 聚类负载 Gini={gini:.3f} — "
                f"少数聚类承载绝大多数 token，伪查询容量严重浪费"
            )
            severity += 1

    # ── 维度 4: 表示区分度 ──
    if repr_disc.get("analyzed"):
        high_sim_pair = repr_disc["high_sim_pair_ratio"]
        if high_sim_pair > 0.1:
            findings.append(
                f"[x] {high_sim_pair:.0%} 的视频对余弦相似度 > 0.9 — "
                f"大量视频压缩后表示近乎相同"
            )
            severity += 3

        inter_mean = repr_disc["inter_video_sim_mean"]
        if inter_mean > 0.7:
            findings.append(
                f"[x] 视频间平均余弦相似度 = {inter_mean:.3f} — "
                f"压缩表示高度同质化，丧失个体视频特征"
            )
            severity += 2
        elif inter_mean > 0.5:
            findings.append(
                f"[!] 视频间平均余弦相似度 = {inter_mean:.3f} — 表示区分度偏低"
            )
            severity += 1

        miss_high_sim = repr_disc["miss_high_sim_ratio"]
        if miss_high_sim > 0.3:
            findings.append(
                f"[x] {miss_high_sim:.0%} 的 R@1 miss 中 GT 与 Pred 视频余弦相似度 > 0.8 — "
                f"这些失效直接由压缩层无法区分视频所致"
            )
            severity += 2

    if not findings:
        findings.append("[ok] 未检测到明显性能瓶颈，R@1 不足可能来自数据噪声或模型容量上限")

    severity = min(severity, 10)
    if severity >= 7:
        severity_label = "严重 — 建议: 开启 L_bal+L_div 辅助损失，调高 tau_end (>=0.3)，或增大 num_pseudo_queries"
    elif severity >= 4:
        severity_label = "中等 — 建议调整辅助损失权重，重点关注聚类崩塌问题"
    elif severity >= 2:
        severity_label = "轻微 — 可针对 near-hit 样本或边界负样本做定向优化"
    else:
        severity_label = "正常"

    return {
        "findings": findings,
        "severity": severity_label,
        "severity_score": severity,
    }


# ─────────────── 综合失效分析入口 ───────────────


def run_failure_analysis(
    all_scores,
    query_vid_map,
    video_ids,
    all_w,
    codebook=None,
    all_c=None,
    logger_fn=None,
):
    """
    运行完整失效分析，诊断 R@1 不足的根本原因。

    Args:
        all_scores:    (N_q, N_v) 得分矩阵
        query_vid_map: list[str] 第 i 个 query 对应的 GT video_id
        video_ids:     list[str] 所有候选视频 ID
        all_w:         dict vid -> (n_frames, m) 软分配权重 (m 从此读取)
        codebook:      (K, h) 保留参数，兼容旧调用方，分析中不再使用
        all_c:         dict vid -> (m, h) 压缩视频表示 (可选，用于区分度分析)
        logger_fn:     可选日志函数

    Returns:
        report: dict 包含四个维度的分析结果及综合诊断
    """
    log = logger_fn or logger.info

    # ──── 1. 得分间距分析 ────
    log("[失效分析] === 1. 得分间距分析 ===")
    margin = analyze_score_margins(all_scores, query_vid_map, video_ids)
    log(f"  总查询: {margin['total_queries']}, "
        f"R@1 Miss: {margin['total_misses']} ({margin['miss_rate']:.1%})")
    log(f"  排名分布: " +
        ", ".join(f"{k}={v:.1%}" for k, v in margin["rank_distribution"].items()))
    log(f"  近命中(rank 2-5): {margin['near_hit_ratio']:.1%}  "
        f"深度miss(rank>100): {margin['deep_miss_ratio']:.1%}")
    log(f"  分数间距: mean={margin['margin_mean']:.3f}  "
        f"median={margin['margin_median']:.3f}  "
        f"p10={margin['margin_p10']:.3f}  p90={margin['margin_p90']:.3f}")
    log(f"  间距<0.5: {margin['margin_lt0_5_ratio']:.1%}  "
        f"间距<1.0: {margin['margin_lt1_ratio']:.1%}")
    log(f"  GT score — Hit: {margin['hit_gt_score_mean']:.3f}  "
        f"Miss: {margin['miss_gt_score_mean']:.3f}")
    log(f"  Miss GT 排名: mean={margin['miss_gt_rank_mean']:.1f}  "
        f"median={margin['miss_gt_rank_median']:.1f}")

    # ──── 2. 得分空间方差分析 ────
    log("[失效分析] === 2. 得分空间方差分析 ===")
    score_var = analyze_score_variance(all_scores, query_vid_map, video_ids)
    log(f"  极平坦得分查询占比: {score_var['flat_query_ratio']:.1%}")
    log(f"  得分 std — Hit: {score_var['hit_score_std_mean']:.4f}  "
        f"Miss: {score_var['miss_score_std_mean']:.4f}  "
        f"(差值: {score_var['std_gap_hit_minus_miss']:+.4f})")
    log(f"  GT 百分位 — Hit: {score_var['hit_gt_percentile_mean']:.3f}  "
        f"Miss均值: {score_var['miss_gt_percentile_mean']:.3f}  "
        f"Miss中位: {score_var['miss_gt_percentile_median']:.3f}")
    log(f"  Miss 中 GT 低于候选中位数比例: {score_var['miss_gt_pct_lt50_ratio']:.1%}")

    # ──── 3. 聚类利用率分析 ────
    log("[失效分析] === 3. 聚类利用率分析 ===")
    cluster_util = analyze_cluster_utilization(all_w)
    if cluster_util["analyzed"]:
        m = cluster_util["m"]
        log(f"  聚类总数 m={m}  (从权重维度读取，非 codebook K)")
        log(f"  死聚类: {cluster_util['dead_clusters']}/{m} ({cluster_util['dead_ratio']:.1%})")
        log(f"  崩塌聚类(>80%视频共用): "
            f"{cluster_util['collapsed_clusters']}/{m} ({cluster_util['collapsed_ratio']:.1%})")
        log(f"  每视频活跃聚类: mean={cluster_util['active_per_video_mean']:.1f}  "
            f"std={cluster_util['active_per_video_std']:.1f}  "
            f"min={cluster_util['active_per_video_min']}  "
            f"p25={cluster_util['active_per_video_p25']:.1f}")
        log(f"  负载 Gini={cluster_util['load_gini']:.3f}  "
            f"Top-5 负载占比: {cluster_util['top5_load_ratio']:.1%}")
    else:
        log("  [跳过] all_w 未提供")

    # ──── 4. 视频表示区分度分析 ────
    log("[失效分析] === 4. 视频表示区分度分析 ===")
    repr_disc = analyze_representation_discrimination(
        all_c,
        margin["_miss_gt_vids"],
        margin["_miss_pred_vids"],
        video_ids,
    )
    if repr_disc["analyzed"]:
        log(f"  采样视频数: {repr_disc['sample_size']}")
        log(f"  视频间余弦相似度: mean={repr_disc['inter_video_sim_mean']:.3f}  "
            f"median={repr_disc['inter_video_sim_median']:.3f}  "
            f"p90={repr_disc['inter_video_sim_p90']:.3f}")
        log(f"  高相似度视频对(>0.9)占比: {repr_disc['high_sim_pair_ratio']:.1%}")
        log(f"  Miss GT-Pred 相似度: mean={repr_disc['miss_gt_pred_sim_mean']:.3f}  "
            f"median={repr_disc['miss_gt_pred_sim_median']:.3f}")
        log(f"  Miss 中 GT-Pred 相似度>0.8 占比: {repr_disc['miss_high_sim_ratio']:.1%}")
    else:
        log(f"  [跳过] {repr_disc.get('reason', 'all_c 未提供')}")

    # ──── 综合诊断 ────
    log("[失效分析] === 综合诊断 ===")
    diagnosis = diagnose(margin, score_var, cluster_util, repr_disc)
    for line in diagnosis["findings"]:
        log(f"  {line}")
    log(f"  严重程度: {diagnosis['severity']}")

    report = {
        "margin_analysis": {k: v for k, v in margin.items() if not k.startswith("_")},
        "score_variance": score_var,
        "cluster_utilization": {k: v for k, v in cluster_util.items()
                                 if k != "cluster_video_usage"},
        "representation_discrimination": repr_disc,
        "diagnosis": diagnosis,
    }
    return report
