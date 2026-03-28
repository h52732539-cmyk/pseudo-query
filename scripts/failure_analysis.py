"""
失效分析脚本：深入分析方式C匹配失败的case。

分析维度：
  1. 按rank分桶的全量统计
  2. GT/Pred伪查询是否在同一聚类
  3. Query到GT/Pred伪查询的距离
  4. GT/Pred伪查询到聚类中心(原型)的距离
  5. 逐原型得分分解（仅失败case，top-10关键原型）
  6. 自动失效模式分类（同聚类混淆/跨聚类偏离/query泛化/聚类拥挤）

用法:
    python scripts/failure_analysis.py --split test
    python scripts/failure_analysis.py --split val
    python scripts/failure_analysis.py --split test --max_failure_details 5000
"""
import argparse
import json
import logging
import sys
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids
from models.clip_encoder import CLIPTextEncoder
from train import encode_video_captions, build_model
from scripts.diagnose_matching import (
    encode_all_videos, encode_all_queries, compute_scores_method_c,
    compute_retrieval_metrics,
)

logger = logging.getLogger("failure_analysis")


# ============================================================
# 聚类信息
# ============================================================

def build_cluster_info(all_mu):
    """
    构建聚类信息。
    聚类定义: cluster(video_i) = argmax(μ_i)
    Returns:
        cluster_ids: (N_v,) 每个视频的聚类ID
        cluster_sizes: dict {cluster_id: count}
    """
    cluster_ids = torch.argmax(all_mu, dim=1).numpy()  # (N_v,)
    cluster_sizes = {}
    for c in cluster_ids:
        cluster_sizes[int(c)] = cluster_sizes.get(int(c), 0) + 1
    return cluster_ids, cluster_sizes


# ============================================================
# 逐原型得分分解
# ============================================================

def compute_per_proto_contributions(h_gt, h_pred, q_tilde, w):
    """
    计算单条query对GT和Pred伪查询的逐原型得分贡献。
    contrib[k] = w[k] * cos(h[k], q_tilde[k])

    Args:
        h_gt: (K, d) — GT伪查询的per-prototype特征
        h_pred: (K, d) — Pred伪查询的per-prototype特征
        q_tilde: (K, d) — Query的per-prototype语义
        w: (K,) — L1归一化的权重
    Returns:
        contrib_gt: (K,)
        contrib_pred: (K,)
    """
    h_gt_n = F.normalize(h_gt, p=2, dim=-1)
    h_pred_n = F.normalize(h_pred, p=2, dim=-1)
    q_n = F.normalize(q_tilde, p=2, dim=-1)

    # cos(h^k, q^k) for each prototype k
    cos_gt = (h_gt_n * q_n).sum(dim=-1)      # (K,)
    cos_pred = (h_pred_n * q_n).sum(dim=-1)   # (K,)

    contrib_gt = w * cos_gt       # (K,)
    contrib_pred = w * cos_pred   # (K,)
    return contrib_gt, contrib_pred


def get_top_k_diff_protos(contrib_gt, contrib_pred, k=10):
    """
    找出导致Pred胜过GT的top-k原型。
    diff[k] = contrib_pred[k] - contrib_gt[k]，取diff最大的k个。
    """
    diff = contrib_pred - contrib_gt  # (K,)
    topk_vals, topk_ids = torch.topk(diff, k)
    return [
        {"proto_id": int(topk_ids[i]), "diff": round(float(topk_vals[i]), 6),
         "contrib_gt": round(float(contrib_gt[topk_ids[i]]), 6),
         "contrib_pred": round(float(contrib_pred[topk_ids[i]]), 6)}
        for i in range(k)
    ]


# ============================================================
# Query 激活集中度 (entropy)
# ============================================================

def compute_activation_entropy(s_T):
    """
    计算 s_T 的 entropy（衡量query激活的分散程度）。
    高 entropy = query 均匀激活多个原型（泛化/模糊）。
    低 entropy = query 集中在少数原型（具体/明确）。
    """
    # s_T 已经 L2 归一化，先转为概率分布
    p = F.softmax(s_T, dim=-1)  # (N, K)
    log_p = torch.log(p + 1e-10)
    entropy = -(p * log_p).sum(dim=-1)  # (N,)
    return entropy


# ============================================================
# 与原型中心的对齐度
# ============================================================

def compute_proto_alignment(h, prototypes, cluster_id):
    """
    计算 per-prototype 特征 h 与原型中心的对齐度。
    重点关注归属聚类对应的原型。
    
    Args:
        h: (K, d) — per-prototype特征
        prototypes: (K, d) — L2归一化的原型中心
        cluster_id: int — 归属聚类ID
    Returns:
        cos_assigned: float — 与归属聚类原型的cosine
        cos_mean_all: float — 与所有原型的平均cosine
    """
    h_n = F.normalize(h, p=2, dim=-1)  # (K, d)
    # 每个h^k与对应proto_k的cosine
    cos_per_proto = (h_n * prototypes).sum(dim=-1)  # (K,)
    cos_assigned = float(cos_per_proto[cluster_id])
    cos_mean_all = float(cos_per_proto.mean())
    return cos_assigned, cos_mean_all


# ============================================================
# 按rank分桶统计
# ============================================================

RANK_BUCKETS = [
    ("rank=1", 1, 1),
    ("rank∈[2,5]", 2, 5),
    ("rank∈[6,10]", 6, 10),
    ("rank∈[11,100]", 11, 100),
    ("rank>100", 101, float("inf")),
]


def bucket_statistics(ranks, scores, query_vid_map, video_ids,
                      all_mu, all_s_T, cluster_ids, cluster_sizes):
    """
    按rank分桶计算聚合统计。
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = len(ranks)
    K = all_s_T.shape[1]

    # query激活entropy
    q_entropy = compute_activation_entropy(all_s_T).numpy()

    bucket_stats = []
    for name, lo, hi in RANK_BUCKETS:
        mask = (ranks >= lo) & (ranks <= hi)
        count = int(mask.sum())
        if count == 0:
            bucket_stats.append({"bucket": name, "count": 0, "ratio": 0.0})
            continue

        indices = np.where(mask)[0]

        # 同聚类比例
        same_cluster_count = 0
        gt_scores_list = []
        pred_scores_list = []
        score_gaps = []
        gt_cluster_sizes_list = []
        entropies = []

        for q_idx in indices:
            gt_vid = query_vid_map[q_idx]
            gt_v_idx = vid_to_idx[gt_vid]
            pred_v_idx = int(torch.argmax(scores[q_idx]).item())

            gt_cluster = int(cluster_ids[gt_v_idx])
            pred_cluster = int(cluster_ids[pred_v_idx])
            if gt_cluster == pred_cluster:
                same_cluster_count += 1

            gt_score = float(scores[q_idx, gt_v_idx])
            pred_score = float(scores[q_idx, pred_v_idx])
            gt_scores_list.append(gt_score)
            pred_scores_list.append(pred_score)
            score_gaps.append(pred_score - gt_score)

            gt_cluster_sizes_list.append(cluster_sizes.get(gt_cluster, 0))
            entropies.append(float(q_entropy[q_idx]))

        bucket_stats.append({
            "bucket": name,
            "count": count,
            "ratio": round(count / N_q * 100, 2),
            "same_cluster_ratio": round(same_cluster_count / count * 100, 2),
            "gt_score_mean": round(float(np.mean(gt_scores_list)), 4),
            "gt_score_median": round(float(np.median(gt_scores_list)), 4),
            "pred_score_mean": round(float(np.mean(pred_scores_list)), 4),
            "pred_score_median": round(float(np.median(pred_scores_list)), 4),
            "score_gap_mean": round(float(np.mean(score_gaps)), 4),
            "score_gap_median": round(float(np.median(score_gaps)), 4),
            "gt_cluster_size_mean": round(float(np.mean(gt_cluster_sizes_list)), 1),
            "query_entropy_mean": round(float(np.mean(entropies)), 4),
            "query_entropy_median": round(float(np.median(entropies)), 4),
        })

    return bucket_stats


# ============================================================
# 失效模式分类
# ============================================================

def classify_failure_mode(gt_cluster, pred_cluster, gt_cluster_size,
                          query_entropy, entropy_threshold, crowd_threshold):
    """
    将每条失败case归类为失效模式：
    - A: 同聚类混淆 — GT和Pred在同一聚类
    - B: 跨聚类偏离 — GT和Pred在不同聚类
    - C: query过于泛化 — entropy高于阈值（在A/B基础上叠加）
    - D: 聚类拥挤 — GT聚类大小高于阈值（在A/B基础上叠加）
    返回主类型(A/B) + 附加标签列表
    """
    main_type = "A_same_cluster" if gt_cluster == pred_cluster else "B_cross_cluster"
    tags = [main_type]
    if query_entropy > entropy_threshold:
        tags.append("C_query_generic")
    if gt_cluster_size > crowd_threshold:
        tags.append("D_crowded_cluster")
    return main_type, tags


# ============================================================
# 逐条失败case分析
# ============================================================

def analyze_failure_details(ranks, scores, all_h, all_q_tilde, all_s_T, all_mu,
                            prototypes, queries, query_vid_map, video_ids,
                            cluster_ids, cluster_sizes,
                            entropy_threshold, crowd_threshold,
                            max_details=5000):
    """
    对 rank>1 的失败case进行详细分析。
    每条保存逐原型top-10关键原型分解。
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    w = F.normalize(all_s_T, p=1, dim=-1)

    q_entropy = compute_activation_entropy(all_s_T).numpy()

    # 找出所有失败case
    failure_indices = np.where(ranks > 1)[0]
    if len(failure_indices) > max_details:
        rng = np.random.RandomState(42)
        failure_indices = rng.choice(failure_indices, max_details, replace=False)
        failure_indices.sort()

    # 失效模式计数（对全部失败case统计，不只是采样的）
    all_failure_indices = np.where(ranks > 1)[0]
    mode_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    for q_idx in all_failure_indices:
        gt_vid = query_vid_map[q_idx]
        gt_v_idx = vid_to_idx[gt_vid]
        pred_v_idx = int(torch.argmax(scores[q_idx]).item())
        gt_cluster = int(cluster_ids[gt_v_idx])
        pred_cluster = int(cluster_ids[pred_v_idx])
        gt_cs = cluster_sizes.get(gt_cluster, 0)

        main_type, tags = classify_failure_mode(
            gt_cluster, pred_cluster, gt_cs,
            float(q_entropy[q_idx]), entropy_threshold, crowd_threshold)
        mode_counts[main_type] += 1
        for t in tags:
            tag_counts[t] += 1

    total_failures = len(all_failure_indices)
    mode_distribution = {
        k: {"count": v, "ratio": round(v / max(total_failures, 1) * 100, 2)}
        for k, v in sorted(mode_counts.items())
    }
    tag_distribution = {
        k: {"count": v, "ratio": round(v / max(total_failures, 1) * 100, 2)}
        for k, v in sorted(tag_counts.items())
    }

    # 详细分析采样的失败case
    details = []
    for q_idx in tqdm(failure_indices, desc="Analyzing failures"):
        q_idx = int(q_idx)
        gt_vid = query_vid_map[q_idx]
        gt_v_idx = vid_to_idx[gt_vid]
        pred_v_idx = int(torch.argmax(scores[q_idx]).item())
        pred_vid = video_ids[pred_v_idx]

        gt_cluster = int(cluster_ids[gt_v_idx])
        pred_cluster = int(cluster_ids[pred_v_idx])
        gt_cs = cluster_sizes.get(gt_cluster, 0)
        pred_cs = cluster_sizes.get(pred_cluster, 0)

        rank = int(ranks[q_idx])
        gt_score = float(scores[q_idx, gt_v_idx])
        pred_score = float(scores[q_idx, pred_v_idx])

        # 失效模式
        main_type, tags = classify_failure_mode(
            gt_cluster, pred_cluster, gt_cs,
            float(q_entropy[q_idx]), entropy_threshold, crowd_threshold)

        # 逐原型分解
        contrib_gt, contrib_pred = compute_per_proto_contributions(
            all_h[gt_v_idx], all_h[pred_v_idx], all_q_tilde[q_idx], w[q_idx])
        top_k_protos = get_top_k_diff_protos(contrib_gt, contrib_pred, k=10)

        # 与原型中心的对齐度
        gt_align_assigned, gt_align_mean = compute_proto_alignment(
            all_h[gt_v_idx], prototypes, gt_cluster)
        pred_align_assigned, pred_align_mean = compute_proto_alignment(
            all_h[pred_v_idx], prototypes, pred_cluster)
        q_align_gt_cluster, q_align_mean = compute_proto_alignment(
            all_q_tilde[q_idx], prototypes, gt_cluster)

        details.append({
            "query_idx": q_idx,
            "query_text": queries[q_idx][:150],
            "rank": rank,
            "gt_video": gt_vid,
            "pred_video": pred_vid,
            "gt_score": round(gt_score, 4),
            "pred_score": round(pred_score, 4),
            "score_gap": round(pred_score - gt_score, 4),
            "gt_cluster": gt_cluster,
            "pred_cluster": pred_cluster,
            "same_cluster": gt_cluster == pred_cluster,
            "gt_cluster_size": gt_cs,
            "pred_cluster_size": pred_cs,
            "query_entropy": round(float(q_entropy[q_idx]), 4),
            "failure_mode": main_type,
            "failure_tags": tags,
            "gt_proto_alignment": round(gt_align_assigned, 4),
            "pred_proto_alignment": round(pred_align_assigned, 4),
            "query_proto_alignment_gt_cluster": round(q_align_gt_cluster, 4),
            "top_10_diff_protos": top_k_protos,
        })

    return details, mode_distribution, tag_distribution


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="方式C 匹配失效分析")
    parser.add_argument("--config", type=str, default="configs/default_hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "test_1ka"])
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_queries_per_video", type=int, default=None)
    parser.add_argument("--max_failure_details", type=int, default=5000,
                        help="最多保存多少条失败case详情")
    parser.add_argument("--entropy_threshold", type=float, default=None,
                        help="query泛化判定阈值（默认自动取 75th percentile）")
    parser.add_argument("--crowd_threshold", type=int, default=None,
                        help="聚类拥挤判定阈值（默认自动取 75th percentile）")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # ---- 日志 ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"failure_analysis_{timestamp}.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # ---- 配置 ----
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}, Device: {device}")

    # ---- 数据加载 ----
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, test_ids = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"])

    if args.split == "test_1ka":
        video_ids = [f"video{i}" for i in range(7010, 8010)]
    else:
        video_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]

    if args.max_videos and len(video_ids) > args.max_videos:
        random.seed(42)
        video_ids = random.sample(video_ids, args.max_videos)
        video_ids.sort(key=lambda x: int(x.replace("video", "")))

    queries, query_vid_map = [], []
    for vid in video_ids:
        if vid in gt:
            caps = gt[vid][:args.max_queries_per_video] if args.max_queries_per_video else gt[vid]
            for cap in caps:
                queries.append(cap)
                query_vid_map.append(vid)

    logger.info(f"Videos: {len(video_ids)}, Queries: {len(queries)}")

    # ---- 模型加载 ----
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    temperature = model.temperature.item()
    logger.info(f"Temperature τ = {temperature:.6f}")

    # 获取原型中心
    with torch.no_grad():
        prototypes = model.prototype_lib(use_ema=False).cpu()  # (K, d)
    logger.info(f"Prototypes: {prototypes.shape}")

    # ---- 编码 ----
    logger.info("=" * 60)
    logger.info("编码视频和查询")
    logger.info("=" * 60)

    all_h, all_mu = encode_all_videos(model, encoder, video_ids, narrations, device)
    all_s_T, all_q_tilde = encode_all_queries(model, encoder, queries, device)

    logger.info(f"  Video h: {all_h.shape}, Video μ: {all_mu.shape}")
    logger.info(f"  Query s_T: {all_s_T.shape}, Query q_tilde: {all_q_tilde.shape}")

    # ---- 评分 ----
    logger.info("=" * 60)
    logger.info("方式C 检索评分")
    logger.info("=" * 60)

    scores = compute_scores_method_c(all_h, all_q_tilde, all_s_T, temperature)
    metrics, ranks = compute_retrieval_metrics(scores, query_vid_map, video_ids)
    logger.info(f"  R@1={metrics['R@1']}, R@5={metrics['R@5']}, R@10={metrics['R@10']}, "
                f"MdR={metrics['MdR']}, MnR={metrics['MnR']}")

    # ---- 聚类信息 ----
    cluster_ids, cluster_sizes = build_cluster_info(all_mu)
    N_v = len(video_ids)
    K = all_mu.shape[1]
    n_nonempty = len(cluster_sizes)
    logger.info(f"  聚类: {n_nonempty}/{K} 非空, 最大={max(cluster_sizes.values())}, "
                f"平均={N_v/n_nonempty:.1f}")

    # ---- 自动确定阈值 ----
    q_entropy = compute_activation_entropy(all_s_T).numpy()
    sizes_array = np.array([cluster_sizes.get(int(cluster_ids[
        {v: i for i, v in enumerate(video_ids)}[query_vid_map[q]]
    ]), 0) for q in range(len(queries))])

    entropy_threshold = args.entropy_threshold or float(np.percentile(q_entropy, 75))
    crowd_threshold = args.crowd_threshold or int(np.percentile(
        list(cluster_sizes.values()), 75))

    logger.info(f"  Entropy threshold: {entropy_threshold:.4f} (75th pctl)")
    logger.info(f"  Crowd threshold: {crowd_threshold}")

    # ---- Part 1: 按rank分桶统计 ----
    logger.info("=" * 60)
    logger.info("Part 1: 按rank分桶统计")
    logger.info("=" * 60)

    bucket_stats = bucket_statistics(
        ranks, scores, query_vid_map, video_ids,
        all_mu, all_s_T, cluster_ids, cluster_sizes)

    header = (f"  {'桶':<16} {'数量':>6} {'占比%':>6} {'同聚类%':>8} "
              f"{'GT分':>8} {'Pred分':>8} {'Gap':>8} {'聚类大小':>8} {'Entropy':>8}")
    logger.info(header)
    logger.info(f"  {'─' * 90}")
    for b in bucket_stats:
        if b["count"] == 0:
            logger.info(f"  {b['bucket']:<16} {0:>6} {0:>6.1f}")
            continue
        logger.info(
            f"  {b['bucket']:<16} {b['count']:>6} {b['ratio']:>6.1f} "
            f"{b['same_cluster_ratio']:>8.1f} "
            f"{b['gt_score_mean']:>8.3f} {b['pred_score_mean']:>8.3f} "
            f"{b['score_gap_mean']:>8.3f} "
            f"{b['gt_cluster_size_mean']:>8.1f} {b['query_entropy_mean']:>8.3f}")

    # ---- Part 2: 失效模式分析 ----
    logger.info("=" * 60)
    logger.info("Part 2: 失效模式分析")
    logger.info("=" * 60)

    details, mode_dist, tag_dist = analyze_failure_details(
        ranks, scores, all_h, all_q_tilde, all_s_T, all_mu,
        prototypes, queries, query_vid_map, video_ids,
        cluster_ids, cluster_sizes,
        entropy_threshold, crowd_threshold,
        max_details=args.max_failure_details)

    total_failures = int((ranks > 1).sum())
    logger.info(f"  总query数: {len(queries)}, 失败数: {total_failures} "
                f"({total_failures/len(queries)*100:.1f}%)")
    logger.info(f"  详细分析case数: {len(details)}")
    logger.info(f"\n  主类型分布:")
    for mode, info in mode_dist.items():
        logger.info(f"    {mode}: {info['count']} ({info['ratio']}%)")
    logger.info(f"\n  标签分布 (可重叠):")
    for tag, info in tag_dist.items():
        logger.info(f"    {tag}: {info['count']} ({info['ratio']}%)")

    # ---- Part 3: 失败case样例展示 ----
    logger.info("=" * 60)
    logger.info("Part 3: 失败case样例 (按rank排序, 前15条)")
    logger.info("=" * 60)

    details_sorted = sorted(details, key=lambda x: x["rank"])
    for d in details_sorted[:15]:
        logger.info(f"  [{d['failure_mode']}] rank={d['rank']} | "
                     f"\"{d['query_text'][:80]}\"")
        logger.info(f"    GT={d['gt_video']}(cluster={d['gt_cluster']}, size={d['gt_cluster_size']}) | "
                     f"Pred={d['pred_video']}(cluster={d['pred_cluster']}, size={d['pred_cluster_size']})")
        logger.info(f"    gt_score={d['gt_score']:.4f}, pred_score={d['pred_score']:.4f}, "
                     f"gap={d['score_gap']:.4f}, entropy={d['query_entropy']:.4f}")
        top3 = d["top_10_diff_protos"][:3]
        proto_str = ", ".join(
            f"p{p['proto_id']}(Δ={p['diff']:+.4f})" for p in top3)
        logger.info(f"    关键原型: {proto_str}")

    # ---- 保存 ----
    result = {
        "config": {
            "split": args.split,
            "num_videos": N_v,
            "num_queries": len(queries),
            "temperature": temperature,
            "entropy_threshold": entropy_threshold,
            "crowd_threshold": crowd_threshold,
        },
        "metrics": metrics,
        "bucket_statistics": bucket_stats,
        "failure_mode_distribution": mode_dist,
        "failure_tag_distribution": tag_dist,
        "failure_details": details,
    }
    json_path = log_dir / f"failure_analysis_{args.split}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {json_path}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == "__main__":
    main()
