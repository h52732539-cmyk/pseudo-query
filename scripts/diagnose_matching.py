"""
诊断脚本：验证训练是否有效 — Query(GT caption)能否通过方式C匹配到其对应伪查询(narrations)。

核心思路：
  训练时，模型学习的是「query(GT caption) ↔ narrations(伪查询)」的逐原型细粒度匹配。
  本脚本使用与训练完全相同的评分方式（方式C: per-prototype weighted contrastive scoring），
  检查每个 query 能否正确匹配到其所属视频的 narrations，从而确认训练是否确实学到了有效匹配。

  若方式C检索结果显著优于随机，说明训练有效；否则说明训练本身有问题。

用法:
    python scripts/diagnose_matching.py [--config configs/default_hybrid.yaml] [--checkpoint checkpoints/best_model.pt]
    python scripts/diagnose_matching.py --split val        # 在验证集上诊断
    python scripts/diagnose_matching.py --split train --max_videos 200  # 训练集子采样
"""
import argparse
import json
import logging
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from models.clip_encoder import CLIPTextEncoder
from train import encode_video_captions, build_model

logger = logging.getLogger("diagnose")


# ============================================================
# Phase 1: 编码 — 保留完整 h 和 q_tilde 表示
# ============================================================

def encode_all_videos(model, encoder, video_ids, narrations, device, batch_size=32):
    """
    编码所有视频 → h (N, K, d) + μ (N, K)。
    使用梯度原型（与推理时一致）。
    """
    all_h = []
    all_mu = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(video_ids), batch_size), desc="Encoding videos"):
            batch_vids = video_ids[i:i + batch_size]
            captions_list = [narrations[vid] for vid in batch_vids]
            video_feats, video_mask = encode_video_captions(encoder, captions_list, device)
            h, mu = model.encode_video(video_feats, video_mask, use_ema=False)
            all_h.append(h.cpu())
            all_mu.append(mu.cpu())
    return torch.cat(all_h, dim=0), torch.cat(all_mu, dim=0)


def encode_all_queries(model, encoder, queries, device, batch_size=256):
    """
    编码所有查询 → s_T (N, K) + q_tilde (N, K, d)。
    使用梯度原型（与推理时一致）。
    """
    all_s_T = []
    all_q_tilde = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_queries = queries[i:i + batch_size]
            query_feats, query_mask = encoder.encode_tokens(batch_queries, device=device)
            s_T, q_tilde = model.encode_query(query_feats, query_mask, use_ema=False)
            all_s_T.append(s_T.cpu())
            all_q_tilde.append(q_tilde.cpu())
    return torch.cat(all_s_T, dim=0), torch.cat(all_q_tilde, dim=0)


# ============================================================
# Phase 2: 评分 — 方式C (per-prototype) vs 方式A (cosine μ·s_T)
# ============================================================

def compute_scores_method_c(all_h, all_q_tilde, all_s_T, temperature, query_batch_size=64):
    """
    方式C: 训练时的逐原型细粒度对比评分。
    scores[q, v] = Σ_k w_qk * cos(h_v^k, q_tilde_q^k) / τ

    分批计算避免 OOM。
    Returns: scores (N_q, N_v)
    """
    N_v = all_h.shape[0]
    N_q = all_q_tilde.shape[0]

    # 预处理：L2 归一化
    h_norm = F.normalize(all_h, p=2, dim=-1)      # (N_v, K, d)
    q_norm = F.normalize(all_q_tilde, p=2, dim=-1)  # (N_q, K, d)
    w = F.normalize(all_s_T, p=1, dim=-1)          # (N_q, K) L1 归一化

    scores = torch.zeros(N_q, N_v)
    for qi in tqdm(range(0, N_q, query_batch_size), desc="Scoring (method C)"):
        qe = min(qi + query_batch_size, N_q)
        q_batch = q_norm[qi:qe]   # (B_q, K, d)
        w_batch = w[qi:qe]        # (B_q, K)

        # 逐原型余弦相似度: (B_q, N_v, K)
        # sim[q, v, k] = cos(h_v^k, q_tilde_q^k)
        sim_per_proto = torch.einsum("qkd, vkd -> qvk", q_batch, h_norm)

        # 加权求和: (B_q, N_v)
        scores_batch = torch.einsum("qvk, qk -> qv", sim_per_proto, w_batch) / temperature
        scores[qi:qe] = scores_batch

    return scores


def compute_scores_method_a(all_s_T, all_mu, temperature):
    """
    方式A: 推理时的简单余弦评分。
    scores[q, v] = s_T_q · μ_v / τ
    """
    return torch.matmul(all_s_T, all_mu.T) / temperature


# ============================================================
# Phase 3: 指标计算
# ============================================================

def compute_retrieval_metrics(scores, query_video_map, video_ids):
    """
    计算 R@1, R@5, R@10, MdR, MnR。
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores.shape[0]

    ranks = []
    for q_idx in range(N_q):
        gt_vid = query_video_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        sorted_indices = torch.argsort(scores[q_idx], descending=True)
        rank = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": round(float((ranks <= 1).mean() * 100), 2),
        "R@5": round(float((ranks <= 5).mean() * 100), 2),
        "R@10": round(float((ranks <= 10).mean() * 100), 2),
        "MdR": float(np.median(ranks)),
        "MnR": round(float(np.mean(ranks)), 2),
    }, ranks


# ============================================================
# Phase 4: 原型分析
# ============================================================

def analyze_prototype_activation(all_s_T, all_mu, query_video_map, video_ids):
    """分析 GT pair 的原型激活分布。"""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = all_s_T.shape[0]
    K = all_s_T.shape[1]

    # 查询和视频各自的top-k原型重叠度
    overlaps_top1 = []
    overlaps_top5 = []
    overlaps_top10 = []
    cosine_sims = []

    for q_idx in range(N_q):
        gt_vid = query_video_map[q_idx]
        v_idx = vid_to_idx[gt_vid]

        q_act = all_s_T[q_idx]   # (K,)
        v_act = all_mu[v_idx]    # (K,)

        # Top-k 重叠
        q_topk1 = set(torch.topk(q_act, 1).indices.tolist())
        v_topk1 = set(torch.topk(v_act, 1).indices.tolist())
        overlaps_top1.append(len(q_topk1 & v_topk1))

        q_topk5 = set(torch.topk(q_act, 5).indices.tolist())
        v_topk5 = set(torch.topk(v_act, 5).indices.tolist())
        overlaps_top5.append(len(q_topk5 & v_topk5))

        q_topk10 = set(torch.topk(q_act, 10).indices.tolist())
        v_topk10 = set(torch.topk(v_act, 10).indices.tolist())
        overlaps_top10.append(len(q_topk10 & v_topk10))

        # 余弦相似度
        cosine_sims.append(F.cosine_similarity(q_act.unsqueeze(0), v_act.unsqueeze(0)).item())

    return {
        "top1_overlap_mean": round(float(np.mean(overlaps_top1)), 4),
        "top5_overlap_mean": round(float(np.mean(overlaps_top5)), 4),
        "top10_overlap_mean": round(float(np.mean(overlaps_top10)), 4),
        "cosine_sim_mean": round(float(np.mean(cosine_sims)), 4),
        "cosine_sim_std": round(float(np.std(cosine_sims)), 4),
    }


def analyze_prototype_utilization(all_mu, all_s_T):
    """分析原型利用率。"""
    N_v, K = all_mu.shape
    N_q = all_s_T.shape[0]

    # 视频侧：各原型被多少视频作为 argmax 归属
    v_cluster = torch.argmax(all_mu, dim=1).numpy()
    v_counts = np.bincount(v_cluster, minlength=K)
    v_nonempty = int((v_counts > 0).sum())

    # 查询侧：各原型被多少查询作为 argmax 归属
    q_cluster = torch.argmax(all_s_T, dim=1).numpy()
    q_counts = np.bincount(q_cluster, minlength=K)
    q_nonempty = int((q_counts > 0).sum())

    # 视频和查询的 argmax 原型重叠
    v_active = set(np.where(v_counts > 0)[0])
    q_active = set(np.where(q_counts > 0)[0])
    shared_active = v_active & q_active

    return {
        "K": int(K),
        "video_nonempty_clusters": v_nonempty,
        "query_nonempty_clusters": q_nonempty,
        "shared_active_clusters": len(shared_active),
        "video_cluster_max": int(v_counts.max()),
        "video_cluster_mean": round(float(v_counts[v_counts > 0].mean()), 2),
        "query_cluster_max": int(q_counts.max()),
        "query_cluster_mean": round(float(q_counts[q_counts > 0].mean()), 2),
    }


# ============================================================
# Phase 5: 详细案例分析
# ============================================================

def sample_case_studies(scores_c, scores_a, all_s_T, all_mu, queries,
                        query_video_map, video_ids, n=20):
    """采样一些具体案例进行详细对比。"""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores_c.shape[0]
    indices = random.sample(range(N_q), min(n, N_q))

    cases = []
    for q_idx in indices:
        gt_vid = query_video_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]

        rank_c = int((torch.argsort(scores_c[q_idx], descending=True) == gt_idx).nonzero(as_tuple=True)[0].item() + 1)
        rank_a = int((torch.argsort(scores_a[q_idx], descending=True) == gt_idx).nonzero(as_tuple=True)[0].item() + 1)

        # GT pair 的 score
        score_c_gt = float(scores_c[q_idx, gt_idx])
        score_a_gt = float(scores_a[q_idx, gt_idx])

        # Top-1 prediction
        top1_c = int(torch.argmax(scores_c[q_idx]).item())
        top1_a = int(torch.argmax(scores_a[q_idx]).item())

        cases.append({
            "query_idx": q_idx,
            "query_text": queries[q_idx][:100],
            "gt_video": gt_vid,
            "method_c": {"rank": rank_c, "gt_score": round(score_c_gt, 4),
                         "top1_video": video_ids[top1_c]},
            "method_a": {"rank": rank_a, "gt_score": round(score_a_gt, 4),
                         "top1_video": video_ids[top1_a]},
        })

    # 按 method_c rank 排序
    cases.sort(key=lambda x: x["method_c"]["rank"])
    return cases


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="诊断训练有效性：方式C vs 方式A 检索对比")
    parser.add_argument("--config", type=str, default="configs/default_hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "test_1ka"],
                        help="在哪个数据集上诊断 (test_1ka = MSR-VTT 1K-A: video7010~video8009)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="限制视频数量（用于训练集子采样）")
    parser.add_argument("--max_queries_per_video", type=int, default=None,
                        help="每个视频最多取多少条 GT query")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # ---- 日志 ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"diagnose_{timestamp}.log"

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

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    if args.split == "test_1ka":
        video_ids = [f"video{i}" for i in range(7010, 8010)]
    else:
        video_ids = split_map[args.split]

    if args.max_videos and len(video_ids) > args.max_videos:
        random.seed(42)
        video_ids = random.sample(video_ids, args.max_videos)
        video_ids.sort(key=lambda x: int(x.replace("video", "")))

    # 构建 query 列表
    queries = []
    query_vid_map = []
    for vid in video_ids:
        if vid in gt:
            caps = gt[vid]
            if args.max_queries_per_video:
                caps = caps[:args.max_queries_per_video]
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
    logger.info(f"Model loaded. Temperature τ = {temperature:.6f}")

    # ---- Phase 2: 编码 ----
    logger.info("=" * 60)
    logger.info("Phase 2: 编码视频和查询")
    logger.info("=" * 60)

    all_h, all_mu = encode_all_videos(model, encoder, video_ids, narrations, device)
    all_s_T, all_q_tilde = encode_all_queries(model, encoder, queries, device)

    logger.info(f"  Video h:       {all_h.shape}  (N_v, K, d)")
    logger.info(f"  Video μ:       {all_mu.shape}  (N_v, K)")
    logger.info(f"  Query s_T:     {all_s_T.shape}  (N_q, K)")
    logger.info(f"  Query q_tilde: {all_q_tilde.shape}  (N_q, K, d)")

    # ---- Phase 3: 评分 ----
    logger.info("=" * 60)
    logger.info("Phase 3: 计算检索分数")
    logger.info("=" * 60)

    scores_c = compute_scores_method_c(all_h, all_q_tilde, all_s_T, temperature)
    scores_a = compute_scores_method_a(all_s_T, all_mu, temperature)

    logger.info(f"  Score matrix shape: {scores_c.shape} (N_q × N_v)")
    logger.info(f"  方式C score 统计: mean={scores_c.mean():.4f}, std={scores_c.std():.4f}, "
                f"min={scores_c.min():.4f}, max={scores_c.max():.4f}")
    logger.info(f"  方式A score 统计: mean={scores_a.mean():.4f}, std={scores_a.std():.4f}, "
                f"min={scores_a.min():.4f}, max={scores_a.max():.4f}")

    # ---- Phase 3.1: 检索指标 ----
    logger.info("=" * 60)
    logger.info("Phase 3: 检索指标对比")
    logger.info("=" * 60)

    metrics_c, ranks_c = compute_retrieval_metrics(scores_c, query_vid_map, video_ids)
    metrics_a, ranks_a = compute_retrieval_metrics(scores_a, query_vid_map, video_ids)

    logger.info(f"  {'指标':<8} {'方式C (训练匹配)':>18} {'方式A (推理匹配)':>18} {'随机基线':>12}")
    logger.info(f"  {'─' * 60}")
    random_r1 = round(100.0 / len(video_ids), 2)
    random_mnr = round(len(video_ids) / 2, 1)
    for k in ["R@1", "R@5", "R@10"]:
        random_rk = round(int(k.split("@")[1]) * 100.0 / len(video_ids), 2)
        logger.info(f"  {k:<8} {metrics_c[k]:>18.2f} {metrics_a[k]:>18.2f} {random_rk:>12.2f}")
    logger.info(f"  {'MdR':<8} {metrics_c['MdR']:>18.1f} {metrics_a['MdR']:>18.1f} {len(video_ids)/2:>12.1f}")
    logger.info(f"  {'MnR':<8} {metrics_c['MnR']:>18.2f} {metrics_a['MnR']:>18.2f} {random_mnr:>12.1f}")

    # ---- Phase 4: 原型分析 ----
    logger.info("=" * 60)
    logger.info("Phase 4: 原型激活分析")
    logger.info("=" * 60)

    proto_overlap = analyze_prototype_activation(all_s_T, all_mu, query_vid_map, video_ids)
    logger.info(f"  GT pair 原型激活重叠度:")
    logger.info(f"    Top-1 overlap:  {proto_overlap['top1_overlap_mean']:.4f} (理想=1.0)")
    logger.info(f"    Top-5 overlap:  {proto_overlap['top5_overlap_mean']:.4f} (理想=5.0)")
    logger.info(f"    Top-10 overlap: {proto_overlap['top10_overlap_mean']:.4f} (理想=10.0)")
    logger.info(f"    s_T·μ 余弦相似度: {proto_overlap['cosine_sim_mean']:.4f} ± {proto_overlap['cosine_sim_std']:.4f}")

    proto_util = analyze_prototype_utilization(all_mu, all_s_T)
    logger.info(f"  原型利用率:")
    logger.info(f"    K = {proto_util['K']}")
    logger.info(f"    视频非空聚类: {proto_util['video_nonempty_clusters']}/{proto_util['K']}")
    logger.info(f"    查询非空聚类: {proto_util['query_nonempty_clusters']}/{proto_util['K']}")
    logger.info(f"    共享活跃聚类: {proto_util['shared_active_clusters']}/{proto_util['K']}")
    logger.info(f"    视频聚类大小: max={proto_util['video_cluster_max']}, mean={proto_util['video_cluster_mean']}")
    logger.info(f"    查询聚类大小: max={proto_util['query_cluster_max']}, mean={proto_util['query_cluster_mean']}")

    # ---- Phase 5: 案例分析 ----
    logger.info("=" * 60)
    logger.info("Phase 5: 案例分析 (随机采样 20 条)")
    logger.info("=" * 60)

    cases = sample_case_studies(scores_c, scores_a, all_s_T, all_mu,
                                queries, query_vid_map, video_ids, n=20)
    for c in cases:
        mc = c["method_c"]
        ma = c["method_a"]
        logger.info(f"  Query: \"{c['query_text']}\"")
        logger.info(f"    GT={c['gt_video']} | C: rank={mc['rank']}, top1={mc['top1_video']} | "
                     f"A: rank={ma['rank']}, top1={ma['top1_video']}")

    # ---- 保存结果 ----
    result = {
        "config": {
            "split": args.split,
            "num_videos": len(video_ids),
            "num_queries": len(queries),
            "temperature": temperature,
            "checkpoint": args.checkpoint,
        },
        "metrics_method_c": metrics_c,
        "metrics_method_a": metrics_a,
        "prototype_overlap": proto_overlap,
        "prototype_utilization": proto_util,
        "case_studies": cases,
    }

    json_path = log_dir / f"diagnose_{args.split}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {json_path}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == "__main__":
    main()
