"""
诊断脚本：验证训练是否有效 — Query(GT caption)能否通过方式C匹配到其对应伪查询(narrations)。

使用与训练完全相同的评分方式（方式C: per-prototype weighted contrastive scoring），
检查每个 query 能否正确匹配到其所属视频的 narrations。

用法:
    python scripts/diagnose_matching.py [--config configs/default_hybrid.yaml] [--checkpoint checkpoints/best_model.pt]
    python scripts/diagnose_matching.py --split val
    python scripts/diagnose_matching.py --split train --max_videos 200
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids
from models.clip_encoder import CLIPTextEncoder
from train import encode_video_captions, build_model

logger = logging.getLogger("diagnose")


# ============================================================
# 编码
# ============================================================

def encode_all_videos(model, encoder, video_ids, narrations, device, batch_size=32):
    """编码所有视频 → h (N, K, d) + μ (N, K)。"""
    all_h, all_mu = [], []
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


def encode_all_queries(model, encoder, queries, device, batch_size=256,
                       return_raw=False):
    """编码所有查询 → s_T (N, K) + q_tilde (N, K, d) [+ s_T_raw (N, K)]。"""
    all_s_T, all_q_tilde = [], []
    all_s_T_raw = [] if return_raw else None
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_queries = queries[i:i + batch_size]
            query_feats, query_mask = encoder.encode_tokens(batch_queries, device=device)
            result = model.encode_query(query_feats, query_mask, use_ema=False,
                                        return_raw=return_raw)
            if return_raw:
                s_T, q_tilde, s_T_raw = result
                all_s_T_raw.append(s_T_raw.cpu())
            else:
                s_T, q_tilde = result
            all_s_T.append(s_T.cpu())
            all_q_tilde.append(q_tilde.cpu())
    if return_raw:
        return (torch.cat(all_s_T, dim=0), torch.cat(all_q_tilde, dim=0),
                torch.cat(all_s_T_raw, dim=0))
    return torch.cat(all_s_T, dim=0), torch.cat(all_q_tilde, dim=0)


# ============================================================
# 评分 — 方式C (per-prototype)
# ============================================================

def compute_scores_method_c(all_h, all_q_tilde, all_s_T, temperature, query_batch_size=64):
    """
    方式C: 训练时的逐原型细粒度对比评分。
    scores[q, v] = Σ_k w_qk * cos(h_v^k, q_tilde_q^k) / τ
    """
    N_v = all_h.shape[0]
    N_q = all_q_tilde.shape[0]

    h_norm = F.normalize(all_h, p=2, dim=-1)
    q_norm = F.normalize(all_q_tilde, p=2, dim=-1)
    w = F.normalize(all_s_T, p=1, dim=-1)

    scores = torch.zeros(N_q, N_v)
    for qi in tqdm(range(0, N_q, query_batch_size), desc="Scoring (method C)"):
        qe = min(qi + query_batch_size, N_q)
        q_batch = q_norm[qi:qe]
        w_batch = w[qi:qe]
        sim_per_proto = torch.einsum("qkd, vkd -> qvk", q_batch, h_norm)
        scores[qi:qe] = torch.einsum("qvk, qk -> qv", sim_per_proto, w_batch) / temperature

    return scores


# ============================================================
# 指标计算
# ============================================================

def compute_retrieval_metrics(scores, query_video_map, video_ids):
    """计算 R@1, R@5, R@10, MdR, MnR。"""
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
# 原型利用率分析
# ============================================================

def analyze_prototype_utilization(all_mu, all_s_T):
    """分析原型利用率。"""
    N_v, K = all_mu.shape

    v_cluster = torch.argmax(all_mu, dim=1).numpy()
    v_counts = np.bincount(v_cluster, minlength=K)
    v_nonempty = int((v_counts > 0).sum())

    q_cluster = torch.argmax(all_s_T, dim=1).numpy()
    q_counts = np.bincount(q_cluster, minlength=K)
    q_nonempty = int((q_counts > 0).sum())

    v_active = set(np.where(v_counts > 0)[0])
    q_active = set(np.where(q_counts > 0)[0])

    return {
        "K": int(K),
        "video_nonempty_clusters": v_nonempty,
        "query_nonempty_clusters": q_nonempty,
        "shared_active_clusters": len(v_active & q_active),
        "video_cluster_max": int(v_counts.max()),
        "video_cluster_mean": round(float(v_counts[v_counts > 0].mean()), 2),
        "query_cluster_max": int(q_counts.max()),
        "query_cluster_mean": round(float(q_counts[q_counts > 0].mean()), 2),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="诊断训练有效性：方式C 伪查询匹配")
    parser.add_argument("--config", type=str, default="configs/default_hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test", "test_1ka"],
                        help="数据集 (test_1ka = MSR-VTT 1K-A: video7010~video8009)")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_queries_per_video", type=int, default=None)
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
    logger.info(f"Model loaded. Temperature τ = {temperature:.6f}")

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
    logger.info(f"  Score: mean={scores.mean():.4f}, std={scores.std():.4f}, "
                f"min={scores.min():.4f}, max={scores.max():.4f}")

    # ---- 检索指标 ----
    logger.info("=" * 60)
    logger.info("检索指标")
    logger.info("=" * 60)

    metrics, ranks = compute_retrieval_metrics(scores, query_vid_map, video_ids)
    N_v = len(video_ids)
    logger.info(f"  {'指标':<8} {'方式C':>12} {'随机基线':>12}")
    logger.info(f"  {'─' * 36}")
    for k in ["R@1", "R@5", "R@10"]:
        rk = round(int(k.split("@")[1]) * 100.0 / N_v, 2)
        logger.info(f"  {k:<8} {metrics[k]:>12.2f} {rk:>12.2f}")
    logger.info(f"  {'MdR':<8} {metrics['MdR']:>12.1f} {N_v/2:>12.1f}")
    logger.info(f"  {'MnR':<8} {metrics['MnR']:>12.2f} {N_v/2:>12.1f}")

    # ---- 原型利用率 ----
    logger.info("=" * 60)
    logger.info("原型利用率")
    logger.info("=" * 60)

    proto_util = analyze_prototype_utilization(all_mu, all_s_T)
    logger.info(f"  K = {proto_util['K']}")
    logger.info(f"  视频非空聚类: {proto_util['video_nonempty_clusters']}/{proto_util['K']}")
    logger.info(f"  查询非空聚类: {proto_util['query_nonempty_clusters']}/{proto_util['K']}")
    logger.info(f"  共享活跃聚类: {proto_util['shared_active_clusters']}/{proto_util['K']}")

    # ---- 保存 ----
    result = {
        "config": {"split": args.split, "num_videos": N_v,
                    "num_queries": len(queries), "temperature": temperature},
        "metrics": metrics,
        "prototype_utilization": proto_util,
    }
    json_path = log_dir / f"diagnose_{args.split}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {json_path}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == "__main__":
    main()
