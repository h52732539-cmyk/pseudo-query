"""
评估脚本：Text→Video Retrieval — R@1, R@5, R@10, MdR, MnR。
支持方案B (SwAV) 和方案C (Hybrid)。

用法:
    python evaluate.py [--config configs/default.yaml] [--checkpoint checkpoints/best_model.pt]
"""
import argparse
import json
import logging
from datetime import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids
from models.clip_encoder import CLIPTextEncoder
from models.scoring import cosine_retrieval_score
from train import encode_video_captions, build_model

logger = logging.getLogger("eval")


def build_video_representations(model, encoder, video_ids, narrations, device, batch_size=32):
    """
    离线编码所有视频 → μ ∈ R^K 表示。
    Returns:
        all_mu: (N_videos, K)
    """
    all_mu = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(video_ids), batch_size), desc="Encoding videos"):
            batch_vids = video_ids[i : i + batch_size]
            captions_list = [narrations[vid] for vid in batch_vids]
            video_feats, video_mask = encode_video_captions(encoder, captions_list, device)
            mu = model.get_video_repr(video_feats, video_mask)
            all_mu.append(mu.cpu())

    return torch.cat(all_mu, dim=0)


def build_query_representations(model, encoder, queries, device, batch_size=256):
    """
    离线编码所有查询 → s_T ∈ R^K 表示。
    Returns:
        all_s_T: (N_queries, K)
    """
    all_s_T = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_queries = queries[i : i + batch_size]
            query_feats, query_mask = encoder.encode_tokens(batch_queries, device=device)
            s_T = model.get_query_repr(query_feats, query_mask)
            all_s_T.append(s_T.cpu())

    return torch.cat(all_s_T, dim=0)


def compute_metrics(scores, query_video_map, video_ids):
    """
    计算 R@1, R@5, R@10, MdR, MnR。
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores.shape[0]

    ranks = []
    for q_idx in range(N_q):
        gt_vid = query_video_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        score_row = scores[q_idx]
        sorted_indices = torch.argsort(score_row, descending=True)
        rank = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    r1 = (ranks <= 1).mean() * 100
    r5 = (ranks <= 5).mean() * 100
    r10 = (ranks <= 10).mean() * 100
    mdr = np.median(ranks)
    mnr = np.mean(ranks)

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "MdR": mdr,
        "MnR": mnr,
    }, ranks


def print_prototype_statistics(all_mu, video_ids):
    """打印测试集视频的原型激活统计信息。"""
    N, K = all_mu.shape
    cluster_assignment = torch.argmax(all_mu, dim=1).numpy()

    cluster_counts = defaultdict(int)
    for c in cluster_assignment:
        cluster_counts[int(c)] += 1

    non_empty = len(cluster_counts)
    empty = K - non_empty
    counts = np.array(list(cluster_counts.values()))

    logger.info("=" * 60)
    logger.info("原型聚类统计信息")
    logger.info("=" * 60)
    logger.info(f"  原型总数 K = {K}")
    logger.info(f"  视频总数 N = {N}")
    logger.info(f"  非空聚类数 = {non_empty}")
    logger.info(f"  空聚类数   = {empty}")
    logger.info(f"  各聚类视频数分布: min={counts.min()}, max={counts.max()}, "
                f"mean={counts.mean():.2f}, median={np.median(counts):.1f}, std={counts.std():.2f}")

    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"  Top-10 最大聚类: {[(c, n) for c, n in sorted_clusters[:10]]}")
    logger.info(f"  Top-10 最小非空聚类: {[(c, n) for c, n in sorted_clusters[-10:]]}")

    mu_max_vals = all_mu.max(dim=1).values
    logger.info(f"  归属聚类维度的 μ 统计: mean={mu_max_vals.mean():.4f}, std={mu_max_vals.std():.4f}")
    logger.info(f"  全局 μ 统计: mean={all_mu.mean():.4f}, std={all_mu.std():.4f}, "
                f"min={all_mu.min():.4f}, max={all_mu.max():.4f}")
    logger.info("=" * 60)

    return cluster_assignment


def analyze_results(scores, all_mu, queries, query_vid_map,
                    video_ids, ranks, cfg, log_dir, tag="full"):
    """对评估结果进行详细分析，保存为 JSON。"""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_videos, K = all_mu.shape
    cluster_assignment = torch.argmax(all_mu, dim=1).numpy()

    details = []
    same_cluster_count = 0
    rank1_same_cluster_count = 0
    ranks_same = []
    ranks_diff = []

    for q_idx in range(len(queries)):
        gt_vid = query_vid_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]

        pred_idx = torch.argsort(scores[q_idx], descending=True)[0].item()
        pred_vid = video_ids[pred_idx]

        gt_cluster = int(cluster_assignment[gt_idx])
        pred_cluster = int(cluster_assignment[pred_idx])
        same_cluster = (gt_cluster == pred_cluster)

        rank = int(ranks[q_idx])

        if same_cluster:
            same_cluster_count += 1
            ranks_same.append(rank)
            if rank == 1:
                rank1_same_cluster_count += 1
        else:
            ranks_diff.append(rank)

        details.append({
            "query_idx": q_idx,
            "query_text": queries[q_idx],
            "gt_video_id": gt_vid,
            "predicted_video_id": pred_vid,
            "rank": rank,
            "gt_cluster": gt_cluster,
            "pred_cluster": pred_cluster,
            "same_cluster": same_cluster,
        })

    total = len(queries)
    summary = {
        "total_queries": total,
        "same_cluster_ratio": round(same_cluster_count / max(total, 1), 4),
        "same_cluster_count": same_cluster_count,
        "diff_cluster_count": total - same_cluster_count,
        "rank1_same_cluster_count": rank1_same_cluster_count,
        "rank1_same_cluster_ratio": round(rank1_same_cluster_count / max(same_cluster_count, 1), 4),
        "avg_rank_same_cluster": round(float(np.mean(ranks_same)), 4) if ranks_same else None,
        "avg_rank_diff_cluster": round(float(np.mean(ranks_diff)), 4) if ranks_diff else None,
    }

    result = {
        "config": {
            "num_prototypes": K,
            "test_mode": tag,
            "num_videos": N_videos,
            "scheme": cfg.get("scheme", "swav"),
        },
        "summary": summary,
        "details": details,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"eval_analysis_{tag}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"分析结果已保存到: {json_path}")
    logger.info(f"分析摘要: {json.dumps(summary, ensure_ascii=False)}")


def evaluate_full_test(model, encoder, test_ids, gt, narrations, device, cfg, log_dir):
    """全量测试集评估（每个GT caption作为独立查询）"""
    logger.info(f"\n=== Full Test Evaluation ({len(test_ids)} videos) ===")

    all_mu = build_video_representations(model, encoder, test_ids, narrations, device)
    print_prototype_statistics(all_mu, test_ids)

    queries = []
    query_vid_map = []
    for vid in test_ids:
        if vid in gt:
            for cap in gt[vid]:
                queries.append(cap)
                query_vid_map.append(vid)

    logger.info(f"Total queries: {len(queries)}")
    all_s_T = build_query_representations(model, encoder, queries, device)

    temperature = model.temperature.item()
    scores = cosine_retrieval_score(all_s_T, all_mu, temperature=temperature)

    metrics, ranks = compute_metrics(scores, query_vid_map, test_ids)

    analyze_results(scores, all_mu, queries, query_vid_map,
                    test_ids, ranks, cfg, log_dir, tag="full")

    return metrics


def evaluate_1k_a(model, encoder, test_ids, gt, narrations, device, cfg, log_dir):
    """MSR-VTT 1K-A 评估：video7010~video8009，每个视频取第一条 GT caption。"""
    test_1k_ids = [f"video{i}" for i in range(7010, 8010)]
    logger.info(f"\n=== 1K-A Test Evaluation ({len(test_1k_ids)} videos) ===")

    all_mu = build_video_representations(model, encoder, test_1k_ids, narrations, device)
    print_prototype_statistics(all_mu, test_1k_ids)

    queries = []
    query_vid_map = []
    for vid in test_1k_ids:
        if vid in gt and len(gt[vid]) > 0:
            queries.append(gt[vid][0])
            query_vid_map.append(vid)

    logger.info(f"Total queries: {len(queries)}")
    all_s_T = build_query_representations(model, encoder, queries, device)

    temperature = model.temperature.item()
    scores = cosine_retrieval_score(all_s_T, all_mu, temperature=temperature)

    metrics, ranks = compute_metrics(scores, query_vid_map, test_1k_ids)

    analyze_results(scores, all_mu, queries, query_vid_map,
                    test_1k_ids, ranks, cfg, log_dir, tag="1k-A")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test_mode", type=str, default=None, help="full | 1k-A | both")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 初始化日志 ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    scheme = cfg.get("scheme", "swav")
    logger.info(f"Config: {args.config}, Scheme: {scheme}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Log file: {log_file}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    test_mode = args.test_mode or cfg["eval"]["test_mode"]

    # 数据
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    _, _, test_ids = get_split_video_ids(cfg["split"]["train_end"], cfg["split"]["val_end"])

    # 编码器
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    # 模型
    model = build_model(cfg, device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Using random weights.")

    model.eval()

    # 评估
    if test_mode in ("full", "both"):
        metrics = evaluate_full_test(model, encoder, test_ids, gt, narrations, device, cfg, log_dir)
        logger.info("\n[Full Test] Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}")

    if test_mode in ("1k-A", "both"):
        metrics = evaluate_1k_a(model, encoder, test_ids, gt, narrations, device, cfg, log_dir)
        logger.info("\n[1K-A Test] Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
