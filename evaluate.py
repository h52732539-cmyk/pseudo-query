"""
评估脚本：Text→Video Retrieval — R@1, R@5, R@10, MdR, MnR。

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
from models.pipeline import VIBPseudoQueryModel
from models.scoring import uncertainty_aware_score
from train import encode_video_captions

logger = logging.getLogger("eval")


def build_video_representations(model, encoder, video_ids, narrations, device, batch_size=32):
    """
    离线编码所有测试视频 → (mu, sigma_sq) 表示。
    Returns:
        all_mu: (N_videos, K)
        all_sigma_sq: (N_videos, K)
    """
    all_mu = []
    all_sigma_sq = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(video_ids), batch_size), desc="Encoding videos"):
            batch_vids = video_ids[i : i + batch_size]
            captions_list = [narrations[vid] for vid in batch_vids]
            video_feats, video_mask = encode_video_captions(encoder, captions_list, device)
            mu, sigma_sq = model.encode_video(video_feats, video_mask)
            all_mu.append(mu.cpu())
            all_sigma_sq.append(sigma_sq.cpu())

    return torch.cat(all_mu, dim=0), torch.cat(all_sigma_sq, dim=0)


def build_query_representations(model, encoder, queries, device, batch_size=256):
    """
    离线编码所有查询 → s_T 表示。
    Returns:
        all_s_T: (N_queries, K)
    """
    all_s_T = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_queries = queries[i : i + batch_size]
            query_feats, query_mask = encoder.encode_tokens(batch_queries, device=device)
            s_T = model.encode_query(query_feats, query_mask)
            all_s_T.append(s_T.cpu())

    return torch.cat(all_s_T, dim=0)


def compute_metrics(scores, query_video_map, video_ids):
    """
    计算 R@1, R@5, R@10, MdR, MnR。
    Args:
        scores: (N_queries, N_videos) — 越大越好
        query_video_map: List[str] — 每个 query 的 ground-truth video_id
        video_ids: List[str] — gallery 中视频的顺序
    Returns:
        metrics: dict
        ranks: np.ndarray — 每个 query 的 GT 视频排名 (1-indexed)
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores.shape[0]

    ranks = []
    for q_idx in range(N_q):
        gt_vid = query_video_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        score_row = scores[q_idx]
        # 降序排列，找出 gt 的排名 (0-indexed)
        sorted_indices = torch.argsort(score_row, descending=True)
        rank = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
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


def print_prototype_statistics(all_mu, all_sigma_sq, video_ids):
    """
    打印测试集视频的原型聚类特征。
    Args:
        all_mu: (N_videos, K)
        all_sigma_sq: (N_videos, K)
        video_ids: List[str]
    """
    N, K = all_mu.shape
    # 每个视频分配到 μ 最大激活值对应的原型
    cluster_assignment = torch.argmax(all_mu, dim=1).numpy()  # (N,)

    # 统计每个聚类的视频数
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

    # Top-10 最大聚类
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"  Top-10 最大聚类: {[(c, n) for c, n in sorted_clusters[:10]]}")
    # Top-10 最小非空聚类
    logger.info(f"  Top-10 最小非空聚类: {[(c, n) for c, n in sorted_clusters[-10:]]}")

    # 每个聚类的平均 μ 和 σ² 统计
    mu_max_vals = all_mu.max(dim=1).values  # (N,) — 每个视频归属聚类的 μ 值
    sigma_at_cluster = all_sigma_sq[torch.arange(N), cluster_assignment]  # (N,)
    logger.info(f"  归属聚类维度的 μ 统计: mean={mu_max_vals.mean():.4f}, std={mu_max_vals.std():.4f}")
    logger.info(f"  归属聚类维度的 σ² 统计: mean={sigma_at_cluster.mean():.4f}, std={sigma_at_cluster.std():.4f}")

    # 全局 μ 和 σ² 统计
    logger.info(f"  全局 μ 统计: mean={all_mu.mean():.4f}, std={all_mu.std():.4f}, "
                f"min={all_mu.min():.4f}, max={all_mu.max():.4f}")
    logger.info(f"  全局 σ² 统计: mean={all_sigma_sq.mean():.4f}, std={all_sigma_sq.std():.4f}, "
                f"min={all_sigma_sq.min():.4f}, max={all_sigma_sq.max():.4f}")
    logger.info("=" * 60)

    return cluster_assignment


def analyze_results(scores, all_mu, all_sigma_sq, queries, query_vid_map,
                    video_ids, ranks, variance_penalty, cfg, log_dir, tag="full"):
    """
    对评估结果进行详细分析，保存为 JSON。
    Args:
        scores: (N_queries, N_videos)
        all_mu: (N_videos, K)
        all_sigma_sq: (N_videos, K)
        queries: List[str]
        query_vid_map: List[str]
        video_ids: List[str]
        ranks: np.ndarray (N_queries,)
        variance_penalty: float
        cfg: dict
        log_dir: Path
        tag: str — 标识评估模式
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_videos, K = all_mu.shape

    # 聚类分配
    cluster_assignment = torch.argmax(all_mu, dim=1).numpy()  # (N_videos,)

    details = []
    same_cluster_count = 0
    rank1_same_cluster_count = 0
    ranks_same = []
    ranks_diff = []

    for q_idx in range(len(queries)):
        gt_vid = query_vid_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]

        # 预测视频 (Top-1)
        pred_idx = torch.argsort(scores[q_idx], descending=True)[0].item()
        pred_vid = video_ids[pred_idx]

        gt_cluster = int(cluster_assignment[gt_idx])
        pred_cluster = int(cluster_assignment[pred_idx])
        same_cluster = (gt_cluster == pred_cluster)

        # GT 视频在其归属聚类维度的激活值
        gt_raw = float(all_mu[gt_idx, gt_cluster].item())
        gt_pen = float((all_mu[gt_idx, gt_cluster] - variance_penalty * all_sigma_sq[gt_idx, gt_cluster]).item())

        # 预测视频在其归属聚类维度的激活值
        pred_raw = float(all_mu[pred_idx, pred_cluster].item())
        pred_pen = float((all_mu[pred_idx, pred_cluster] - variance_penalty * all_sigma_sq[pred_idx, pred_cluster]).item())

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
            "gt_raw_activation": round(gt_raw, 6),
            "gt_penalized_activation": round(gt_pen, 6),
            "pred_raw_activation": round(pred_raw, 6),
            "pred_penalized_activation": round(pred_pen, 6),
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
            "variance_penalty": variance_penalty,
            "num_prototypes": K,
            "test_mode": tag,
            "num_videos": N_videos,
        },
        "summary": summary,
        "details": details,
    }

    # 保存 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"eval_analysis_{tag}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"分析结果已保存到: {json_path}")
    logger.info(f"分析摘要: {json.dumps(summary, ensure_ascii=False)}")


def evaluate_full_test(model, encoder, test_ids, gt, narrations, device, cfg, log_dir):
    """全量测试集评估（每个GT caption作为独立查询）"""
    logger.info(f"\n=== Full Test Evaluation ({len(test_ids)} videos) ===")

    # 视频侧
    all_mu, all_sigma_sq = build_video_representations(
        model, encoder, test_ids, narrations, device
    )

    # 打印原型聚类统计
    print_prototype_statistics(all_mu, all_sigma_sq, test_ids)

    # 查询侧：收集所有 GT query 及对应 video_id
    queries = []
    query_vid_map = []
    for vid in test_ids:
        if vid in gt:
            for cap in gt[vid]:
                queries.append(cap)
                query_vid_map.append(vid)

    logger.info(f"Total queries: {len(queries)}")
    all_s_T = build_query_representations(model, encoder, queries, device)

    # 计算分数矩阵
    variance_penalty = cfg["model"]["variance_penalty"]
    temperature = model.temperature.item()
    scores = uncertainty_aware_score(all_s_T, all_mu, all_sigma_sq, variance_penalty, temperature=temperature)

    metrics, ranks = compute_metrics(scores, query_vid_map, test_ids)

    # 详细分析
    analyze_results(scores, all_mu, all_sigma_sq, queries, query_vid_map,
                    test_ids, ranks, variance_penalty, cfg, log_dir, tag="full")

    return metrics


def evaluate_1k_a(model, encoder, test_ids, gt, narrations, device, cfg, log_dir):
    """
    MSR-VTT 1K-A 评估：video7010~video8009 的 1000 个视频，
    每个视频取第一条 GT caption 作为查询。
    """
    test_1k_ids = [f"video{i}" for i in range(7010, 8010)]
    logger.info(f"\n=== 1K-A Test Evaluation ({len(test_1k_ids)} videos) ===")

    all_mu, all_sigma_sq = build_video_representations(
        model, encoder, test_1k_ids, narrations, device
    )

    # 打印原型聚类统计
    print_prototype_statistics(all_mu, all_sigma_sq, test_1k_ids)

    queries = []
    query_vid_map = []
    for vid in test_1k_ids:
        if vid in gt and len(gt[vid]) > 0:
            queries.append(gt[vid][0])
            query_vid_map.append(vid)

    logger.info(f"Total queries: {len(queries)}")
    all_s_T = build_query_representations(model, encoder, queries, device)

    variance_penalty = cfg["model"]["variance_penalty"]
    temperature = model.temperature.item()
    scores = uncertainty_aware_score(all_s_T, all_mu, all_sigma_sq, variance_penalty, temperature=temperature)

    metrics, ranks = compute_metrics(scores, query_vid_map, test_1k_ids)

    # 详细分析
    analyze_results(scores, all_mu, all_sigma_sq, queries, query_vid_map,
                    test_1k_ids, ranks, variance_penalty, cfg, log_dir, tag="1k-A")

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

    logger.info(f"Config: {args.config}")
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

    # 模型 — 动态推断 num_prototypes
    proto_path = Path(cfg["data"].get("prototype_path", "checkpoints/prototypes.pt"))
    num_prototypes = cfg["prototype"]["num_prototypes"]  # 默认值
    if proto_path.exists():
        proto_data = torch.load(proto_path, map_location="cpu", weights_only=False)
        if isinstance(proto_data, dict) and "prototypes" in proto_data:
            num_prototypes = proto_data["prototypes"].shape[0]
        else:
            num_prototypes = proto_data.shape[0]
        logger.info(f"Inferred num_prototypes={num_prototypes} from {proto_path}")

    model = VIBPseudoQueryModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_prototypes=num_prototypes,
        aggregation=cfg["model"]["aggregation"],
        temperature_init=cfg["model"]["temperature_init"],
        variance_penalty=cfg["model"]["variance_penalty"],
        free_bits=cfg["model"].get("free_bits", 0.1),
    ).to(device)

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
