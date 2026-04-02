"""
评估脚本：两阶段 Text→Video Retrieval — R@1, R@5, R@10, MdR, MnR。
流程：预增强 narration → 逐查询核过滤 → Sinkhorn 聚类 → 粗筛 → 精排。

用法:
    python evaluate.py [--config configs/default_pq.yaml] [--checkpoint checkpoints/best_model.pt]
"""
import argparse
import json
import logging
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids
from models.clip_encoder import CLIPTextEncoder
from models.nucleus_filter import NucleusFilter
from models.pipeline_pq import PseudoQueryPipeline
from models.prototype_builder import sinkhorn_cluster, build_inverted_index

logger = logging.getLogger("eval")


# ─────────────────────── 指标计算 ───────────────────────

def compute_metrics(scores, query_video_map, video_ids):
    """
    计算 R@1, R@5, R@10, MdR, MnR。
    Args:
        scores: (N_queries, N_videos) tensor 或 ndarray
        query_video_map: list — 第 i 个 query 对应的 gt video_id
        video_ids: list — 全部候选 video_id（与 scores 列对应）
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores.shape[0]

    ranks = []
    for q_idx in range(N_q):
        gt_vid = query_video_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        score_row = scores[q_idx] if isinstance(scores, torch.Tensor) else torch.tensor(scores[q_idx])
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


# ─────────────── 离线预计算：增强所有 narration ─────────────────

@torch.no_grad()
def precompute_enhanced_narrations(
    encoder, nucleus_filter, video_ids, narrations, frame_feat_dir, device,
    max_narrs=64, num_frames=12, feat_dim=512,
):
    """
    一次性对所有视频执行 Co-Attention + Temporal 增强。
    Returns:
        all_narr_sent: dict {vid: (N_i, d)} — 增强后 narration 句子 embedding
        all_narr_tok:  dict {vid: (token_feats, token_mask)} — 原始 narration token（精排用）
        narr_metadata: [(vid, narr_local_idx), ...] — 全局 narration 索引 → 视频映射
        all_enhanced_flat: (N_total, d) — 增强后 narration embeddings 展平
    """
    all_narr_sent = {}
    all_narr_tok = {}
    narr_metadata = []
    enhanced_flat = []

    frame_feat_path = Path(frame_feat_dir)

    for vid in tqdm(video_ids, desc="预增强 narrations"):
        captions = narrations.get(vid, [])
        if not captions:
            continue
        captions = captions[:max_narrs]

        # 帧特征
        pt_file = frame_feat_path / f"{vid}.pt"
        if pt_file.exists():
            frame_feats = torch.load(pt_file, map_location=device, weights_only=True)
        else:
            frame_feats = torch.randn(num_frames, feat_dim, device=device)
            frame_feats = F.normalize(frame_feats, p=2, dim=-1)
        frame_feats = frame_feats.unsqueeze(0)  # (1, K, d)

        # CLIP 编码 narrations
        narr_tok, narr_tok_mask = encoder.encode_tokens(captions, device=device)  # (N_i, L, d), (N_i, L)
        narr_sent = encoder.encode_sentence(captions, device=device)  # (N_i, d)
        N_i = narr_sent.shape[0]

        # 扩展帧特征以匹配 batch 维度 = 1
        narr_sent_batch = narr_sent.unsqueeze(0)  # (1, N_i, d)
        narr_mask = torch.ones(1, N_i, device=device)

        # Co-Attention + Temporal 增强
        _, enhanced_n = nucleus_filter.enhance_features(frame_feats, narr_sent_batch, narr_mask)
        enhanced_n = enhanced_n.squeeze(0)  # (N_i, d)

        all_narr_sent[vid] = enhanced_n.cpu()

        # 缓存原始 token 特征（精排用）
        all_narr_tok[vid] = (narr_tok.cpu(), narr_tok_mask.cpu())

        for local_idx in range(N_i):
            narr_metadata.append((vid, local_idx))
            enhanced_flat.append(enhanced_n[local_idx])

    all_enhanced_flat = torch.stack(enhanced_flat, dim=0)  # (N_total, d)
    return all_narr_sent, all_narr_tok, narr_metadata, all_enhanced_flat


# ─────────────── 逐查询核过滤 + 聚类 + 粗筛 + 精排 ─────────────────

@torch.no_grad()
def two_stage_retrieve(
    query_text, encoder, nucleus_filter, model,
    all_enhanced_flat, narr_metadata, all_narr_sent, all_narr_tok,
    video_ids, device, cfg,
):
    """
    单条查询的两阶段检索。
    Returns:
        video_scores: dict {vid: float} — 所有候选视频的最终分数
    """
    proto_cfg = cfg["prototype"]
    retr_cfg = cfg["retrieval"]
    nf_cfg = cfg["nucleus_filter"]

    # 编码查询
    q_sent = encoder.encode_sentence([query_text], device=device)  # (1, d)
    q_tok, q_mask = encoder.encode_tokens([query_text], device=device)  # (1, L, d), (1, L)

    # 核过滤：query vs 增强后所有 narrations
    N_total = all_enhanced_flat.shape[0]
    enhanced_on_device = all_enhanced_flat.to(device)

    # 计算 query-narration 相似度
    q_norm = F.normalize(q_sent, p=2, dim=-1)  # (1, d)
    n_norm = F.normalize(enhanced_on_device, p=2, dim=-1)  # (N_total, d)
    sim = (q_norm @ n_norm.T).squeeze(0)  # (N_total,)
    weights = F.softmax(sim, dim=0)

    # Nucleus 硬截断
    selected_indices, selected_weights = nucleus_filter.nucleus_select(
        weights.unsqueeze(0), threshold_p=nf_cfg["inference_threshold"]
    )
    sel_idx = selected_indices[0]  # (K_sel,)
    sel_embs = enhanced_on_device[sel_idx]  # (K_sel, d)

    # 收集选中 narrations 对应的视频
    sel_metadata = [narr_metadata[i.item()] for i in sel_idx]

    # Sinkhorn-Knopp 聚类
    K = min(proto_cfg["num_prototypes"], sel_embs.shape[0])
    if K < 2:
        # 太少 narrations，直接暴力匹配
        candidate_vids = list(set(vid for vid, _ in sel_metadata))
    else:
        prototypes, assignments = sinkhorn_cluster(
            sel_embs, K,
            eps=proto_cfg["sinkhorn_eps"],
            niters=proto_cfg["sinkhorn_iters"],
            n_rounds=proto_cfg["cluster_rounds"],
        )
        inverted_index, _ = build_inverted_index(assignments, sel_metadata)

        # 粗筛：query vs 原型 → top-M → 候选视频
        coarse_sim = (q_norm @ prototypes.T).squeeze(0)  # (K,)
        top_m = min(retr_cfg["coarse_top_m"], K)
        _, top_proto_idx = coarse_sim.topk(top_m)

        candidate_vids = set()
        for idx in top_proto_idx.tolist():
            if idx in inverted_index:
                candidate_vids.update(inverted_index[idx])
        candidate_vids = list(candidate_vids)[:retr_cfg["fine_max_candidates"]]

    if not candidate_vids:
        return {}

    # 精排：query tokens × 候选视频 narration tokens
    video_scores = {}
    temperature = model.temperature.item()

    for vid in candidate_vids:
        # 粗筛分数：adapted query vs 该视频 narrations 的均值
        if vid in all_narr_sent:
            vid_narr_embs = all_narr_sent[vid].to(device)  # (N_v, d)
            vid_centroid = F.normalize(vid_narr_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
        else:
            continue

        adapted_q = model.adapt_query(q_sent)  # (1, d)
        coarse_s = (adapted_q @ vid_centroid.T).item() / temperature

        # 精排分数：token-level cross-attention
        if vid in all_narr_tok:
            narr_tok_feats, narr_tok_mask = all_narr_tok[vid]
            narr_tok_feats = narr_tok_feats.to(device)
            narr_tok_mask = narr_tok_mask.to(device)

            # 拼接所有 narration tokens
            N_narr, L_n, d = narr_tok_feats.shape
            flat_tok = narr_tok_feats.reshape(1, N_narr * L_n, d)
            flat_mask = narr_tok_mask.reshape(1, N_narr * L_n)

            # Truncate
            max_len = cfg.get("training", {}).get("max_narr_tokens", 512)
            if flat_tok.shape[1] > max_len:
                flat_tok = flat_tok[:, :max_len, :]
                flat_mask = flat_mask[:, :max_len]

            fine_s = model.rerank(q_tok, q_mask, flat_tok, flat_mask).item()
        else:
            fine_s = 0.0

        video_scores[vid] = coarse_s + cfg["model"]["fine_loss_weight"] * fine_s

    return video_scores


# ─────────────── 评估入口 ─────────────────

@torch.no_grad()
def evaluate_two_stage(
    encoder, nucleus_filter, model, test_video_ids, gt, narrations,
    frame_feat_dir, device, cfg, log_dir, tag="full",
):
    """两阶段评估主流程。"""
    N_test = len(test_video_ids)
    logger.info(f"\n=== Two-Stage Evaluation [{tag}] ({N_test} videos) ===")

    # Step 1: 离线预增强所有 narrations
    logger.info("Step 1: 预增强 narrations (Co-Attention + Temporal) ...")
    all_narr_sent, all_narr_tok, narr_metadata, all_enhanced_flat = \
        precompute_enhanced_narrations(
            encoder, nucleus_filter, test_video_ids, narrations,
            frame_feat_dir, device,
            feat_dim=cfg["encoder"]["feature_dim"],
        )
    logger.info(f"  增强完毕: {len(narr_metadata)} 条 narrations, {len(all_narr_sent)} 个视频")

    # 打印核过滤统计
    log_narr_statistics(all_narr_sent, test_video_ids)

    # 构建查询列表
    queries = []
    query_vid_map = []
    for vid in test_video_ids:
        if vid in gt:
            for cap in gt[vid]:
                queries.append(cap)
                query_vid_map.append(vid)
    logger.info(f"  查询总数: {len(queries)}")

    # Step 2: 逐查询两阶段检索
    logger.info("Step 2: 逐查询核过滤 → 聚类 → 粗筛 → 精排 ...")
    all_scores = np.zeros((len(queries), N_test))
    vid_to_col = {v: i for i, v in enumerate(test_video_ids)}

    for q_idx, query_text in enumerate(tqdm(queries, desc="检索")):
        video_scores = two_stage_retrieve(
            query_text, encoder, nucleus_filter, model,
            all_enhanced_flat, narr_metadata, all_narr_sent, all_narr_tok,
            test_video_ids, device, cfg,
        )
        for vid, score in video_scores.items():
            if vid in vid_to_col:
                all_scores[q_idx, vid_to_col[vid]] = score

    # Step 3: 计算指标
    metrics, ranks = compute_metrics(
        torch.tensor(all_scores), query_vid_map, test_video_ids
    )

    # 保存分析
    analyze_results(all_scores, queries, query_vid_map, test_video_ids,
                    ranks, cfg, log_dir, tag)

    return metrics


def log_narr_statistics(all_narr_sent, video_ids):
    """打印增强后 narration 统计信息。"""
    narr_counts = []
    for vid in video_ids:
        if vid in all_narr_sent:
            narr_counts.append(all_narr_sent[vid].shape[0])
        else:
            narr_counts.append(0)
    counts = np.array(narr_counts)
    logger.info("=" * 60)
    logger.info("增强后 Narration 统计")
    logger.info("=" * 60)
    logger.info(f"  视频总数: {len(video_ids)}, 有 narration 的: {(counts > 0).sum()}")
    if counts.sum() > 0:
        logger.info(f"  每视频 narration 数: min={counts.min()}, max={counts.max()}, "
                     f"mean={counts.mean():.2f}, median={np.median(counts):.1f}")
    logger.info("=" * 60)


def analyze_results(scores, queries, query_vid_map, video_ids, ranks, cfg, log_dir, tag):
    """保存评估分析结果为 JSON。"""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}

    details = []
    for q_idx in range(len(queries)):
        gt_vid = query_vid_map[q_idx]
        score_row = scores[q_idx] if isinstance(scores, np.ndarray) else scores[q_idx].numpy()
        pred_idx = int(np.argmax(score_row))
        pred_vid = video_ids[pred_idx]
        rank = int(ranks[q_idx])

        details.append({
            "query_idx": q_idx,
            "query_text": queries[q_idx],
            "gt_video_id": gt_vid,
            "predicted_video_id": pred_vid,
            "rank": rank,
        })

    summary = {
        "total_queries": len(queries),
        "total_videos": len(video_ids),
        "R@1": float((ranks <= 1).mean() * 100),
        "R@5": float((ranks <= 5).mean() * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "MdR": float(np.median(ranks)),
        "MnR": float(np.mean(ranks)),
    }

    result = {
        "config": {
            "num_prototypes": cfg["prototype"]["num_prototypes"],
            "nucleus_threshold": cfg["nucleus_filter"]["inference_threshold"],
            "test_mode": tag,
            "scheme": "pq",
        },
        "summary": summary,
        "details": details,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"eval_analysis_{tag}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"分析结果已保存到: {json_path}")


# ─────────────── main ─────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_pq.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test_mode", type=str, default="full",
                        help="full | 1k-A | both")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 日志
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 数据
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    _, _, test_ids = get_split_video_ids(cfg["split"]["train_end"], cfg["split"]["val_end"])

    # CLIP 编码器
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    # NucleusFilter
    nf_cfg = cfg["nucleus_filter"]
    nucleus_filter = NucleusFilter(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_heads=nf_cfg["num_heads"],
        temporal_layers=nf_cfg["temporal_layers"],
        temporal_ffn=nf_cfg["temporal_ffn"],
        max_seq_len=nf_cfg["max_seq_len"],
    ).to(device)
    nucleus_filter.eval()

    # PseudoQueryPipeline
    model = PseudoQueryPipeline(
        feature_dim=cfg["encoder"]["feature_dim"],
        adapter_hidden_mult=cfg["model"]["adapter_hidden_mult"],
        reranker_num_heads=cfg["model"]["reranker_num_heads"],
        temperature_init=cfg["model"]["temperature_init"],
        fine_loss_weight=cfg["model"]["fine_loss_weight"],
    ).to(device)
    model.eval()

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        nucleus_filter.load_state_dict(ckpt["nucleus_filter_state_dict"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Using random weights.")

    frame_feat_dir = cfg["data"]["frame_feat_dir"]
    test_mode = args.test_mode

    # 评估
    if test_mode in ("full", "both"):
        metrics = evaluate_two_stage(
            encoder, nucleus_filter, model, test_ids, gt, narrations,
            frame_feat_dir, device, cfg, log_dir, tag="full",
        )
        logger.info("\n[Full Test] Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}")

    if test_mode in ("1k-A", "both"):
        test_1k_ids = [f"video{i}" for i in range(7010, 8010)]
        metrics = evaluate_two_stage(
            encoder, nucleus_filter, model, test_1k_ids, gt, narrations,
            frame_feat_dir, device, cfg, log_dir, tag="1k-A",
        )
        logger.info("\n[1K-A Test] Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
