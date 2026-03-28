"""
诊断脚本：对比训练方式(方案C: per-prototype matching) vs 推理方式(μ·s_T) 的检索效果。

方案C训练时:
  - 视频编码使用 EMA 原型（稳定目标）
  - 查询编码使用 梯度原型（可微）
  - 评分: Σ_k w_k * cos(h_k, q̃_k) / τ  (per-prototype细粒度)

推理时:
  - 两侧都使用梯度原型
  - 评分: s_T · μ^T / τ  (K维点积)

本脚本诊断训练是否有效 & 推理评分方式是否为性能瓶颈。

用法:
    python scripts/diagnose_matching.py [--config configs/default_hybrid.yaml]
                                        [--checkpoint checkpoints/best_model.pt]
                                        [--mode 1k-A|full]
                                        [--train_samples 100]
"""
import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from models.clip_encoder import CLIPTextEncoder
from train import encode_video_captions, build_model

logger = logging.getLogger("diagnose")


# ──────────────────── 编码函数（保留完整表示） ────────────────────

def encode_videos_full(model, encoder, video_ids, narrations, device,
                       use_ema=True, batch_size=32):
    """
    编码所有视频，返回完整表示 h (N, K, d) 和 μ (N, K)。
    use_ema=True 对应训练时视频侧使用 EMA 原型（方案C）。
    """
    all_h, all_mu = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(video_ids), batch_size), desc=f"Encoding videos (ema={use_ema})"):
            batch_vids = video_ids[i:i + batch_size]
            captions_list = [narrations[vid] for vid in batch_vids]
            video_feats, video_mask = encode_video_captions(encoder, captions_list, device)
            h, mu = model.encode_video(video_feats, video_mask, use_ema=use_ema)
            all_h.append(h.cpu())
            all_mu.append(mu.cpu())
    return torch.cat(all_h, dim=0), torch.cat(all_mu, dim=0)


def encode_queries_full(model, encoder, queries, device,
                        use_ema=False, batch_size=256):
    """
    编码所有查询，返回完整表示 s_T (N, K) 和 q_tilde (N, K, d)。
    use_ema=False 对应训练时查询侧使用梯度原型（方案C）。
    """
    all_s_T, all_q_tilde = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Encoding queries (ema={use_ema})"):
            batch_q = queries[i:i + batch_size]
            q_feats, q_mask = encoder.encode_tokens(batch_q, device=device)
            s_T, q_tilde = model.encode_query(q_feats, q_mask, use_ema=use_ema)
            all_s_T.append(s_T.cpu())
            all_q_tilde.append(q_tilde.cpu())
    return torch.cat(all_s_T, dim=0), torch.cat(all_q_tilde, dim=0)


# ──────────────────── 评分函数 ────────────────────

def score_method_A(s_T, mu, temperature):
    """推理方式: s_T · μ^T / τ"""
    return torch.matmul(s_T, mu.T) / temperature


def score_method_C(all_h, all_q_tilde, all_s_T, temperature,
                   query_batch_size=64):
    """
    训练方式(方案C): 逐原型细粒度匹配。
    分批计算避免 OOM: 每次取一批 query 对全部 video 计算。

    scores[q, v] = Σ_k w_qk * cos(h_v^k, q̃_q^k) / τ
    """
    N_q = all_q_tilde.shape[0]
    N_v = all_h.shape[0]
    scores = torch.zeros(N_q, N_v)

    # L2 归一化
    h_norm = F.normalize(all_h, p=2, dim=-1)  # (N_v, K, d)

    for qi in tqdm(range(0, N_q, query_batch_size), desc="Scoring method C"):
        qe = min(qi + query_batch_size, N_q)
        q_batch = F.normalize(all_q_tilde[qi:qe], p=2, dim=-1)  # (Bq, K, d)
        s_batch = all_s_T[qi:qe]  # (Bq, K)

        # 逐原型相似度: (Bq, N_v, K)
        # einsum: query k-th proto vs video k-th proto
        sim_per_proto = torch.einsum("qkd, vkd -> qvk", q_batch, h_norm)

        # L1 归一化权重
        w = F.normalize(s_batch, p=1, dim=-1)  # (Bq, K)

        # 加权求和: (Bq, N_v)
        batch_scores = torch.einsum("qvk, qk -> qv", sim_per_proto, w) / temperature
        scores[qi:qe] = batch_scores

    return scores


# ──────────────────── 指标计算 ────────────────────

def compute_metrics(scores, query_video_map, video_ids):
    """计算 R@1, R@5, R@10, MdR, MnR"""
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
        "R@1": round((ranks <= 1).mean() * 100, 2),
        "R@5": round((ranks <= 5).mean() * 100, 2),
        "R@10": round((ranks <= 10).mean() * 100, 2),
        "MdR": float(np.median(ranks)),
        "MnR": round(float(np.mean(ranks)), 2),
    }, ranks


# ──────────────────── 激活分布诊断 ────────────────────

def diagnose_activation_distribution(all_s_T, all_mu, all_h, all_q_tilde,
                                     query_vid_map, video_ids):
    """分析原型激活分布，诊断 prototype collapse 等问题"""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q, K = all_s_T.shape
    info = {}

    # 1. s_T 稀疏度: 每个 query 平均激活了多少原型(>0.01)
    s_T_active = (all_s_T.abs() > 0.01).float()
    info["query_avg_active_protos"] = round(s_T_active.sum(dim=1).mean().item(), 1)
    info["query_max_active_protos"] = int(s_T_active.sum(dim=1).max().item())
    info["query_min_active_protos"] = int(s_T_active.sum(dim=1).min().item())

    # 2. μ 稀疏度
    mu_active = (all_mu.abs() > 0.01).float()
    info["video_avg_active_protos"] = round(mu_active.sum(dim=1).mean().item(), 1)

    # 3. s_T 每个原型的平均激活值
    s_T_per_proto_mean = all_s_T.mean(dim=0)  # (K,)
    s_T_per_proto_std = all_s_T.std(dim=0)
    info["s_T_per_proto_mean_stats"] = {
        "mean": round(s_T_per_proto_mean.mean().item(), 4),
        "std": round(s_T_per_proto_std.mean().item(), 4),
        "max": round(s_T_per_proto_mean.max().item(), 4),
        "min": round(s_T_per_proto_mean.min().item(), 4),
    }

    # 4. μ 每个原型的分布
    mu_per_proto_mean = all_mu.mean(dim=0)
    info["mu_per_proto_mean_stats"] = {
        "mean": round(mu_per_proto_mean.mean().item(), 4),
        "std": round(mu_per_proto_mean.std().item(), 4),
        "max": round(mu_per_proto_mean.max().item(), 4),
        "min": round(mu_per_proto_mean.min().item(), 4),
    }

    # 5. GT pair 的原型协同激活分析
    #    对每个 (query, gt_video) 对，看 s_T 和 μ 是否在同一组原型上有高激活
    top_k = 10
    overlap_counts = []
    cos_sims_st_mu = []
    for q_idx in range(min(N_q, 5000)):  # 抽样分析
        gt_vid = query_vid_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]

        # s_T 的 top-k 原型
        q_topk = set(torch.argsort(all_s_T[q_idx], descending=True)[:top_k].tolist())
        # μ 的 top-k 原型
        v_topk = set(torch.argsort(all_mu[gt_idx], descending=True)[:top_k].tolist())
        overlap_counts.append(len(q_topk & v_topk))

        # s_T 和 μ 的余弦相似度
        cos_sim = F.cosine_similarity(
            all_s_T[q_idx].unsqueeze(0),
            all_mu[gt_idx].unsqueeze(0)
        ).item()
        cos_sims_st_mu.append(cos_sim)

    info[f"gt_pair_top{top_k}_overlap"] = {
        "mean": round(np.mean(overlap_counts), 2),
        "std": round(np.std(overlap_counts), 2),
    }
    info["gt_pair_cosine_s_T_mu"] = {
        "mean": round(np.mean(cos_sims_st_mu), 4),
        "std": round(np.std(cos_sims_st_mu), 4),
    }

    # 6. 随机负例对比
    neg_cos_sims = []
    rng = np.random.RandomState(42)
    for q_idx in range(min(N_q, 5000)):
        neg_idx = rng.randint(0, len(video_ids))
        cos_sim = F.cosine_similarity(
            all_s_T[q_idx].unsqueeze(0),
            all_mu[neg_idx].unsqueeze(0)
        ).item()
        neg_cos_sims.append(cos_sim)

    info["neg_pair_cosine_s_T_mu"] = {
        "mean": round(np.mean(neg_cos_sims), 4),
        "std": round(np.std(neg_cos_sims), 4),
    }

    # 7. GT pair 用方式C的逐原型cos sim vs 负例
    gt_proto_sims = []
    neg_proto_sims = []
    h_norm = F.normalize(all_h, p=2, dim=-1)
    q_norm = F.normalize(all_q_tilde, p=2, dim=-1)
    for q_idx in range(min(N_q, 2000)):
        gt_idx = vid_to_idx[query_vid_map[q_idx]]
        # per-proto cos sim with GT
        sim_gt = (q_norm[q_idx] * h_norm[gt_idx]).sum(dim=-1)  # (K,)
        gt_proto_sims.append(sim_gt.mean().item())
        # per-proto cos sim with random neg
        neg_idx = rng.randint(0, len(video_ids))
        sim_neg = (q_norm[q_idx] * h_norm[neg_idx]).sum(dim=-1)
        neg_proto_sims.append(sim_neg.mean().item())

    info["gt_pair_avg_proto_cosine"] = {
        "mean": round(np.mean(gt_proto_sims), 4),
        "std": round(np.std(gt_proto_sims), 4),
    }
    info["neg_pair_avg_proto_cosine"] = {
        "mean": round(np.mean(neg_proto_sims), 4),
        "std": round(np.std(neg_proto_sims), 4),
    }

    return info


# ──────────────────── 训练集抽样验证 ────────────────────

def diagnose_train_samples(model, encoder, train_pairs, narrations, device,
                           n_samples=100, seed=42):
    """
    从训练集随机采样 n_samples 个 pair，用方式A和方式C分别评分，
    检查训练数据上的匹配质量。
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(train_pairs), min(n_samples, len(train_pairs)), replace=False)
    sampled = [train_pairs[i] for i in indices]

    # 取唯一视频
    vid_set = list(set(p[0] for p in sampled))
    queries = [p[1] for p in sampled]
    query_vid_map = [p[0] for p in sampled]

    logger.info(f"训练集抽样: {len(sampled)} pairs, {len(vid_set)} unique videos")

    # 编码: 方案C — 视频用 EMA，查询用梯度原型
    all_h, all_mu = encode_videos_full(model, encoder, vid_set, narrations, device,
                                       use_ema=True, batch_size=32)
    all_s_T, all_q_tilde = encode_queries_full(model, encoder, queries, device,
                                               use_ema=False, batch_size=256)

    temperature = model.temperature.item()

    # 方式A
    scores_A = score_method_A(all_s_T, all_mu, temperature)
    metrics_A, _ = compute_metrics(scores_A, query_vid_map, vid_set)

    # 方式C
    scores_C = score_method_C(all_h, all_q_tilde, all_s_T, temperature)
    metrics_C, _ = compute_metrics(scores_C, query_vid_map, vid_set)

    return {
        "n_samples": len(sampled),
        "n_videos": len(vid_set),
        "method_A_inference": metrics_A,
        "method_C_training": metrics_C,
    }


# ──────────────────── 主入口 ────────────────────

def main():
    parser = argparse.ArgumentParser(description="诊断训练方式 vs 推理方式检索效果")
    parser.add_argument("--config", type=str, default="configs/default_hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--mode", type=str, default="1k-A", choices=["1k-A", "full"])
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # ---- 日志 ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"diagnose_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Config: {args.config}, Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {args.mode}, Device: {device}")

    # ---- 数据 ----
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    split_cfg = cfg.get("split", {})
    train_ids, val_ids, test_ids = get_split_video_ids(
        split_cfg.get("train_end", 6513),
        split_cfg.get("val_end", 7010),
    )

    # ---- 模型 ----
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"].get("max_token_length", 77),
    ).to(device)

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    temperature = model.temperature.item()
    logger.info(f"Temperature: {temperature:.6f}")

    # ---- 构建测试集 ----
    if args.mode == "1k-A":
        eval_vids = [f"video{i}" for i in range(7010, 8010)]
        eval_vids = [v for v in eval_vids if v in narrations]
    else:
        eval_vids = [v for v in test_ids if v in narrations]

    queries, query_vid_map = [], []
    for vid in eval_vids:
        if vid in gt:
            caps = gt[vid] if args.mode == "full" else gt[vid][:1]
            for cap in caps:
                queries.append(cap)
                query_vid_map.append(vid)

    logger.info(f"Test videos: {len(eval_vids)}, Test queries: {len(queries)}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 1: 方案C编码 — 视频用 EMA 原型, 查询用梯度原型
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: 方案C编码 (video=EMA, query=gradient)")
    logger.info("=" * 60)

    all_h_ema, all_mu_ema = encode_videos_full(
        model, encoder, eval_vids, narrations, device,
        use_ema=True, batch_size=32
    )
    all_s_T_grad, all_q_tilde_grad = encode_queries_full(
        model, encoder, queries, device,
        use_ema=False, batch_size=256
    )

    logger.info(f"Video h shape: {all_h_ema.shape}, μ shape: {all_mu_ema.shape}")
    logger.info(f"Query s_T shape: {all_s_T_grad.shape}, q̃ shape: {all_q_tilde_grad.shape}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 2: 方式A (推理方式) — s_T · μ / τ
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: 评分方式对比")
    logger.info("=" * 60)

    # 方式A: 用方案C编码的 s_T 和 μ
    scores_A_ema = score_method_A(all_s_T_grad, all_mu_ema, temperature)
    metrics_A_ema, ranks_A_ema = compute_metrics(scores_A_ema, query_vid_map, eval_vids)

    logger.info(f"[方式A, 方案C编码] s_T(grad)·μ(ema)/τ:")
    for k, v in metrics_A_ema.items():
        logger.info(f"  {k}: {v}")

    # 方式C: 逐原型细粒度匹配
    scores_C = score_method_C(all_h_ema, all_q_tilde_grad, all_s_T_grad, temperature)
    metrics_C, ranks_C = compute_metrics(scores_C, query_vid_map, eval_vids)

    logger.info(f"\n[方式C, 方案C编码] Σ w_k·cos(h_k,q̃_k)/τ:")
    for k, v in metrics_C.items():
        logger.info(f"  {k}: {v}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 2b: 推理方式编码对比 (两侧都用梯度原型，即现有evaluate.py的做法)
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "-" * 60)
    logger.info("Phase 2b: 推理编码 (video=gradient, query=gradient) — 现有evaluate.py方式")
    logger.info("-" * 60)

    all_h_grad, all_mu_grad = encode_videos_full(
        model, encoder, eval_vids, narrations, device,
        use_ema=False, batch_size=32
    )
    all_s_T_grad2, all_q_tilde_grad2 = encode_queries_full(
        model, encoder, queries, device,
        use_ema=False, batch_size=256
    )

    scores_A_grad = score_method_A(all_s_T_grad2, all_mu_grad, temperature)
    metrics_A_grad, ranks_A_grad = compute_metrics(scores_A_grad, query_vid_map, eval_vids)

    logger.info(f"[方式A, 推理编码] s_T(grad)·μ(grad)/τ:")
    for k, v in metrics_A_grad.items():
        logger.info(f"  {k}: {v}")

    scores_C_grad = score_method_C(all_h_grad, all_q_tilde_grad2, all_s_T_grad2, temperature)
    metrics_C_grad, ranks_C_grad = compute_metrics(scores_C_grad, query_vid_map, eval_vids)

    logger.info(f"\n[方式C, 推理编码] Σ w_k·cos(h_k,q̃_k)/τ:")
    for k, v in metrics_C_grad.items():
        logger.info(f"  {k}: {v}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 3: 激活分布诊断
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: 原型激活分布诊断")
    logger.info("=" * 60)

    activation_info = diagnose_activation_distribution(
        all_s_T_grad, all_mu_ema, all_h_ema, all_q_tilde_grad,
        query_vid_map, eval_vids
    )

    for k, v in activation_info.items():
        logger.info(f"  {k}: {v}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 4: 训练集抽样验证
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: 训练集抽样验证")
    logger.info("=" * 60)

    train_pairs = build_retrieval_pairs(train_ids, gt)
    train_diag = diagnose_train_samples(
        model, encoder, train_pairs, narrations, device,
        n_samples=args.train_samples, seed=42
    )

    logger.info(f"训练集抽样: {train_diag['n_samples']} pairs / {train_diag['n_videos']} videos")
    logger.info(f"  方式A (推理): {train_diag['method_A_inference']}")
    logger.info(f"  方式C (训练): {train_diag['method_C_training']}")

    # ════════════════════════════════════════════════════════════════
    #  Phase 5: 汇总输出
    # ════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("汇总对比表")
    logger.info("=" * 60)

    header = f"{'配置':<40} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MdR':>8} {'MnR':>8}"
    logger.info(header)
    logger.info("-" * 78)

    rows = [
        ("方案C编码 + 方式A(s_T·μ/τ)", metrics_A_ema),
        ("方案C编码 + 方式C(per-proto)", metrics_C),
        ("推理编码 + 方式A(s_T·μ/τ) [baseline]", metrics_A_grad),
        ("推理编码 + 方式C(per-proto)", metrics_C_grad),
    ]
    for name, m in rows:
        logger.info(f"{name:<40} {m['R@1']:>6.2f} {m['R@5']:>6.2f} {m['R@10']:>7.2f} {m['MdR']:>8.1f} {m['MnR']:>8.2f}")

    logger.info(f"\n训练集(抽样 {train_diag['n_samples']} pairs):")
    logger.info(f"  方式A(推理): R@1={train_diag['method_A_inference']['R@1']}")
    logger.info(f"  方式C(训练): R@1={train_diag['method_C_training']['R@1']}")

    # ---- 保存结果 ----
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "temperature": temperature,
        "n_videos": len(eval_vids),
        "n_queries": len(queries),
        "metrics": {
            "schemeC_encode_methodA": metrics_A_ema,
            "schemeC_encode_methodC": metrics_C,
            "inference_encode_methodA": metrics_A_grad,
            "inference_encode_methodC": metrics_C_grad,
        },
        "activation_diagnosis": activation_info,
        "train_sample_diagnosis": train_diag,
    }

    json_path = log_dir / f"diagnose_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"\n详细结果已保存: {json_path}")
    logger.info(f"日志已保存: {log_file}")


if __name__ == "__main__":
    main()
