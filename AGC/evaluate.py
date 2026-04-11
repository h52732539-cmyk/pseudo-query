"""
AGC 评估脚本: 计算 R@1, R@5, R@10, MdR, MnR。

用法:
    python evaluate.py [--config configs/default_agc.yaml] [--checkpoint checkpoints/best.pt]
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

AGC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGC_DIR))

from models.clip_encoder import CLIPTextEncoder
from models.agc_module import AGCModel
from models.losses import max_sim_scores
from failure_analysis import run_failure_analysis

logger = logging.getLogger("agc_eval")


# ─────────────── 数据加载 ───────────────

def load_gt_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann["caption"])
    return dict(gt)


def get_split_video_ids(train_end=6513, val_end=7010, total=10000):
    train_ids = [f"video{i}" for i in range(0, train_end)]
    val_ids = [f"video{i}" for i in range(train_end, val_end)]
    test_ids = [f"video{i}" for i in range(val_end, total)]
    return train_ids, val_ids, test_ids


# ─────────────── 检索指标 ───────────────

def compute_metrics(scores, query_vid_map, video_ids):
    """
    计算检索指标。
    Args:
        scores: (N_q, N_v) tensor
        query_vid_map: list[str] — 第 i 个 query 对应的 GT video_id
        video_ids: list[str] — 所有候选视频 ID
    Returns:
        metrics: dict, ranks: np.array
    """
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    N_q = scores.shape[0]
    ranks = []

    for q_idx in range(N_q):
        gt_vid = query_vid_map[q_idx]
        gt_idx = vid_to_idx[gt_vid]
        row = scores[q_idx] if isinstance(scores, torch.Tensor) else torch.tensor(scores[q_idx])
        sorted_idx = torch.argsort(row, descending=True)
        rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": float((ranks <= 1).mean() * 100),
        "R@5": float((ranks <= 5).mean() * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "MdR": float(np.median(ranks)),
        "MnR": float(np.mean(ranks)),
    }, ranks


# ─────────────── 预计算视频表示 ───────────────

@torch.no_grad()
def precompute_video_representations(model, video_ids, frame_feat_dir, device, batch_size=64):
    """对所有视频计算压缩表示 {c_k}。"""
    feat_dir = Path(frame_feat_dir)
    all_c = {}

    valid_vids = [v for v in video_ids if (feat_dir / f"{v}.pt").exists()]
    logger.info(f"Computing representations for {len(valid_vids)} videos ...")

    for start in tqdm(range(0, len(valid_vids), batch_size), desc="Video encoding"):
        batch_vids = valid_vids[start:start + batch_size]
        feats = []
        for vid in batch_vids:
            f = torch.load(feat_dir / f"{vid}.pt", weights_only=True).float()
            feats.append(f)

        # 对齐长度
        max_n = max(f.size(0) for f in feats)
        h = feats[0].size(1)
        B = len(feats)
        padded = torch.zeros(B, max_n, h)
        for i, f in enumerate(feats):
            padded[i, :f.size(0)] = f
        padded = padded.to(device)

        c, aux = model(padded)  # (B, m, h)
        for i, vid in enumerate(batch_vids):
            all_c[vid] = c[i].cpu()

    return all_c


# ─────────────── 评估入口 ───────────────

@torch.no_grad()
def evaluate(model, text_encoder, video_ids, gt, frame_feat_dir, device, cfg):
    """完整评估流程。"""
    model.eval()

    # Step 1: 预计算视频表示
    all_c = precompute_video_representations(model, video_ids, frame_feat_dir, device)
    valid_vids = [v for v in video_ids if v in all_c]

    # Step 2: 构建查询列表
    queries = []
    query_vid_map = []
    for vid in valid_vids:
        if vid in gt:
            for cap in gt[vid]:
                queries.append(cap)
                query_vid_map.append(vid)

    logger.info(f"Videos: {len(valid_vids)}, Queries: {len(queries)}")

    # Step 3: 计算所有 (query, video) 分数
    all_c_tensor = torch.stack([all_c[v] for v in valid_vids]).to(device)  # (N_v, m, h)
    c_norm = F.normalize(all_c_tensor, dim=-1)

    N_q = len(queries)
    N_v = len(valid_vids)
    all_scores = torch.zeros(N_q, N_v)

    text_batch_size = 128
    for start in tqdm(range(0, N_q, text_batch_size), desc="Query scoring"):
        end = min(start + text_batch_size, N_q)
        batch_texts = queries[start:end]
        q_tok, q_msk = text_encoder.encode_tokens(batch_texts, device=device)
        q_norm = F.normalize(q_tok, dim=-1)

        # (N_v, batch, m, L)
        sim = torch.einsum("vmh,blh->vbml", c_norm, q_norm)
        max_sim = sim.max(dim=2).values  # (N_v, batch, L)
        mask_exp = q_msk.unsqueeze(0).float()
        max_sim = max_sim * mask_exp
        batch_scores = max_sim.sum(dim=-1).T  # (batch, N_v)
        all_scores[start:end] = batch_scores.cpu()

    # Step 4: 计算指标
    metrics, ranks = compute_metrics(all_scores, query_vid_map, valid_vids)
    return metrics, ranks, all_scores, query_vid_map, valid_vids, all_c


def main():
    parser = argparse.ArgumentParser(description="AGC Evaluation")
    parser.add_argument("--config", type=str, default="configs/default_agc.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--no-failure-analysis", action="store_true", help="跳过失效分析")
    args = parser.parse_args()

    # 日志
    log_dir = AGC_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / f"eval_{timestamp}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Checkpoint: {args.checkpoint}")

    # 加载模型 (checkpoint 包含 numpy 标量，需注册 safe global)
    import numpy as np
    torch.serialization.add_safe_globals([np.float64, np.float32, np.int64, np.int32])
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", cfg)

    agc_cfg = saved_cfg["agc"]
    loss_cfg = saved_cfg["loss"]
    model = AGCModel(
        feature_dim=saved_cfg["encoder"]["feature_dim"],
        num_pseudo_queries=agc_cfg["num_pseudo_queries"],
        num_qformer_layers=agc_cfg.get("num_qformer_layers", agc_cfg.get("num_encoder_layers", 4)),
        num_heads=agc_cfg["num_heads"],
        ffn_dim=agc_cfg["ffn_dim"],
        dropout=agc_cfg["dropout"],
        pool_type=agc_cfg["pool_type"],
        temperature_init=loss_cfg["temperature_init"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text_encoder = CLIPTextEncoder(
        model_name=saved_cfg["encoder"]["name"],
        max_length=saved_cfg["encoder"]["max_token_length"],
    ).to(device)
    text_encoder.eval()

    # 数据
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, test_ids = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )
    eval_ids = val_ids if args.split == "val" else test_ids

    metrics, ranks, all_scores, query_vid_map, valid_vids, all_c = evaluate(
        model, text_encoder, eval_ids, gt, cfg["data"]["frame_feat_dir"], device, cfg
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"  {args.split.upper()} Results")
    logger.info(f"  R@1:  {metrics['R@1']:.2f}")
    logger.info(f"  R@5:  {metrics['R@5']:.2f}")
    logger.info(f"  R@10: {metrics['R@10']:.2f}")
    logger.info(f"  MdR:  {metrics['MdR']:.1f}")
    logger.info(f"  MnR:  {metrics['MnR']:.1f}")
    logger.info(f"{'='*50}")

    # 失效分析
    if not args.no_failure_analysis:
        logger.info(f"\n{'='*50}")
        logger.info(f"  {args.split.upper()} 失效分析")
        logger.info(f"{'='*50}")
        fa_report = run_failure_analysis(
            all_scores=all_scores,
            query_vid_map=query_vid_map,
            video_ids=valid_vids,
            all_c=all_c,
            logger_fn=logger.info,
        )

    # 保存结果
    result_path = log_dir / f"eval_{args.split}_{timestamp}.json"
    result_data = {"metrics": metrics, "config": cfg}
    if not args.no_failure_analysis:
        # 去掉不可序列化的内部字段
        result_data["failure_analysis"] = fa_report
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
