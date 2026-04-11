"""
AGC 训练主循环: 伪查询聚类中心视频检索。

用法:
    python train.py [--config configs/default_agc.yaml] [--device cuda] [--resume checkpoints/latest.pt]
    python train.py --config configs/default_agc.yaml --overrides 'loss.beta_bal=0.2;training.lr=3e-5'
"""
import argparse
import json
import math
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# 将 AGC 目录加入 path
AGC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGC_DIR))

from data.dataset import AGCDataset, agc_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.agc_module import AGCModel
from models.losses import max_sim_scores, info_nce_loss, orthogonal_regularization_loss
from failure_analysis import run_failure_analysis


# ─────────────── 数据加载工具 ───────────────

def load_gt_annotations(msrvtt_json_path: str):
    """加载 MSR-VTT GT annotations → {video_id: [caption, ...]}"""
    with open(msrvtt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann["caption"])
    return dict(gt)


def get_split_video_ids(train_end=6513, val_end=7010, total=10000):
    """返回 train / val / test 的 video id 列表。"""
    train_ids = [f"video{i}" for i in range(0, train_end)]
    val_ids = [f"video{i}" for i in range(train_end, val_end)]
    test_ids = [f"video{i}" for i in range(val_end, total)]
    return train_ids, val_ids, test_ids


def build_retrieval_pairs(video_ids, gt, frame_feat_dir):
    """构建 (video_id, caption) 对，只保留有特征文件的视频。"""
    feat_dir = Path(frame_feat_dir)
    pairs = []
    for vid in video_ids:
        if vid in gt and (feat_dir / f"{vid}.pt").exists():
            for cap in gt[vid]:
                pairs.append((vid, cap))
    return pairs


# ─────────────── 验证 ───────────────

@torch.no_grad()
def validate(model, text_encoder, val_loader, device, cfg):
    """验证集上计算平均 loss 和 R@1/R@5/R@10。"""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    # 同时收集视频表示用于检索评估
    vid_reprs = {}   # vid → (m, h)
    query_vids = []  # 每个 query 对应的 GT vid
    query_texts = []

    use_amp = cfg["training"]["fp16"] and device != "cpu"

    for video_ids, captions, features, mask in val_loader:
        features = features.to(device)

        text_tokens, text_mask = text_encoder.encode_tokens(captions, device=device)

        with autocast("cuda", enabled=use_amp):
            c, aux = model(features)
            scores = max_sim_scores(c, text_tokens, text_mask)
            logit_scale_max = cfg.get("loss", {}).get("logit_scale_max", 100.0)
            scores = scores * model.logit_scale.exp().clamp(max=logit_scale_max)
            label_smoothing = cfg.get("loss", {}).get("label_smoothing", 0.0)
            loss = info_nce_loss(scores, label_smoothing=label_smoothing)

        total_loss += loss.item()
        total_steps += 1

        # 收集
        for i, vid in enumerate(video_ids):
            if vid not in vid_reprs:
                vid_reprs[vid] = c[i].cpu()
            query_vids.append(vid)
            query_texts.append(captions[i])

    avg_loss = total_loss / max(total_steps, 1)

    # 计算检索指标
    unique_vids = sorted(vid_reprs.keys())
    vid_to_idx = {v: i for i, v in enumerate(unique_vids)}
    all_c = torch.stack([vid_reprs[v] for v in unique_vids]).to(device)  # (N_v, m, h)

    N_q = len(query_vids)
    N_v = len(unique_vids)
    all_scores = torch.zeros(N_q, N_v, device=device)

    # 分批计算查询
    batch_size = 64
    for start in range(0, N_q, batch_size):
        end = min(start + batch_size, N_q)
        batch_texts = query_texts[start:end]
        q_tok, q_msk = text_encoder.encode_tokens(batch_texts, device=device)
        q_tok_norm = F.normalize(q_tok, dim=-1)
        c_norm = F.normalize(all_c, dim=-1)

        # (N_v, batch, m, L)
        sim = torch.einsum("vmh,blh->vbml", c_norm, q_tok_norm)
        max_sim = sim.max(dim=2).values  # (N_v, batch, L)
        mask_exp = q_msk.unsqueeze(0).float()  # (1, batch, L)
        max_sim = max_sim * mask_exp
        batch_scores = max_sim.sum(dim=-1).T  # (batch, N_v)
        all_scores[start:end] = batch_scores

    # 计算 R@K
    ranks = []
    for q_idx in range(N_q):
        gt_vid = query_vids[q_idx]
        gt_col = vid_to_idx[gt_vid]
        sorted_idx = torch.argsort(all_scores[q_idx], descending=True)
        rank = (sorted_idx == gt_col).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    metrics = {
        "R@1": (ranks <= 1).mean() * 100,
        "R@5": (ranks <= 5).mean() * 100,
        "R@10": (ranks <= 10).mean() * 100,
        "MdR": float(np.median(ranks)),
        "MnR": float(np.mean(ranks)),
    }

    model.train()
    return avg_loss, metrics, {"all_scores": all_scores.cpu(), "query_vids": query_vids,
                                "unique_vids": unique_vids,
                                "vid_reprs": vid_reprs}


# ─────────────── 主训练函数 ───────────────

def apply_overrides(cfg, overrides_str):
    """将 'a.b=val;c.d=val' 格式的覆盖应用到 cfg dict。"""
    if not overrides_str:
        return cfg
    for pair in overrides_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        key, val = pair.split("=", 1)
        keys = key.strip().split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # 自动类型转换
        raw = val.strip()
        if raw.lower() in ("true", "false"):
            parsed = raw.lower() == "true"
        else:
            try:
                parsed = int(raw)
            except ValueError:
                try:
                    parsed = float(raw)
                except ValueError:
                    parsed = raw
        d[keys[-1]] = parsed
    return cfg


def main():
    parser = argparse.ArgumentParser(description="AGC Training")
    parser.add_argument("--config", type=str, default="configs/default_agc.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None,
                        help="Semicolon-separated k=v overrides, e.g. 'loss.beta_bal=0.2;training.lr=3e-5'")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for checkpoint/log naming (used by hpsearch)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    # 日志
    log_dir = AGC_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_name or timestamp
    log_file = log_dir / f"train_{run_tag}.log"

    logger = logging.getLogger("agc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    torch.manual_seed(cfg["training"]["seed"])
    if device != "cpu":
        torch.cuda.manual_seed_all(cfg["training"]["seed"])

    # 数据
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, _ = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )
    frame_feat_dir = cfg["data"]["frame_feat_dir"]

    train_pairs = build_retrieval_pairs(train_ids, gt, frame_feat_dir)
    val_pairs = build_retrieval_pairs(val_ids, gt, frame_feat_dir)
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    if len(train_pairs) == 0:
        logger.error("No training data! Run extract_features.py first.")
        return

    train_dataset = AGCDataset(train_pairs, frame_feat_dir)
    val_dataset = AGCDataset(val_pairs, frame_feat_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=agc_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=agc_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # 编码器 (冻结)
    text_encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    text_encoder.eval()

    # AGC 模型
    agc_cfg = cfg["agc"]
    loss_cfg = cfg["loss"]
    model = AGCModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_pseudo_queries=agc_cfg["num_pseudo_queries"],
        num_qformer_layers=agc_cfg["num_qformer_layers"],
        num_heads=agc_cfg["num_heads"],
        ffn_dim=agc_cfg["ffn_dim"],
        dropout=agc_cfg["dropout"],
        pool_type=agc_cfg["pool_type"],
        temperature_init=loss_cfg["temperature_init"],
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"AGC trainable params: {trainable_params:,}")

    # 恢复训练
    start_epoch = 0
    if args.resume:
        import numpy as _np
        torch.serialization.add_safe_globals([_np.float64, _np.float32, _np.int64, _np.int32])
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    use_amp = cfg["training"]["fp16"] and device != "cpu"
    scaler = GradScaler("cuda", enabled=use_amp)

    # 训练循环
    save_dir = AGC_DIR / "checkpoints"
    if args.run_name:
        save_dir = save_dir / args.run_name
    save_dir.mkdir(exist_ok=True, parents=True)
    best_val_r1 = 0.0
    patience_counter = 0
    early_stop_patience = cfg["training"].get("early_stop_patience", 0)
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for video_ids, captions, features, mask in pbar:
            features = features.to(device)

            # 文本编码 (冻结)
            with torch.no_grad():
                text_tokens, text_mask = text_encoder.encode_tokens(captions, device=device)

            with autocast("cuda", enabled=use_amp):
                # Q-Former 前向
                c, aux = model(features)

                # MaxSim 打分
                scores = max_sim_scores(c, text_tokens, text_mask)
                logit_scale_max = loss_cfg.get("logit_scale_max", 100.0)
                scores = scores * model.logit_scale.exp().clamp(max=logit_scale_max)

                # 主损失
                label_smoothing = loss_cfg.get("label_smoothing", 0.0)
                loss = info_nce_loss(scores, label_smoothing=label_smoothing)

                # 正交正则化损失
                beta_ortho = loss_cfg.get("beta_ortho", 0.0)
                if beta_ortho > 0:
                    loss_ortho = orthogonal_regularization_loss(c)
                    loss = loss + beta_ortho * loss_ortho

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_loss_sum += loss.item()
            train_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                scale=f"{model.logit_scale.exp().item():.2f}",
            )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # 验证
        mon_cfg = cfg.get("monitor", {})
        fa_every = mon_cfg.get("failure_analysis_every_n_epochs", 5)
        do_fa = ((epoch + 1) % fa_every == 0) or (epoch == cfg["training"]["epochs"] - 1)
        val_loss, val_metrics, val_extra = validate(
            model, text_encoder, val_loader, device, cfg
        )

        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"R@1: {val_metrics['R@1']:.2f} | "
            f"R@5: {val_metrics['R@5']:.2f} | "
            f"R@10: {val_metrics['R@10']:.2f} | "
            f"MdR: {val_metrics['MdR']:.1f} | "
            f"scale: {model.logit_scale.exp().item():.2f}"
        )

        # 监控: 伪查询多样性 (每 N epochs)
        check_every = mon_cfg.get("check_every_n_epochs", 1)
        if (epoch + 1) % check_every == 0:
            # 抽取最后一个 batch 的输出计算伪查询间平均余弦相似度
            with torch.no_grad():
                c_last = c.detach()
                c_norm_mon = F.normalize(c_last, dim=-1)
                gram = torch.bmm(c_norm_mon, c_norm_mon.transpose(1, 2))  # (B, m, m)
                m_size = gram.size(1)
                mask_diag = ~torch.eye(m_size, dtype=torch.bool, device=gram.device).unsqueeze(0)
                off_diag = gram.masked_select(mask_diag)
                pq_cos_mean = off_diag.mean().item()
                pq_cos_max = off_diag.max().item()
            logger.info(
                f"  [Monitor] PseudoQuery diversity — cos_mean: {pq_cos_mean:.3f}, "
                f"cos_max: {pq_cos_max:.3f}"
            )

        # 失效分析 (每 N epochs 或最后一个 epoch)
        if do_fa:
            logger.info(f"  [Epoch {epoch+1}] 运行失效分析 ...")
            fa_report = run_failure_analysis(
                all_scores=val_extra["all_scores"],
                query_vid_map=val_extra["query_vids"],
                video_ids=val_extra["unique_vids"],
                all_c=val_extra["vid_reprs"],
                logger_fn=logger.info,
            )

        # 保存 checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": cfg,
        }
        torch.save(ckpt_data, save_dir / "latest.pt")

        if val_metrics["R@1"] > best_val_r1:
            best_val_r1 = val_metrics["R@1"]
            patience_counter = 0
            torch.save(ckpt_data, save_dir / "best.pt")
            logger.info(f"  ★ New best R@1: {best_val_r1:.2f}")
        else:
            patience_counter += 1

        # 早停
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={early_stop_patience})")
            break

    logger.info(f"Training complete. Best Val R@1: {best_val_r1:.2f}")
    return best_val_r1


if __name__ == "__main__":
    main()
