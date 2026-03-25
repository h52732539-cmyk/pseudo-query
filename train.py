"""
训练主循环：逐原型对比损失 + SwAV 原型学习。
支持方案B (SwAV) 和方案C (Hybrid EMA+SwAV)。

用法:
    python train.py [--config configs/default.yaml]
"""
import argparse
import os
import math
import logging
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from data.dataset import PseudoQueryMultiViewDataset, multiview_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.pipeline_swav import SwAVPipelineModel
from models.pipeline_hybrid import HybridPipelineModel


def encode_video_captions(encoder, captions_list, device, max_tokens=512):
    """
    将一个 batch 中所有视频的伪查询 captions 编码为 padded token 特征。

    Args:
        captions_list: List[List[str]] — batch 中每个视频的 caption 列表
    Returns:
        video_token_features: (B, M_max, d)
        video_token_mask: (B, M_max)
    """
    batch_features = []
    batch_lengths = []

    for captions in captions_list:
        tok_feats, attn_mask = encoder.encode_tokens(captions, device=device)
        tokens_list = []
        for j in range(tok_feats.shape[0]):
            valid_len = attn_mask[j].sum().item()
            tokens_list.append(tok_feats[j, :valid_len])
        all_tokens = torch.cat(tokens_list, dim=0)
        if all_tokens.shape[0] > max_tokens:
            all_tokens = all_tokens[:max_tokens]
        batch_features.append(all_tokens)
        batch_lengths.append(all_tokens.shape[0])

    M_max = max(batch_lengths)
    d = batch_features[0].shape[-1]
    B = len(batch_features)

    padded = torch.zeros(B, M_max, d, device=device)
    mask = torch.zeros(B, M_max, device=device)
    for i, feat in enumerate(batch_features):
        L = feat.shape[0]
        padded[i, :L] = feat
        mask[i, :L] = 1.0

    return padded, mask


def build_model(cfg, device):
    """根据 scheme 配置构建模型"""
    scheme = cfg.get("scheme", "swav")
    swav_cfg = cfg.get("swav", {})

    if scheme == "hybrid":
        ema_cfg = cfg.get("ema", {})
        model = HybridPipelineModel(
            feature_dim=cfg["encoder"]["feature_dim"],
            num_prototypes=cfg["prototype"]["num_prototypes"],
            aggregation=cfg["model"]["aggregation"],
            temperature_init=cfg["model"]["temperature_init"],
            sinkhorn_eps=swav_cfg.get("sinkhorn_eps", 0.05),
            sinkhorn_iters=swav_cfg.get("sinkhorn_iters", 3),
            swav_temperature=swav_cfg.get("temperature", 0.1),
            ema_decay=ema_cfg.get("decay", 0.999),
            dead_proto_threshold=ema_cfg.get("dead_proto_threshold", 100),
        )
    else:
        model = SwAVPipelineModel(
            feature_dim=cfg["encoder"]["feature_dim"],
            num_prototypes=cfg["prototype"]["num_prototypes"],
            aggregation=cfg["model"]["aggregation"],
            temperature_init=cfg["model"]["temperature_init"],
            sinkhorn_eps=swav_cfg.get("sinkhorn_eps", 0.05),
            sinkhorn_iters=swav_cfg.get("sinkhorn_iters", 3),
            swav_temperature=swav_cfg.get("temperature", 0.1),
        )

    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 初始化日志 ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
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
    logger.info(f"Log file: {log_file}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    torch.manual_seed(cfg["training"]["seed"])

    # ---- 数据加载 ----
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, _ = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )

    train_pairs = build_retrieval_pairs(train_ids, gt)
    val_pairs = build_retrieval_pairs(val_ids, gt)
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    train_dataset = PseudoQueryMultiViewDataset(train_pairs, narrations)
    val_dataset = PseudoQueryMultiViewDataset(val_pairs, narrations)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=multiview_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=multiview_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # ---- 编码器 (冻结) ----
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    # ---- 模型 ----
    model = build_model(cfg, device)
    logger.info(f"Model: {model.__class__.__name__}, K={cfg['prototype']['num_prototypes']}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # ---- 优化器 & 调度器 ----
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

    use_amp = cfg["training"]["fp16"] and device != "cpu"
    amp_device = "cuda" if device != "cpu" else "cpu"
    scaler = GradScaler(amp_device, enabled=use_amp)

    swav_alpha = cfg["training"].get("swav_alpha", 0.5)
    is_hybrid = scheme == "hybrid"

    # ---- 训练循环 ----
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        # === Train ===
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]")
        for video_ids, caps_v1, caps_v2, queries in pbar:
            # 编码（冻结 CLIP）
            with torch.no_grad():
                v1_feats, v1_mask = encode_video_captions(encoder, caps_v1, device)
                v2_feats, v2_mask = encode_video_captions(encoder, caps_v2, device)
                q_feats, q_mask = encoder.encode_tokens(queries, device=device)

            with autocast(amp_device, enabled=use_amp):
                h_avg, mu1, mu2, s_T, q_tilde = model(
                    v1_feats, v1_mask, v2_feats, v2_mask, q_feats, q_mask
                )
                loss_total, loss_match, loss_swav = model.compute_loss(
                    h_avg, mu1, mu2, s_T, q_tilde, swav_alpha=swav_alpha
                )

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 方案C: 每步更新 EMA + 死原型检查
            if is_hybrid:
                n_dead = model.post_step(s_T.detach(), v1_feats.detach())
                if n_dead > 0:
                    logger.info(f"  [EMA] Reinitialized {n_dead} dead prototypes")

            train_loss_sum += loss_total.item()
            train_steps += 1
            pbar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                match=f"{loss_match.item():.4f}",
                swav=f"{loss_swav.item():.4f}",
                tau=f"{model.temperature.item():.4f}",
            )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # === Validation ===
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for video_ids, caps_v1, caps_v2, queries in tqdm(val_loader, desc="[Val]"):
                v1_feats, v1_mask = encode_video_captions(encoder, caps_v1, device)
                v2_feats, v2_mask = encode_video_captions(encoder, caps_v2, device)
                q_feats, q_mask = encoder.encode_tokens(queries, device=device)

                with autocast(amp_device, enabled=use_amp):
                    h_avg, mu1, mu2, s_T, q_tilde = model(
                        v1_feats, v1_mask, v2_feats, v2_mask, q_feats, q_mask
                    )
                    loss_total, _, _ = model.compute_loss(
                        h_avg, mu1, mu2, s_T, q_tilde, swav_alpha=swav_alpha
                    )

                val_loss_sum += loss_total.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        logger.info(
            f"EPOCH_SUMMARY | epoch={epoch+1} | train_loss={avg_train_loss:.6f} "
            f"| val_loss={avg_val_loss:.6f} | swav_alpha={swav_alpha} "
            f"| tau={model.temperature.item():.6f} | lr={current_lr:.8f}"
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "scheme": scheme,
                },
                save_dir / "best_model.pt",
            )
            logger.info(f"  ✓ Best model saved (val_loss={avg_val_loss:.4f})")

        # Save latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "scheme": scheme,
            },
            save_dir / "latest_model.pt",
        )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
