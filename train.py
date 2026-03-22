"""
训练主循环：InfoNCE + KL loss + β-annealing。

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
from data.dataset import PseudoQueryTrainDataset, train_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.pipeline import VIBPseudoQueryModel


def get_beta(epoch, total_epochs, beta_start, beta_end):
    """β-annealing: 线性从 beta_start 增长到 beta_end"""
    ratio = min(epoch / max(total_epochs - 1, 1), 1.0)
    return beta_start + (beta_end - beta_start) * ratio


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
        # 编码单个视频的所有 captions
        tok_feats, attn_mask = encoder.encode_tokens(captions, device=device)
        # 收集所有 valid tokens
        tokens_list = []
        for j in range(tok_feats.shape[0]):
            valid_len = attn_mask[j].sum().item()
            tokens_list.append(tok_feats[j, :valid_len])  # (valid_len, d)
        all_tokens = torch.cat(tokens_list, dim=0)  # (M_i, d)
        # 截断过长的 token 序列
        if all_tokens.shape[0] > max_tokens:
            all_tokens = all_tokens[:max_tokens]
        batch_features.append(all_tokens)
        batch_lengths.append(all_tokens.shape[0])

    # Pad to max length in batch
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

    logger.info(f"Config: {args.config}")
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

    train_dataset = PseudoQueryTrainDataset(train_pairs, narrations)
    val_dataset = PseudoQueryTrainDataset(val_pairs, narrations)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=train_collate_fn,
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

    # ---- 加载原型 & 模型 ----
    proto_path = Path(cfg["data"]["prototype_path"])
    proto_init = None
    num_prototypes = cfg["prototype"]["num_prototypes"]  # 默认值

    if proto_path.exists():
        proto_data = torch.load(proto_path, map_location="cpu", weights_only=False)
        # 支持新格式（dict with metadata）和旧格式（直接 tensor）
        if isinstance(proto_data, dict) and "prototypes" in proto_data:
            proto_init = proto_data["prototypes"]
            num_prototypes = proto_init.shape[0]
            logger.info(f"Loaded prototypes from {proto_path} (method={proto_data.get('method')}, K={num_prototypes})")
        else:
            proto_init = proto_data
            num_prototypes = proto_init.shape[0]
            logger.info(f"Loaded prototypes from {proto_path} (legacy format, K={num_prototypes})")
    else:
        logger.warning(f"{proto_path} not found. Using random initialization with K={num_prototypes}.")

    # Memory Bank 配置
    mb_size = 0
    if cfg["training"].get("use_memory_bank", False):
        mb_size = cfg["training"].get("memory_bank_size", 4096)

    model = VIBPseudoQueryModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_prototypes=num_prototypes,
        aggregation=cfg["model"]["aggregation"],
        temperature_init=cfg["model"]["temperature_init"],
        variance_penalty=cfg["model"]["variance_penalty"],
        prototype_init_weights=proto_init,
        free_bits=cfg["model"].get("free_bits", 0.1),
        memory_bank_size=mb_size,
    ).to(device)

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

    # ---- 训练循环 ----
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        beta = get_beta(
            epoch, cfg["training"]["epochs"],
            cfg["training"]["beta_start"], cfg["training"]["beta_end"],
        )

        # === Train ===
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]")
        for video_ids, all_captions, queries in pbar:
            # 编码视频伪查询
            with torch.no_grad():
                video_feats, video_mask = encode_video_captions(
                    encoder, all_captions, device
                )
                query_feats, query_mask = encoder.encode_tokens(queries, device=device)

            with autocast(amp_device, enabled=use_amp):
                scores, mu, sigma_sq = model(
                    video_feats, video_mask, query_feats, query_mask
                )
                loss_total, loss_task, loss_kl = model.compute_loss(
                    scores, mu, sigma_sq, beta=beta
                )

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss_total.item()
            train_steps += 1
            pbar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                task=f"{loss_task.item():.4f}",
                kl=f"{loss_kl.item():.4f}",
                beta=f"{beta:.6f}",
                tau=f"{model.temperature.item():.4f}",
            )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # === Validation ===
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for video_ids, all_captions, queries in tqdm(val_loader, desc="[Val]"):
                video_feats, video_mask = encode_video_captions(
                    encoder, all_captions, device
                )
                query_feats, query_mask = encoder.encode_tokens(queries, device=device)

                with autocast(amp_device, enabled=use_amp):
                    scores, mu, sigma_sq = model(
                        video_feats, video_mask, query_feats, query_mask
                    )
                    loss_total, _, _ = model.compute_loss(scores, mu, sigma_sq, beta=beta)

                val_loss_sum += loss_total.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, beta={beta:.6f}")
        logger.info(
            f"EPOCH_SUMMARY | epoch={epoch+1} | train_loss={avg_train_loss:.6f} "
            f"| val_loss={avg_val_loss:.6f} | beta={beta:.6f} "
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
            },
            save_dir / "latest_model.pt",
        )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
