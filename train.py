"""
训练主循环：跨模态核过滤 + 查询-伪查询映射学习（Adapter + Reranker 联合训练）。

用法:
    python train.py [--config configs/default_pq.yaml]
"""
import argparse
import os
import math
import logging
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from data.dataset import QueryNarrationDataset, query_narration_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.nucleus_filter import NucleusFilter
from models.pipeline_pq import PseudoQueryPipeline


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


def encode_narr_sentences(encoder, captions_list, device, max_narrs=64):
    """
    编码一个 batch 中所有视频的 narration sentence embeddings。
    Returns:
        narr_sent: (B, N_max, d) — padded
        narr_mask: (B, N_max) — 1=valid, 0=pad
    """
    batch_embs = []
    batch_lens = []

    for captions in captions_list:
        caps = captions[:max_narrs]
        embs = encoder.encode_sentence(caps, device=device)  # (N_i, d)
        batch_embs.append(embs)
        batch_lens.append(embs.shape[0])

    N_max = max(batch_lens)
    d = batch_embs[0].shape[-1]
    B = len(batch_embs)

    padded = torch.zeros(B, N_max, d, device=device)
    mask = torch.zeros(B, N_max, device=device)
    for i, emb in enumerate(batch_embs):
        n = emb.shape[0]
        padded[i, :n] = emb
        mask[i, :n] = 1.0

    return padded, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_pq.yaml")
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

    logger.info(f"Config: {args.config}, Scheme: pq")
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

    frame_feat_dir = cfg["data"].get("frame_feat_dir", None)
    train_dataset = QueryNarrationDataset(train_pairs, narrations, frame_feat_dir)
    val_dataset = QueryNarrationDataset(val_pairs, narrations, frame_feat_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=query_narration_collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=query_narration_collate_fn,
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

    # ---- 核过滤模块 ----
    nf_cfg = cfg.get("nucleus_filter", {})
    nucleus_filter = NucleusFilter(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_heads=nf_cfg.get("num_heads", 8),
        temporal_layers=nf_cfg.get("temporal_layers", 4),
        temporal_ffn=nf_cfg.get("temporal_ffn", 2048),
        max_seq_len=nf_cfg.get("max_seq_len", 128),
    ).to(device)

    # ---- Pipeline 模型 ----
    model_cfg = cfg.get("model", {})
    model = PseudoQueryPipeline(
        feature_dim=cfg["encoder"]["feature_dim"],
        adapter_hidden_mult=model_cfg.get("adapter_hidden_mult", 4),
        reranker_num_heads=model_cfg.get("reranker_num_heads", 8),
        temperature_init=model_cfg.get("temperature_init", 0.07),
        fine_loss_weight=model_cfg.get("fine_loss_weight", 0.5),
    ).to(device)

    logger.info(f"NucleusFilter params: {sum(p.numel() for p in nucleus_filter.parameters() if p.requires_grad):,}")
    logger.info(f"Pipeline params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "nucleus_filter_state_dict" in ckpt:
            nucleus_filter.load_state_dict(ckpt["nucleus_filter_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # ---- 优化器 & 调度器 ----
    all_params = list(model.parameters()) + list(nucleus_filter.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
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

    max_narr_tokens = cfg["training"].get("max_narr_tokens", 512)

    # ---- 训练循环 ----
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        # === Train ===
        model.train()
        nucleus_filter.train()
        train_loss_sum = 0.0
        train_coarse_sum = 0.0
        train_fine_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]")
        for video_ids, queries, narr_texts, frame_features in pbar:
            # CLIP 编码（冻结，无梯度）
            with torch.no_grad():
                q_sent = encoder.encode_sentence(queries, device=device)  # (B, d)
                q_tokens, q_mask = encoder.encode_tokens(queries, device=device)  # (B, L, d), (B, L)
                narr_sent, narr_mask = encode_narr_sentences(encoder, narr_texts, device)  # (B, N, d), (B, N)
                narr_tokens, narr_tok_mask = encode_video_captions(encoder, narr_texts, device, max_narr_tokens)

            # 帧特征
            if frame_features is not None:
                frame_features = frame_features.to(device)
            else:
                # fallback: 随机帧特征（开发测试用）
                B = q_sent.shape[0]
                frame_features = F.normalize(torch.randn(B, 12, q_sent.shape[-1], device=device), p=2, dim=-1)

            with autocast(amp_device, enabled=use_amp):
                # 核过滤（可训练，有梯度）
                _, enhanced_n = nucleus_filter.enhance_features(frame_features, narr_sent, narr_mask)
                soft_weights = nucleus_filter.compute_filter_weights(q_sent, enhanced_n, narr_mask)  # (B, N)

                # 过滤后 centroid = 加权聚合 原始 CLIP narration embeddings
                # narr_sent: (B, N, d), soft_weights: (B, N)
                filtered_centroid = (soft_weights.unsqueeze(-1) * narr_sent).sum(dim=1)  # (B, d)
                filtered_centroid = F.normalize(filtered_centroid, p=2, dim=-1)

                # Pipeline 前向（纯文本空间）
                adapted_query, fine_score_matrix = model(
                    q_sent, q_tokens, q_mask, narr_tokens, narr_tok_mask
                )

                # 损失计算
                loss_total, loss_coarse, loss_fine = model.compute_loss(
                    adapted_query, filtered_centroid, fine_score_matrix
                )

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, cfg["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss_total.item()
            train_coarse_sum += loss_coarse.item()
            train_fine_sum += loss_fine.item()
            train_steps += 1
            pbar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                coarse=f"{loss_coarse.item():.4f}",
                fine=f"{loss_fine.item():.4f}",
                tau=f"{model.temperature.item():.4f}",
            )

        avg_train_loss = train_loss_sum / max(train_steps, 1)
        avg_coarse = train_coarse_sum / max(train_steps, 1)
        avg_fine = train_fine_sum / max(train_steps, 1)

        # === Validation ===
        model.eval()
        nucleus_filter.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for video_ids, queries, narr_texts, frame_features in tqdm(val_loader, desc="[Val]"):
                q_sent = encoder.encode_sentence(queries, device=device)
                q_tokens, q_mask = encoder.encode_tokens(queries, device=device)
                narr_sent, narr_mask = encode_narr_sentences(encoder, narr_texts, device)
                narr_tokens, narr_tok_mask = encode_video_captions(encoder, narr_texts, device, max_narr_tokens)

                if frame_features is not None:
                    frame_features = frame_features.to(device)
                else:
                    B = q_sent.shape[0]
                    frame_features = F.normalize(torch.randn(B, 12, q_sent.shape[-1], device=device), p=2, dim=-1)

                with autocast(amp_device, enabled=use_amp):
                    _, enhanced_n = nucleus_filter.enhance_features(frame_features, narr_sent, narr_mask)
                    soft_weights = nucleus_filter.compute_filter_weights(q_sent, enhanced_n, narr_mask)
                    filtered_centroid = F.normalize(
                        (soft_weights.unsqueeze(-1) * narr_sent).sum(dim=1), p=2, dim=-1
                    )

                    adapted_query, fine_score_matrix = model(
                        q_sent, q_tokens, q_mask, narr_tokens, narr_tok_mask
                    )
                    loss_total, _, _ = model.compute_loss(
                        adapted_query, filtered_centroid, fine_score_matrix
                    )

                val_loss_sum += loss_total.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} "
            f"(coarse={avg_coarse:.4f}, fine={avg_fine:.4f}), "
            f"val_loss={avg_val_loss:.4f}"
        )
        logger.info(
            f"EPOCH_SUMMARY | epoch={epoch+1} | train_loss={avg_train_loss:.6f} "
            f"| val_loss={avg_val_loss:.6f} "
            f"| tau={model.temperature.item():.6f} | lr={current_lr:.8f}"
        )

        # Save best
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "nucleus_filter_state_dict": nucleus_filter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "scheme": "pq",
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, save_dir / "best_model.pt")
            logger.info(f"  ✓ Best model saved (val_loss={avg_val_loss:.4f})")

        # Save latest
        torch.save(ckpt_data, save_dir / "latest_model.pt")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
