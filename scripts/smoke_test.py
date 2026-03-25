"""
快速验证脚本：用少量数据测试 SwAV / Hybrid Pipeline 的正确性。
python scripts/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from data.dataset import PseudoQueryMultiViewDataset, multiview_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.scoring import cosine_retrieval_score
from models.pipeline_swav import SwAVPipelineModel
from models.pipeline_hybrid import HybridPipelineModel
from train import encode_video_captions


def test_pipeline(model, encoder, dataset, device, scheme_name):
    """测试单个 pipeline 的完整前向/反向/评估流程。"""
    print(f"\n{'='*50}")
    print(f"Testing {scheme_name} Pipeline")
    print(f"{'='*50}")

    # 构造 mini-batch
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    video_ids, caps_v1, caps_v2, queries = multiview_collate_fn(batch)

    with torch.no_grad():
        v1_feats, v1_mask = encode_video_captions(encoder, caps_v1, device)
        v2_feats, v2_mask = encode_video_captions(encoder, caps_v2, device)
        q_feats, q_mask = encoder.encode_tokens(queries, device=device)

    print(f"View1 features: {v1_feats.shape}, mask: {v1_mask.shape}")
    print(f"View2 features: {v2_feats.shape}, mask: {v2_mask.shape}")
    print(f"Query features: {q_feats.shape}, mask: {q_mask.shape}")

    # 前向
    model.train()
    h_avg, mu1, mu2, s_T, q_tilde = model(
        v1_feats, v1_mask, v2_feats, v2_mask, q_feats, q_mask
    )
    print(f"h_avg: {h_avg.shape}")
    print(f"mu1: {mu1.shape}, mu2: {mu2.shape}")
    print(f"s_T: {s_T.shape}")
    print(f"q_tilde: {q_tilde.shape}")
    print(f"Temperature: {model.temperature.item():.4f}")

    # Loss
    loss_total, loss_match, loss_swav = model.compute_loss(
        h_avg, mu1, mu2, s_T, q_tilde, swav_alpha=0.5
    )
    print(f"Loss total: {loss_total.item():.4f}, match: {loss_match.item():.4f}, swav: {loss_swav.item():.4f}")

    # 反向传播
    loss_total.backward()
    print("Backward pass: OK")

    # 检查梯度
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    print(f"Parameters with gradients: {len(grad_norms)}")
    for name, gn in list(grad_norms.items())[:5]:
        print(f"  {name}: grad_norm={gn:.6f}")

    # Hybrid 特有: post_step
    if scheme_name == "Hybrid":
        n_dead = model.post_step(s_T.detach(), v1_feats.detach())
        print(f"EMA post_step: {n_dead} dead prototypes reinitialized")

    # 评估流程
    print(f"\n--- Eval flow ({scheme_name}) ---")
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        mu_v = model.get_video_repr(v1_feats, v1_mask)
        s_T_q = model.get_query_repr(q_feats, q_mask)
        scores = cosine_retrieval_score(s_T_q, mu_v, temperature=model.temperature.item())
        print(f"Eval scores shape: {scores.shape}")
        diag = torch.diag(scores)
        print(f"Diagonal scores: {diag}")

    print(f"\n{scheme_name} Pipeline: ALL TESTS PASSED")


def main():
    config_path = "configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ===== Step 1: 数据加载 =====
    print("\n===== Step 1: 数据加载 =====")
    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, test_ids = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )
    print(f"Narrations: {len(narrations)} videos")
    print(f"GT: {len(gt)} videos")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_pairs = build_retrieval_pairs(train_ids[:10], gt)
    print(f"Sample train pairs: {len(train_pairs)}")

    # ===== Step 2: 编码器 =====
    print("\n===== Step 2: CLIP 编码器 =====")
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    test_texts = ["a man is driving a car", "a woman is singing"]
    tok_feats, attn_mask = encoder.encode_tokens(test_texts, device=device)
    print(f"Token features shape: {tok_feats.shape}")
    sent_feats = encoder.encode_sentence(test_texts, device=device)
    print(f"Sentence features shape: {sent_feats.shape}")

    # ===== Step 3: 构造 Multi-View Dataset =====
    print("\n===== Step 3: Multi-View Dataset =====")
    dataset = PseudoQueryMultiViewDataset(train_pairs, narrations)
    sample = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: video_id={sample[0]}, view1 caps={len(sample[1])}, view2 caps={len(sample[2])}, query={sample[3][:50]}...")

    # ===== Step 4: SwAV Pipeline =====
    K_test = cfg["prototype"]["num_prototypes"]
    swav_cfg = cfg.get("swav", {})

    swav_model = SwAVPipelineModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_prototypes=K_test,
        aggregation=cfg["model"]["aggregation"],
        temperature_init=cfg["model"]["temperature_init"],
        sinkhorn_eps=swav_cfg.get("sinkhorn_eps", 0.05),
        sinkhorn_iters=swav_cfg.get("sinkhorn_iters", 3),
        swav_temperature=swav_cfg.get("temperature", 0.1),
    ).to(device)

    test_pipeline(swav_model, encoder, dataset, device, "SwAV")

    # ===== Step 5: Hybrid Pipeline =====
    ema_cfg = cfg.get("ema", {})
    hybrid_model = HybridPipelineModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_prototypes=K_test,
        aggregation=cfg["model"]["aggregation"],
        temperature_init=cfg["model"]["temperature_init"],
        sinkhorn_eps=swav_cfg.get("sinkhorn_eps", 0.05),
        sinkhorn_iters=swav_cfg.get("sinkhorn_iters", 3),
        swav_temperature=swav_cfg.get("temperature", 0.1),
        ema_decay=ema_cfg.get("decay", 0.999),
        dead_proto_threshold=ema_cfg.get("dead_proto_threshold", 100),
    ).to(device)

    test_pipeline(hybrid_model, encoder, dataset, device, "Hybrid")

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
