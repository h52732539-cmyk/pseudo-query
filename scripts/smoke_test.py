"""
快速验证脚本：用少量数据测试整个 pipeline 的正确性。
python scripts/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import numpy as np
from pathlib import Path

from data.preprocess import load_narrations, load_gt_annotations, get_split_video_ids, build_retrieval_pairs
from data.dataset import PseudoQueryTrainDataset, train_collate_fn
from models.clip_encoder import CLIPTextEncoder
from models.prototype import build_prototypes_kmeans, PrototypeLibrary
from models.variational_encoder import VariationalCompressor
from models.query_assembly import QueryAssembly
from models.scoring import uncertainty_aware_score, info_nce_loss, kl_divergence
from models.pipeline import VIBPseudoQueryModel
from train import encode_video_captions


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

    train_pairs = build_retrieval_pairs(train_ids[:10], gt)  # 只取10个视频
    print(f"Sample train pairs: {len(train_pairs)}")

    # ===== Step 2: 编码器 =====
    print("\n===== Step 2: CLIP 编码器 =====")
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    # 测试编码
    test_texts = ["a man is driving a car", "a woman is singing"]
    tok_feats, attn_mask = encoder.encode_tokens(test_texts, device=device)
    print(f"Token features shape: {tok_feats.shape}")  # (2, L, 512)
    print(f"Attention mask shape: {attn_mask.shape}")

    sent_feats = encoder.encode_sentence(test_texts, device=device)
    print(f"Sentence features shape: {sent_feats.shape}")  # (2, 512)

    # ===== Step 3: 小规模 K-Means 原型构建 =====
    print("\n===== Step 3: 小规模原型构建 =====")
    # 取少量 caption 做测试
    sample_captions = []
    for vid in list(narrations.keys())[:20]:
        sample_captions.extend(narrations[vid])
    print(f"Sample captions: {len(sample_captions)}")

    all_tokens_np = []
    bs = 64
    for i in range(0, len(sample_captions), bs):
        batch = sample_captions[i:i+bs]
        feats, mask = encoder.encode_tokens(batch, device=device)
        for j in range(feats.shape[0]):
            vl = mask[j].sum().item()
            all_tokens_np.append(feats[j, :vl].cpu().numpy())

    global_pool = np.concatenate(all_tokens_np, axis=0)
    print(f"Token pool shape: {global_pool.shape}")

    K_test = 32  # 小 K 用于测试
    centroids = build_prototypes_kmeans(
        global_pool, num_prototypes=K_test, batch_size=5000, max_iter=10
    )
    print(f"Centroids shape: {centroids.shape}")

    # ===== Step 4: 模型前向传播 =====
    print("\n===== Step 4: 模型前向传播 =====")
    model = VIBPseudoQueryModel(
        feature_dim=cfg["encoder"]["feature_dim"],
        num_prototypes=K_test,
        aggregation=cfg["model"]["aggregation"],
        temperature_init=cfg["model"]["temperature_init"],
        variance_penalty=cfg["model"]["variance_penalty"],
        prototype_init_weights=torch.from_numpy(centroids),
    ).to(device)

    # 构造 mini-batch
    dataset = PseudoQueryTrainDataset(train_pairs[:8], narrations)
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    video_ids, all_captions, queries = train_collate_fn(batch)

    with torch.no_grad():
        video_feats, video_mask = encode_video_captions(encoder, all_captions, device)
        query_feats, query_mask = encoder.encode_tokens(queries, device=device)

    print(f"Video features: {video_feats.shape}, mask: {video_mask.shape}")
    print(f"Query features: {query_feats.shape}, mask: {query_mask.shape}")

    # 前向
    model.train()
    scores, mu, sigma_sq = model(video_feats, video_mask, query_feats, query_mask)
    print(f"Scores: {scores.shape}")
    print(f"Mu: {mu.shape}, Sigma_sq: {sigma_sq.shape}")
    print(f"Temperature: {model.temperature.item():.4f}")

    # Loss
    loss_total, loss_task, loss_kl = model.compute_loss(scores, mu, sigma_sq, beta=1e-3)
    print(f"Loss total: {loss_total.item():.4f}, task: {loss_task.item():.4f}, kl: {loss_kl.item():.4f}")

    # 反向传播测试
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

    # ===== Step 5: 评估流程测试 =====
    print("\n===== Step 5: 评估流程测试 =====")
    model.eval()
    with torch.no_grad():
        # Encode a few videos
        test_vids = test_ids[:8]
        caps_list = [narrations[v] for v in test_vids]
        v_feats, v_mask = encode_video_captions(encoder, caps_list, device)
        mu_v, sq_v = model.encode_video(v_feats, v_mask)

        # Encode queries
        test_queries = [gt[v][0] for v in test_vids if v in gt][:8]
        q_feats, q_mask = encoder.encode_tokens(test_queries, device=device)
        s_T = model.encode_query(q_feats, q_mask)

        # Score
        scores = uncertainty_aware_score(s_T, mu_v, sq_v, cfg["model"]["variance_penalty"])
        print(f"Eval scores shape: {scores.shape}")
        print(f"Diagonal scores (should be highest in row): {torch.diag(scores)[:4]}")

    print("\n===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    main()
