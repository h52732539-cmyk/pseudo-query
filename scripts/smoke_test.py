"""
快速验证脚本：用少量伪数据测试新 Pipeline（核过滤 + Adapter + Reranker）的正确性。
python scripts/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

from models.clip_encoder import CLIPTextEncoder
from models.nucleus_filter import NucleusFilter
from models.pipeline_pq import PseudoQueryPipeline
from models.prototype_builder import sinkhorn_cluster, build_inverted_index
from models.scoring import symmetric_infonce, coarse_prototype_score


def main():
    config_path = "configs/default_pq.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    B = 4  # mini-batch size

    # ===== Step 1: CLIP 编码器 =====
    print("\n===== Step 1: CLIP 编码器 =====")
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    queries = ["a man is driving a car", "a woman is singing",
               "people are playing basketball", "a cat is sleeping"]
    narr_captions = [
        ["man drives vehicle on highway", "car moving on road"],
        ["woman sings on stage", "singer performing live"],
        ["basketball game in progress", "players running on court"],
        ["cat sleeping on couch", "kitten resting peacefully"],
    ]

    with torch.no_grad():
        q_sent = encoder.encode_sentence(queries, device=device)  # (B, d)
        q_tok, q_mask = encoder.encode_tokens(queries, device=device)  # (B, L_q, d)
        print(f"  q_sent: {q_sent.shape}, q_tok: {q_tok.shape}")

    # 编码 narrations
    narr_sents = []
    narr_toks = []
    narr_tok_masks = []
    with torch.no_grad():
        for caps in narr_captions:
            s = encoder.encode_sentence(caps, device=device)  # (N_i, d)
            t, m = encoder.encode_tokens(caps, device=device)
            narr_sents.append(s)
            narr_toks.append(t)
            narr_tok_masks.append(m)

    # Pad narration sentences to batch
    N_max = max(s.shape[0] for s in narr_sents)
    d = narr_sents[0].shape[-1]
    narr_sent_padded = torch.zeros(B, N_max, d, device=device)
    narr_mask = torch.zeros(B, N_max, device=device)
    for i, s in enumerate(narr_sents):
        n = s.shape[0]
        narr_sent_padded[i, :n] = s
        narr_mask[i, :n] = 1.0

    # Fake frame features
    num_frames = 12
    frame_feats = torch.randn(B, num_frames, d, device=device)
    frame_feats = F.normalize(frame_feats, p=2, dim=-1)
    print(f"  frame_feats: {frame_feats.shape}")

    # ===== Step 2: NucleusFilter =====
    print("\n===== Step 2: NucleusFilter =====")
    nf_cfg = cfg["nucleus_filter"]
    nucleus_filter = NucleusFilter(
        feature_dim=d,
        num_heads=nf_cfg["num_heads"],
        temporal_layers=nf_cfg["temporal_layers"],
        temporal_ffn=nf_cfg["temporal_ffn"],
        max_seq_len=nf_cfg["max_seq_len"],
    ).to(device)

    # 增强
    enhanced_v, enhanced_n = nucleus_filter.enhance_features(
        frame_feats, narr_sent_padded, narr_mask
    )
    print(f"  enhanced_v: {enhanced_v.shape}, enhanced_n: {enhanced_n.shape}")

    # 计算 filter weights (训练软权重)
    weights = nucleus_filter.compute_filter_weights(q_sent, enhanced_n, narr_mask)
    print(f"  filter weights: {weights.shape}, sum per sample: {weights.sum(dim=-1)}")

    # 核截断 (推理硬截断)
    sel_indices, sel_weights = nucleus_filter.nucleus_select(
        weights, threshold_p=nf_cfg["inference_threshold"]
    )
    for i, (idx, w) in enumerate(zip(sel_indices, sel_weights)):
        print(f"  Sample {i}: selected {len(idx)} narrations, weights sum={w.sum().item():.4f}")

    # Filtered centroid (训练用 soft weighted)
    filtered_centroid = (weights.unsqueeze(-1) * enhanced_n).sum(dim=1)  # (B, d)
    filtered_centroid = F.normalize(filtered_centroid, p=2, dim=-1)
    print(f"  filtered_centroid: {filtered_centroid.shape}")

    # ===== Step 3: PseudoQueryPipeline =====
    print("\n===== Step 3: PseudoQueryPipeline =====")
    model = PseudoQueryPipeline(
        feature_dim=d,
        adapter_hidden_mult=cfg["model"]["adapter_hidden_mult"],
        reranker_num_heads=cfg["model"]["reranker_num_heads"],
        temperature_init=cfg["model"]["temperature_init"],
        fine_loss_weight=cfg["model"]["fine_loss_weight"],
    ).to(device)

    # Pad narration tokens for reranker
    L_n_max = max(t.shape[1] for t in narr_toks)
    N_tok_max = max(t.shape[0] for t in narr_toks)
    narr_tok_padded = torch.zeros(B, N_tok_max * L_n_max, d, device=device)
    narr_tok_mask_padded = torch.zeros(B, N_tok_max * L_n_max, device=device)
    for i, (t, m) in enumerate(zip(narr_toks, narr_tok_masks)):
        flat_len = t.shape[0] * t.shape[1]
        narr_tok_padded[i, :flat_len] = t.reshape(-1, d)
        narr_tok_mask_padded[i, :flat_len] = m.reshape(-1)

    # 训练前向
    model.train()
    adapted_query, fine_score_matrix = model(
        q_sent, q_tok, q_mask, narr_tok_padded, narr_tok_mask_padded
    )
    print(f"  adapted_query: {adapted_query.shape}")
    print(f"  fine_score_matrix: {fine_score_matrix.shape}")
    print(f"  temperature: {model.temperature.item():.4f}")

    # 计算 loss
    loss_total, loss_coarse, loss_fine = model.compute_loss(
        adapted_query, filtered_centroid, fine_score_matrix
    )
    print(f"  loss_total: {loss_total.item():.4f}, coarse: {loss_coarse.item():.4f}, fine: {loss_fine.item():.4f}")

    # 反向传播
    loss_total.backward()
    print("  Backward: OK")

    grad_count = 0
    for name, p in list(model.named_parameters()) + list(nucleus_filter.named_parameters()):
        if p.grad is not None:
            grad_count += 1
    print(f"  Parameters with gradients: {grad_count}")

    # ===== Step 4: Sinkhorn 聚类 =====
    print("\n===== Step 4: Sinkhorn-Knopp 聚类 =====")
    # 模拟推理：用 enhanced_n 的选中子集做聚类
    fake_embs = F.normalize(torch.randn(50, d, device=device), p=2, dim=-1)
    K = 8
    prototypes, assignments = sinkhorn_cluster(fake_embs, K, eps=0.05, niters=3, n_rounds=5)
    print(f"  prototypes: {prototypes.shape}, assignments: {assignments.shape}")
    print(f"  assignment sum per sample (should ~1): {assignments.sum(dim=1)[:5]}")

    # 倒排索引
    metadata = [(f"video{i // 5}", i % 5) for i in range(50)]
    inv_idx, vid_narr_idx = build_inverted_index(assignments, metadata)
    print(f"  倒排索引: {len(inv_idx)} clusters, {len(vid_narr_idx)} videos")

    # ===== Step 5: 推理评分 =====
    print("\n===== Step 5: 推理评分 =====")
    model.eval()
    with torch.no_grad():
        adapted = model.adapt_query(q_sent)
        print(f"  adapted query: {adapted.shape}")
        proto_scores = coarse_prototype_score(adapted, prototypes)
        print(f"  coarse prototype scores: {proto_scores.shape}")

        # 精排
        single_score = model.rerank(
            q_tok[:1], q_mask[:1],
            narr_tok_padded[:1], narr_tok_mask_padded[:1]
        )
        print(f"  fine rerank score: {single_score.item():.4f}")

    # ===== Step 6: symmetric_infonce =====
    print("\n===== Step 6: symmetric_infonce =====")
    fake_scores = torch.randn(B, B)
    loss = symmetric_infonce(fake_scores)
    print(f"  symmetric_infonce loss: {loss.item():.4f}")

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
