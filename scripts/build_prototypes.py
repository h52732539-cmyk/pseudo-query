"""
离线脚本：提取训练集视频伪查询的 sentence-level 特征 → GMM+BIC / K-Means 聚类 → 保存原型。

用法:
    python scripts/build_prototypes.py [--config configs/default.yaml]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.preprocess import load_narrations, get_split_video_ids
from models.clip_encoder import CLIPTextEncoder
from models.prototype import build_prototypes_kmeans, build_prototypes_gmm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="编码时的 batch size")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 narrations
    narrations = load_narrations(cfg["data"]["narration_json"])
    print(f"Loaded narrations for {len(narrations)} videos.")

    # 2. 仅使用训练集视频（避免信息泄露）
    train_ids, _, _ = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )
    print(f"Using {len(train_ids)} training videos for prototype construction.")

    # 3. 初始化编码器
    encoder = CLIPTextEncoder(
        model_name=cfg["encoder"]["name"],
        max_length=cfg["encoder"]["max_token_length"],
    ).to(device)
    encoder.eval()

    # 4. 收集训练集所有 caption 文本
    all_captions = []
    for vid in sorted(train_ids, key=lambda x: int(x.replace("video", ""))):
        if vid in narrations:
            all_captions.extend(narrations[vid])
    print(f"Total captions to encode: {len(all_captions)}")

    # 5. 分批编码，提取 sentence-level 特征（而非 token-level）
    all_features = []
    bs = args.batch_size
    for i in tqdm(range(0, len(all_captions), bs), desc="Encoding captions (sentence-level)"):
        batch_texts = all_captions[i : i + bs]
        sent_feats = encoder.encode_sentence(batch_texts, device=device)  # (bs, d)
        all_features.append(sent_feats.cpu().numpy())

    # 6. 拼接全局特征池
    global_pool = np.concatenate(all_features, axis=0)  # (N_total, d)
    print(f"Global sentence feature pool shape: {global_pool.shape}")

    # 7. 聚类
    proto_cfg = cfg["prototype"]
    method = proto_cfg.get("method", "gmm")

    if method == "gmm":
        candidate_ks = proto_cfg.get("candidate_ks", [32, 64, 128, 256, 512])
        centroids, best_k = build_prototypes_gmm(
            global_pool,
            candidate_ks=candidate_ks,
        )
        meta = {"method": "gmm", "K": best_k, "split": "train"}
    else:
        num_prototypes = proto_cfg.get("num_prototypes", 512)
        centroids = build_prototypes_kmeans(
            global_pool,
            num_prototypes=num_prototypes,
            batch_size=proto_cfg.get("kmeans_batch_size", 100000),
            max_iter=proto_cfg.get("kmeans_max_iter", 100),
        )
        meta = {"method": "kmeans", "K": num_prototypes, "split": "train"}

    # 8. 保存（包含元信息）
    save_path = Path(cfg["data"]["prototype_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "prototypes": torch.from_numpy(centroids),
        **meta,
    }
    torch.save(save_data, save_path)
    print(f"Prototypes saved to {save_path}  (shape: {centroids.shape}, meta: {meta})")


if __name__ == "__main__":
    main()
