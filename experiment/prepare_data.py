"""
数据准备脚本：将 MSRVTT_narration.json + MSR_VTT.json 转换为
OmniColPress 框架所需的格式，用于 AGC 伪查询实验。

输出目录：experiment/data/
生成文件：
  - train_corpus.jsonl     (训练集语料，每行: {"docid", "text"})
  - test_corpus.jsonl      (测试集语料)
  - train.jsonl            (训练数据，每行: {"query_id", "query", "positive_document_ids", "negative_document_ids"})
  - test_queries.csv       (测试查询，每行: query_id, query)
  - test_qrels.jsonl       (相关性标注，每行: {"query_id", "document_id", "relevance"})
  - stats.json             (数据统计信息)

MSR-VTT 标准划分:
  - Train: video0 - video6512  (6513 videos)
  - Val:   video6513 - video7009 (497 videos)
  - Test:  video7010 - video9999 (2990 videos)

MSR-VTT 1k-A Test Split:
  - 1000 个 query-video 对 (每 test video 只选第一个 caption 作为 query)
  - 候选库: 全部 1000 个 test video
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict


# ─── 路径配置 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NARRATION_PATH = PROJECT_ROOT / "MSRVTT_narration.json"
MSRVTT_PATH = PROJECT_ROOT / "MSR_VTT.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

# ─── 数据划分 ───────────────────────────────────────────────
TRAIN_END = 6513       # video0 ~ video6512
VAL_END = 7010         # video6513 ~ video7009
TOTAL = 10000          # video7010 ~ video9999
TEST_1K_SIZE = 1000    # MSR-VTT 1k-A test split


def load_narrations(path: Path) -> dict:
    """加载伪查询 narration → {video_id: [caption_1, ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    narrations = {}
    for item in data:
        vid = item["video_file"]
        captions = []
        idx = 1
        while f"caption_{idx}" in item:
            captions.append(item[f"caption_{idx}"])
            idx += 1
        narrations[vid] = captions
    return narrations


def load_gt_annotations(path: Path) -> dict:
    """加载 GT 标注 → {video_id: [caption, ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann["caption"])
    return dict(gt)


def build_corpus(narrations: dict, video_ids: list) -> list:
    """
    为指定 video_ids 构建语料。
    每个视频的所有伪查询拼接为一个文本文档。
    """
    corpus = []
    for vid in video_ids:
        if vid not in narrations:
            print(f"Warning: {vid} not found in narrations, skipping")
            continue
        captions = narrations[vid]
        # 用句号分隔拼接所有 caption
        text = " ".join(cap.strip().rstrip(".") + "." for cap in captions)
        corpus.append({"docid": vid, "text": text})
    return corpus


def build_train_data(gt: dict, train_ids: list) -> list:
    """
    构建训练数据：每个 GT caption 作为一个独立的 query。
    OmniColPress 训练格式：
    {"query_id", "query", "positive_document_ids", "negative_document_ids"}
    """
    train_data = []
    qid = 0
    for vid in train_ids:
        if vid not in gt:
            continue
        for cap in gt[vid]:
            train_data.append({
                "query_id": f"train_{qid}",
                "query": cap,
                "positive_document_ids": [vid],
                "negative_document_ids": [],
            })
            qid += 1
    return train_data


def build_test_data(gt: dict, test_ids: list, num_queries: int = 1000):
    """
    构建测试数据 (MSR-VTT 1k-A test split)。
    从 test_ids 中选前 num_queries 个视频，每视频取第一个 caption 作为 query。

    返回: (queries_csv_rows, qrels)
      - queries_csv_rows: [{"query_id": ..., "query": ...}]
      - qrels: [{"query_id": ..., "document_id": ..., "relevance": 1}]
    """
    queries = []
    qrels = []
    count = 0
    for vid in test_ids:
        if count >= num_queries:
            break
        if vid not in gt or len(gt[vid]) == 0:
            continue
        qid = f"test_{count}"
        queries.append({"query_id": qid, "query": gt[vid][0]})
        qrels.append({
            "query_id": qid,
            "document_id": vid,
            "relevance": 1,
        })
        count += 1
    return queries, qrels


def write_jsonl(data: list, path: Path):
    """写入 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(data: list, path: Path, fieldnames: list):
    """写入 CSV 文件"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def compute_stats(narrations, gt, train_corpus, test_corpus, train_data, queries, qrels):
    """计算并返回数据统计"""
    # 统计伪查询拼接后的 token 长度（近似用空格分词）
    all_corpus = train_corpus + test_corpus
    word_counts = [len(item["text"].split()) for item in all_corpus]
    char_counts = [len(item["text"]) for item in all_corpus]

    return {
        "narration_videos": len(narrations),
        "gt_videos": len(gt),
        "train_corpus_size": len(train_corpus),
        "test_corpus_size": len(test_corpus),
        "train_queries": len(train_data),
        "test_queries": len(queries),
        "test_qrels": len(qrels),
        "corpus_text_length": {
            "word_count": {
                "min": min(word_counts),
                "max": max(word_counts),
                "mean": sum(word_counts) / len(word_counts),
                "p50": sorted(word_counts)[len(word_counts) // 2],
                "p95": sorted(word_counts)[int(len(word_counts) * 0.95)],
            },
            "char_count": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": sum(char_counts) / len(char_counts),
            },
        },
    }


def main():
    print("=" * 60)
    print("AGC 伪查询实验 — 数据准备")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n[1/6] 加载 MSRVTT_narration.json ...")
    narrations = load_narrations(NARRATION_PATH)
    print(f"  → {len(narrations)} videos loaded")

    print("[2/6] 加载 MSR_VTT.json ...")
    gt = load_gt_annotations(MSRVTT_PATH)
    print(f"  → {len(gt)} videos, {sum(len(v) for v in gt.values())} annotations")

    # 2. 划分 video IDs
    train_ids = [f"video{i}" for i in range(0, TRAIN_END)]
    val_ids = [f"video{i}" for i in range(TRAIN_END, VAL_END)]
    test_ids = [f"video{i}" for i in range(VAL_END, TOTAL)]
    print(f"\n  Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # 3. 构建语料
    print("\n[3/6] 构建 corpus (伪查询拼接) ...")
    # 训练语料: train + val (与论文 Train 9k split 对齐)
    train_corpus = build_corpus(narrations, train_ids + val_ids)
    test_corpus = build_corpus(narrations, test_ids)
    print(f"  → train corpus: {len(train_corpus)} docs")
    print(f"  → test corpus:  {len(test_corpus)} docs")

    # 4. 构建训练数据
    print("[4/6] 构建训练数据 (query-document pairs) ...")
    train_data = build_train_data(gt, train_ids + val_ids)
    print(f"  → {len(train_data)} training pairs")

    # 5. 构建测试数据 (1k-A split)
    print("[5/6] 构建测试数据 (MSR-VTT 1k-A) ...")
    queries, qrels = build_test_data(gt, test_ids, TEST_1K_SIZE)
    print(f"  → {len(queries)} test queries, {len(qrels)} qrels")

    # 6. 写入文件
    print("\n[6/6] 写入文件 ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_corpus, OUTPUT_DIR / "train_corpus.jsonl")
    print(f"  ✓ {OUTPUT_DIR / 'train_corpus.jsonl'}")

    write_jsonl(test_corpus, OUTPUT_DIR / "test_corpus.jsonl")
    print(f"  ✓ {OUTPUT_DIR / 'test_corpus.jsonl'}")

    write_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    print(f"  ✓ {OUTPUT_DIR / 'train.jsonl'}")

    write_csv(queries, OUTPUT_DIR / "test_queries.csv", ["query_id", "query"])
    print(f"  ✓ {OUTPUT_DIR / 'test_queries.csv'}")

    write_jsonl(qrels, OUTPUT_DIR / "test_qrels.jsonl")
    print(f"  ✓ {OUTPUT_DIR / 'test_qrels.jsonl'}")

    # 7. 统计信息
    stats = compute_stats(narrations, gt, train_corpus, test_corpus, train_data, queries, qrels)
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {OUTPUT_DIR / 'stats.json'}")

    print("\n" + "=" * 60)
    print("数据统计:")
    print(f"  训练语料: {stats['train_corpus_size']} docs")
    print(f"  测试语料: {stats['test_corpus_size']} docs")
    print(f"  训练 queries: {stats['train_queries']}")
    print(f"  测试 queries: {stats['test_queries']}")
    wc = stats["corpus_text_length"]["word_count"]
    print(f"  文档词数: min={wc['min']}, max={wc['max']}, mean={wc['mean']:.0f}, p50={wc['p50']}, p95={wc['p95']}")
    print(f"\n  → 建议 passage_max_len: {max(512, ((wc['p95'] * 4 // 3) // 128 + 1) * 128)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
