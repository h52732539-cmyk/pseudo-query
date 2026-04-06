"""
数据准备脚本（视频原始实验）：将 MSR_VTT.json 转换为
OmniColPress 框架所需的格式，用于 AGC 官方视频实验。

与 prepare_data.py 的核心区别：
  - 语料库 (corpus) 使用原始 MP4 视频文件，而非伪查询文本
  - corpus JSONL 的每条记录含 "video" 字段（视频文件名）
  - 配合 assets_path 参数指定视频目录

输出目录：experiment/data-video/
生成文件：
  - train_corpus.jsonl     (训练集语料，每行: {"docid", "video"})
  - test_corpus.jsonl      (测试集语料)
  - train.jsonl            (训练数据，每行: {"query_id", "query", "positive_document_ids", "negative_document_ids"})
  - test_queries.csv       (测试查询，每行: query_id, query)   -- 与 text 实验完全相同
  - test_qrels.jsonl       (相关性标注，每行: {"query_id", "document_id", "relevance"})  -- 相同
  - stats.json             (数据统计)

MSR-VTT 标准划分:
  - Train: video0 - video6512  (6513 videos)
  - Val:   video6513 - video7009 (497 videos)
  - Test:  video7010 - video9999 (2990 videos)

MSR-VTT 1k-A Test Split:
  - 全量 2990 个 test video 作为候选库
  - 1000 个 query-video 对（测试集中每个视频取第一条 caption）
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict


# ─── 路径配置 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MSRVTT_PATH = PROJECT_ROOT / "MSR_VTT.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data-video"

# MSRVTT 视频目录（所有 mp4 文件）
VIDEO_DIR = Path("/hpc2hdd/home/yuxuanzhao/xuhaodong/MSRVTT/videos/all")

# ─── 数据划分 ───────────────────────────────────────────────
TRAIN_END = 6513       # video0 ~ video6512
VAL_END = 7010         # video6513 ~ video7009
TOTAL = 10000          # video7010 ~ video9999
NUM_TEST_QUERIES = 1000  # MSR-VTT 1k-A test split (前1000个 test video 作为 query)


def load_gt_annotations(path: Path) -> dict:
    """加载 GT 标注 → {video_id: [caption, ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann["caption"])
    return dict(gt)


def build_video_corpus(video_ids: list) -> list:
    """
    为指定 video_ids 构建视频语料。
    每条记录仅含 docid 和 video 文件名（配合 assets_path 解析完整路径）。
    """
    corpus = []
    missing = 0
    for vid in video_ids:
        mp4_name = f"{vid}.mp4"
        mp4_path = VIDEO_DIR / mp4_name
        if not mp4_path.exists():
            print(f"  Warning: {mp4_path} not found, skipping")
            missing += 1
            continue
        corpus.append({"docid": vid, "video": mp4_name})
    if missing > 0:
        print(f"  Warning: {missing} video files missing")
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
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(data: list, path: Path, fieldnames: list):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def main():
    print("=" * 60)
    print("AGC 视频原始实验 — 数据准备")
    print("=" * 60)

    # 检查视频目录
    if not VIDEO_DIR.exists():
        print(f"ERROR: 视频目录不存在: {VIDEO_DIR}")
        sys.exit(1)

    # 1. 加载 GT 标注
    print(f"\n[1/5] 加载 MSR_VTT.json ...")
    if not MSRVTT_PATH.exists():
        print(f"ERROR: MSR_VTT.json not found at {MSRVTT_PATH}")
        sys.exit(1)
    gt = load_gt_annotations(MSRVTT_PATH)
    print(f"  → {len(gt)} videos, {sum(len(v) for v in gt.values())} annotations")

    # 2. 划分 video IDs（与 text 实验完全一致）
    print("\n[2/5] 划分数据集 ...")
    all_ids = [f"video{i}" for i in range(TOTAL)]
    train_ids = all_ids[:TRAIN_END]                  # video0  ~ video6512
    val_ids   = all_ids[TRAIN_END:VAL_END]           # video6513 ~ video7009
    test_ids  = all_ids[VAL_END:]                    # video7010 ~ video9999
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # 3. 构建视频语料库
    print("\n[3/5] 构建视频语料库 ...")
    print("  → 构建训练语料 ...")
    train_corpus = build_video_corpus(train_ids)
    print(f"  → {len(train_corpus)} train docs")

    print("  → 构建测试语料 ...")
    test_corpus = build_video_corpus(test_ids)
    print(f"  → {len(test_corpus)} test docs")

    # 4. 构建训练 query 数据（与 text 实验完全相同格式）
    print("\n[4/5] 构建训练查询数据 ...")
    train_data = build_train_data(gt, train_ids)
    print(f"  → {len(train_data)} training query-doc pairs")

    # 5. 构建测试 queries 和 qrels（与 text 实验结构相同）
    print("\n[5/5] 构建测试查询 (MSR-VTT 1k-A) ...")
    queries, qrels = build_test_data(gt, test_ids, num_queries=NUM_TEST_QUERIES)
    print(f"  → {len(queries)} test queries, {len(qrels)} qrels")

    # ─── 输出 ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n写入文件到 {OUTPUT_DIR} ...")

    write_jsonl(train_corpus, OUTPUT_DIR / "train_corpus.jsonl")
    write_jsonl(test_corpus,  OUTPUT_DIR / "test_corpus.jsonl")
    write_jsonl(train_data,   OUTPUT_DIR / "train.jsonl")
    write_csv(queries,        OUTPUT_DIR / "test_queries.csv", fieldnames=["query_id", "query"])
    write_jsonl(qrels,        OUTPUT_DIR / "test_qrels.jsonl")

    # ─── 统计 ────────────────────────────────────────────────
    stats = {
        "video_dir": str(VIDEO_DIR),
        "train_corpus_size": len(train_corpus),
        "test_corpus_size": len(test_corpus),
        "train_queries": len(train_data),
        "test_queries": len(queries),
        "test_qrels": len(qrels),
        "note": "corpus records contain 'video' field (filename), use assets_path to resolve full path",
    }
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print(f"  train_corpus.jsonl : {len(train_corpus)} 条 (video field)")
    print(f"  test_corpus.jsonl  : {len(test_corpus)} 条 (video field)")
    print(f"  train.jsonl        : {len(train_data)} 条")
    print(f"  test_queries.csv   : {len(queries)} 条")
    print(f"  test_qrels.jsonl   : {len(qrels)} 条")
    print("=" * 60)
    print("\n下一步：")
    print("  验证: python experiment/validate_data.py --data-dir experiment/data-video")
    print("  训练: bash experiment/scripts/run_train_video.sh")


if __name__ == "__main__":
    main()
