"""
数据加载与预处理：加载 MSR_VTT.json + MSRVTT_narration.json，
构建统一数据结构，并按标准 split 划分。
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_narrations(narration_path: str) -> Dict[str, List[str]]:
    """加载 VLM 伪查询 narration，返回 {video_id: [caption_1, ..., caption_N]}"""
    with open(narration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    narrations: Dict[str, List[str]] = {}
    for item in data:
        vid = item["video_file"]  # e.g. "video0"
        captions = []
        idx = 1
        while f"caption_{idx}" in item:
            captions.append(item[f"caption_{idx}"])
            idx += 1
        narrations[vid] = captions
    return narrations


def load_gt_annotations(msrvtt_json_path: str) -> Dict[str, List[str]]:
    """加载 MSR-VTT ground-truth annotations，返回 {video_id: [sentence_1, ...]}"""
    with open(msrvtt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt: Dict[str, List[str]] = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann["caption"])
    return dict(gt)


def get_split_video_ids(
    train_end: int = 6513,
    val_end: int = 7010,
    total: int = 10000,
) -> Tuple[List[str], List[str], List[str]]:
    """返回 train / val / test 的 video id 列表（MSR-VTT 标准划分）"""
    train_ids = [f"video{i}" for i in range(0, train_end)]
    val_ids = [f"video{i}" for i in range(train_end, val_end)]
    test_ids = [f"video{i}" for i in range(val_end, total)]
    return train_ids, val_ids, test_ids


def build_retrieval_pairs(
    video_ids: List[str],
    gt: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """
    为给定 split 的视频构建 (video_id, query_text) 对。
    每个 GT caption 均作为独立检索查询。
    """
    pairs = []
    for vid in video_ids:
        if vid in gt:
            for cap in gt[vid]:
                pairs.append((vid, cap))
    return pairs


if __name__ == "__main__":
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    narrations = load_narrations(cfg["data"]["narration_json"])
    gt = load_gt_annotations(cfg["data"]["msrvtt_json"])
    train_ids, val_ids, test_ids = get_split_video_ids(
        cfg["split"]["train_end"], cfg["split"]["val_end"]
    )
    print(f"Narrations loaded: {len(narrations)} videos")
    print(f"GT annotations loaded: {len(gt)} videos")
    print(f"Split — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Train pairs: {len(build_retrieval_pairs(train_ids, gt))}")
