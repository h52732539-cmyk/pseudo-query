"""
PyTorch Dataset: 返回 (视频伪查询token特征, 查询文本) 用于训练和评估。
训练时按 (video_id, query_text) 配对采样，支持多视图（caption 分两组）；
评估时分别构建 video gallery 和 query list。
"""
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class PseudoQueryMultiViewDataset(Dataset):
    """
    多视图训练数据集。每条样本输出 (video_id, captions_view1, captions_view2, query_text)。
    视频的 captions 随机打乱后分为两组作为多视图。
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        narrations: Dict[str, List[str]],
    ):
        self.pairs = pairs
        self.narrations = narrations

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vid, query = self.pairs[idx]
        captions = list(self.narrations[vid])  # copy to avoid mutating
        random.shuffle(captions)
        mid = max(1, len(captions) // 2)
        view1 = captions[:mid]
        view2 = captions[mid:] if mid < len(captions) else captions[:mid]
        return vid, view1, view2, query


class PseudoQueryEvalDataset(Dataset):
    """评估用数据集，按 video_id 索引，返回伪查询 caption 列表。"""

    def __init__(
        self,
        video_ids: List[str],
        narrations: Dict[str, List[str]],
    ):
        self.video_ids = video_ids
        self.narrations = narrations

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        captions = self.narrations[vid]
        return vid, captions


def multiview_collate_fn(batch):
    """
    多视图训练 collate。
    返回:
        video_ids: List[str]
        captions_view1: List[List[str]]
        captions_view2: List[List[str]]
        queries: List[str]
    """
    video_ids = [b[0] for b in batch]
    captions_view1 = [b[1] for b in batch]
    captions_view2 = [b[2] for b in batch]
    queries = [b[3] for b in batch]
    return video_ids, captions_view1, captions_view2, queries


def eval_collate_fn(batch):
    video_ids = [b[0] for b in batch]
    all_captions = [b[1] for b in batch]
    return video_ids, all_captions
