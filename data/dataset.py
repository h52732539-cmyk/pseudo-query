"""
PyTorch Dataset: 返回 (视频伪查询token特征, 查询文本) 用于训练和评估。
训练时按 (video_id, query_text) 配对采样；
评估时分别构建 video gallery 和 query list。
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class PseudoQueryTrainDataset(Dataset):
    """
    训练数据集。每条样本是一个 (video_id, query_text) 正样本对。
    视频侧：该视频的所有伪查询 caption 文本列表（由 collate 时编码）。
    查询侧：一条 ground-truth query 文本。
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
        captions = self.narrations[vid]  # List[str]
        return vid, captions, query


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


def train_collate_fn(batch):
    """
    返回:
        video_ids: List[str]
        all_captions: List[List[str]]  — 每个视频的伪查询文本列表
        queries: List[str]
    """
    video_ids = [b[0] for b in batch]
    all_captions = [b[1] for b in batch]
    queries = [b[2] for b in batch]
    return video_ids, all_captions, queries


def eval_collate_fn(batch):
    video_ids = [b[0] for b in batch]
    all_captions = [b[1] for b in batch]
    return video_ids, all_captions
