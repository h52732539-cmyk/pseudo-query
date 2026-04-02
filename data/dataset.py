"""
PyTorch Dataset: 为新 PQ Pipeline 提供训练和评估数据。
训练: QueryNarrationDataset — 返回 (video_id, query_text, narration_texts, frame_features)
评估: PseudoQueryEvalDataset — 返回 (video_id, captions, frame_features)
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class QueryNarrationDataset(Dataset):
    """
    训练数据集。每条样本输出 (video_id, query_text, narration_texts, frame_features)。
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        narrations: Dict[str, List[str]],
        frame_feat_dir: Optional[str] = None,
    ):
        self.pairs = pairs
        self.narrations = narrations
        self.frame_feat_dir = Path(frame_feat_dir) if frame_feat_dir else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vid, query = self.pairs[idx]
        captions = list(self.narrations[vid])

        frame_feats = None
        if self.frame_feat_dir is not None:
            feat_path = self.frame_feat_dir / f"{vid}.pt"
            if feat_path.exists():
                frame_feats = torch.load(feat_path, weights_only=True)

        return vid, query, captions, frame_feats


class PseudoQueryEvalDataset(Dataset):
    """评估用数据集，按 video_id 索引，返回伪查询 caption 列表 + 帧特征。"""

    def __init__(
        self,
        video_ids: List[str],
        narrations: Dict[str, List[str]],
        frame_feat_dir: Optional[str] = None,
    ):
        self.video_ids = video_ids
        self.narrations = narrations
        self.frame_feat_dir = Path(frame_feat_dir) if frame_feat_dir else None

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        captions = self.narrations[vid]

        frame_feats = None
        if self.frame_feat_dir is not None:
            feat_path = self.frame_feat_dir / f"{vid}.pt"
            if feat_path.exists():
                frame_feats = torch.load(feat_path, weights_only=True)

        return vid, captions, frame_feats


def query_narration_collate_fn(batch):
    """
    训练 collate。
    Returns:
        video_ids: List[str]
        queries: List[str]
        all_narrations: List[List[str]]
        frame_features: (B, K, d) or None
    """
    video_ids = [b[0] for b in batch]
    queries = [b[1] for b in batch]
    all_narrations = [b[2] for b in batch]

    frame_features = None
    if batch[0][3] is not None:
        frame_features = torch.stack([b[3] for b in batch])

    return video_ids, queries, all_narrations, frame_features


def eval_collate_fn(batch):
    """
    评估 collate。
    Returns:
        video_ids: List[str]
        all_captions: List[List[str]]
        frame_features: (B, K, d) or None
    """
    video_ids = [b[0] for b in batch]
    all_captions = [b[1] for b in batch]

    frame_features = None
    if batch[0][2] is not None:
        frame_features = torch.stack([b[2] for b in batch])

    return video_ids, all_captions, frame_features
