"""
Memory Bank: 固定大小队列，存储最近 batch 的视频 μ 表示（detached），
用于扩充负样本以增强 InfoNCE 对比学习。
"""
import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """
    FIFO 队列式 Memory Bank。
    存储 detached 的视频表示 μ，每个 batch 入队新样本、出队最老的。
    """

    def __init__(self, queue_size: int, feature_dim: int):
        super().__init__()
        self.queue_size = queue_size
        # 队列不参与梯度
        self.register_buffer("queue", torch.randn(queue_size, feature_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        # 初始化时 L2 归一化
        self.queue = torch.nn.functional.normalize(self.queue, p=2, dim=-1)

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor):
        """
        将一个 batch 的特征入队。
        Args:
            features: (B, K) — detached, L2 归一化后的 μ 表示
        """
        B = features.shape[0]
        ptr = int(self.ptr.item())

        if ptr + B <= self.queue_size:
            self.queue[ptr : ptr + B] = features
        else:
            # 环绕写入
            overflow = (ptr + B) - self.queue_size
            self.queue[ptr:] = features[: B - overflow]
            self.queue[:overflow] = features[B - overflow :]

        self.ptr[0] = (ptr + B) % self.queue_size
        self.count[0] = min(self.count[0] + B, self.queue_size)

    def get_negatives(self) -> torch.Tensor:
        """
        返回队列中所有有效的负样本表示。
        Returns:
            negatives: (Q, K) — Q = min(count, queue_size)
        """
        q = int(self.count.item())
        if q == 0:
            return None
        return self.queue[:q].detach()
