"""
原型库模块：
- PrototypeLibrary（方案B: 纯 nn.Parameter，端到端梯度更新）
- EMAPrototypeLibrary（方案C: nn.Parameter + EMA 影子副本 + 死原型重初始化）
- sinkhorn() — Sinkhorn-Knopp 最优传输软分配
- SwAVPrototypeLoss — 多视图交叉预测损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLibrary(nn.Module):
    """
    可学习原型库 P ∈ R^{K × d}（方案B）。
    随机初始化，端到端通过梯度更新。
    """

    def __init__(self, num_prototypes: int, feature_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes, gain=0.02)

    @property
    def K(self):
        return self.prototypes.shape[0]

    @property
    def d(self):
        return self.prototypes.shape[1]

    def forward(self):
        """返回 L2 归一化的原型矩阵 (K, d)"""
        return F.normalize(self.prototypes, p=2, dim=-1)


class EMAPrototypeLibrary(nn.Module):
    """
    可学习原型库 + EMA 影子副本（方案C）。
    - prototypes: nn.Parameter，参与梯度优化
    - ema_prototypes: buffer，指数移动平均副本，用于 Attention（稳定）
    - usage_count: buffer，追踪每个原型的使用频率
    """

    def __init__(self, num_prototypes: int, feature_dim: int, ema_decay: float = 0.999):
        super().__init__()
        self.ema_decay = ema_decay
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes, gain=0.02)

        self.register_buffer("ema_prototypes", self.prototypes.data.clone())
        self.register_buffer("usage_count", torch.zeros(num_prototypes))
        self.register_buffer("total_steps", torch.zeros(1, dtype=torch.long))

    @property
    def K(self):
        return self.prototypes.shape[0]

    @property
    def d(self):
        return self.prototypes.shape[1]

    def forward(self, use_ema: bool = False):
        """返回 L2 归一化的原型矩阵 (K, d)"""
        if use_ema:
            return F.normalize(self.ema_prototypes, p=2, dim=-1)
        return F.normalize(self.prototypes, p=2, dim=-1)

    @torch.no_grad()
    def update_ema(self):
        """每步调用，更新 EMA 影子副本"""
        self.ema_prototypes.mul_(self.ema_decay).add_(
            self.prototypes.data, alpha=1.0 - self.ema_decay
        )
        self.total_steps += 1

    @torch.no_grad()
    def update_usage(self, s_T: torch.Tensor, threshold: float = 0.01):
        """
        根据查询激活向量更新使用计数。
        Args:
            s_T: (B, K) — 查询激活向量
            threshold: 激活超过此值视为"被使用"
        """
        activated = (s_T.abs() > threshold).float().sum(dim=0)  # (K,)
        mask = activated > 0
        self.usage_count[mask] = 0
        self.usage_count[~mask] += 1

    @torch.no_grad()
    def reinit_dead_prototypes(self, token_features: torch.Tensor, dead_threshold: int = 100):
        """
        重初始化连续 dead_threshold 步未被使用的原型。
        Args:
            token_features: (B, M, d) — 当前 batch 的 token 特征
            dead_threshold: 连续未使用步数阈值
        Returns:
            n_dead: 重初始化的原型数量
        """
        dead_mask = self.usage_count >= dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        B, M, d = token_features.shape
        all_tokens = token_features.reshape(-1, d)
        n_tokens = all_tokens.shape[0]
        if n_tokens == 0:
            return 0

        indices = torch.randint(0, n_tokens, (n_dead,), device=all_tokens.device)
        new_protos = F.normalize(all_tokens[indices], p=2, dim=-1)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.prototypes.data[dead_indices] = new_protos
        self.ema_prototypes[dead_indices] = new_protos
        self.usage_count[dead_indices] = 0
        return n_dead


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, niters: int = 3):
    """
    Sinkhorn-Knopp 最优传输，产出满足等分约束的软分配矩阵。
    Args:
        scores: (N, K) — 样本与原型的相似度
        eps: 温度参数，越小分配越硬
        niters: 迭代次数
    Returns:
        Q: (N, K) — detached 软分配矩阵
    """
    Q = torch.exp(scores / eps).T  # (K, N)
    K, N = Q.shape
    Q /= Q.sum()

    for _ in range(niters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q *= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q *= N

    return (Q / N).T  # (N, K)


class SwAVPrototypeLoss(nn.Module):
    """
    SwAV 式多视图交叉预测损失。
    两个视图的 μ 通过 Sinkhorn 得到伪标签 Q，交叉预测对方的 softmax 分布。
    """

    def __init__(self, sinkhorn_eps: float = 0.05, sinkhorn_iters: int = 3,
                 swav_temperature: float = 0.1):
        super().__init__()
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.swav_temperature = swav_temperature

    def forward(self, mu1: torch.Tensor, mu2: torch.Tensor):
        """
        Args:
            mu1: (B, K) — 视图1的视频均值表示
            mu2: (B, K) — 视图2的视频均值表示
        Returns:
            loss: scalar
        """
        Q1 = sinkhorn(mu1, self.sinkhorn_eps, self.sinkhorn_iters)
        Q2 = sinkhorn(mu2, self.sinkhorn_eps, self.sinkhorn_iters)

        p1 = F.log_softmax(mu1 / self.swav_temperature, dim=-1)
        p2 = F.log_softmax(mu2 / self.swav_temperature, dim=-1)

        loss = -0.5 * (Q2 * p1).sum(dim=-1).mean() \
               -0.5 * (Q1 * p2).sum(dim=-1).mean()
        return loss
