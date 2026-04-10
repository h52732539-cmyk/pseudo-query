"""
损失函数: MaxSim 评分、InfoNCE、密码本多样性损失、聚类均衡损失。
"""
import torch
import torch.nn.functional as F


def max_sim_scores(
    c: torch.Tensor,
    q_tokens: torch.Tensor,
    q_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算 MaxSim 分数矩阵。

    对每个文本 token，找到与之最相似的视频聚类表示，求和得到总匹配分数。

    Args:
        c: (B_v, m, h) — 压缩后视频表示
        q_tokens: (B_q, L, h) — 文本 token 特征
        q_mask: (B_q, L) — 文本注意力掩码 (1=valid, 0=pad)
    Returns:
        scores: (B_v, B_q) — MaxSim 分数矩阵
    """
    c_norm = F.normalize(c, dim=-1)        # (B_v, m, h)
    q_norm = F.normalize(q_tokens, dim=-1)  # (B_q, L, h)

    # 所有 (video, text) 对的余弦相似度: (B_v, B_q, m, L)
    sim = torch.einsum("imh,jlh->ijml", c_norm, q_norm)

    # 对每个文本 token 取 max over clusters: (B_v, B_q, L)
    max_sim = sim.max(dim=2).values

    # 掩码 padding token
    mask = q_mask.unsqueeze(0).float()  # (1, B_q, L)
    max_sim = max_sim * mask

    # sum over text tokens: (B_v, B_q)
    scores = max_sim.sum(dim=-1)

    return scores


def info_nce_loss(scores: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """
    对称 InfoNCE 损失。

    Args:
        scores: (B, B) — 分数矩阵，对角线为正样本
        label_smoothing: float — label smoothing 系数
    Returns:
        loss: scalar
    """
    B = scores.size(0)
    labels = torch.arange(B, device=scores.device)
    loss_v2t = F.cross_entropy(scores, labels, label_smoothing=label_smoothing)
    loss_t2v = F.cross_entropy(scores.T, labels, label_smoothing=label_smoothing)
    return (loss_v2t + loss_t2v) / 2.0


def codebook_diversity_loss(codebook: torch.Tensor, tau_div: float = 0.5) -> torch.Tensor:
    """
    密码本多样性损失: 惩罚过高的条目间余弦相似度。

    L_div = mean(max(0, cos(E_i, E_j) - τ_div))  for i ≠ j

    Args:
        codebook: (K, h)
        tau_div: 相似度阈值
    Returns:
        loss: scalar
    """
    K = codebook.size(0)
    E_norm = F.normalize(codebook, dim=-1)
    cos_matrix = E_norm @ E_norm.T  # (K, K)

    # 去除对角线
    mask = ~torch.eye(K, dtype=torch.bool, device=codebook.device)
    cos_off_diag = cos_matrix.masked_select(mask)

    loss = F.relu(cos_off_diag - tau_div).mean()
    return loss


def cluster_balance_loss(w: torch.Tensor) -> torch.Tensor:
    """
    聚类均衡损失 (Switch Transformer 风格)。

    L_bal = m * Σ_k (f_k * p_k)
    f_k = 硬分配比例 (detached), p_k = 软分配均值

    Args:
        w: (B, n, m) — 软分配权重 (softmax over m)
    Returns:
        loss: scalar
    """
    B, n, m = w.shape

    # p_k: 平均软分配概率
    p = w.mean(dim=(0, 1))  # (m,)

    # f_k: 硬分配比例 (no gradient)
    with torch.no_grad():
        hard = w.argmax(dim=-1).view(-1)  # (B*n,)
        counts = torch.zeros(m, device=w.device)
        counts.scatter_add_(0, hard, torch.ones_like(hard, dtype=torch.float))
        f = counts / (B * n)

    loss = m * (f * p).sum()
    return loss
