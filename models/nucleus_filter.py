"""
跨模态核过滤门控模块（NarVid-Style）。
- FrameLevelCoAttention: 帧↔narration Co-Attention 互增强
- TemporalBlock: Transformer Encoder 建模时序关系
- NucleusFilter: 封装增强 + 核过滤（训练软权重 / 推理硬截断）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrameLevelCoAttention(nn.Module):
    """
    帧-narration 双向 Co-Attention。
    帧 attend narrations、narrations attend 帧，互相增强。
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.v2n_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.n2v_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.v_norm = nn.LayerNorm(feature_dim)
        self.n_norm = nn.LayerNorm(feature_dim)

    def forward(self, frame_feats, narr_feats, narr_mask=None):
        """
        Args:
            frame_feats: (B, K, d) — CLIP 视觉帧特征
            narr_feats:  (B, N, d) — CLIP narration 句子 embedding
            narr_mask:   (B, N) — narration padding mask (1=valid, 0=pad), optional
        Returns:
            enhanced_v: (B, K, d)
            enhanced_n: (B, N, d)
        """
        # key_padding_mask expects True=ignore, so invert
        narr_kpm = (~narr_mask.bool()) if narr_mask is not None else None

        # 帧 attend narrations
        v_attn_out, _ = self.v2n_attn(frame_feats, narr_feats, narr_feats,
                                       key_padding_mask=narr_kpm)
        enhanced_v = self.v_norm(frame_feats + v_attn_out)

        # narrations attend 帧
        n_attn_out, _ = self.n2v_attn(narr_feats, frame_feats, frame_feats)
        enhanced_n = self.n_norm(narr_feats + n_attn_out)

        return enhanced_v, enhanced_n


class TemporalBlock(nn.Module):
    """
    Transformer Encoder + 可学习位置编码，建模时序关系。
    视频帧和 narration 共享同一套权重。
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 4, ffn_dim: int = 2048, max_seq_len: int = 128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, L, d)
            mask: (B, L) — padding mask (1=valid, 0=pad), optional
        Returns:
            out: (B, L, d) — 残差连接 + L2归一化
        """
        L = x.shape[1]
        x_pos = x + self.pos_embed[:, :L, :]

        src_key_padding_mask = (~mask.bool()) if mask is not None else None
        out = self.encoder(x_pos, src_key_padding_mask=src_key_padding_mask)
        out = self.norm(x + out)  # 残差
        out = F.normalize(out, p=2, dim=-1)
        return out


class NucleusFilter(nn.Module):
    """
    核过滤门控：Co-Attention + Temporal + Nucleus 策略。
    训练时产出软权重（可微），推理时硬截断。
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8,
                 temporal_layers: int = 4, temporal_ffn: int = 2048,
                 max_seq_len: int = 128):
        super().__init__()
        self.co_attention = FrameLevelCoAttention(feature_dim, num_heads)
        self.temporal_block = TemporalBlock(
            feature_dim, num_heads, temporal_layers, temporal_ffn, max_seq_len
        )

    def enhance_features(self, frame_feats, narr_feats, narr_mask=None):
        """
        执行 Co-Attention + Temporal 增强。
        Args:
            frame_feats: (B, K, d)
            narr_feats:  (B, N, d)
            narr_mask:   (B, N) optional
        Returns:
            enhanced_v: (B, K, d)
            enhanced_n: (B, N, d)
        """
        enhanced_v, enhanced_n = self.co_attention(frame_feats, narr_feats, narr_mask)
        enhanced_v = self.temporal_block(enhanced_v)
        enhanced_n = self.temporal_block(enhanced_n, narr_mask)
        return enhanced_v, enhanced_n

    def compute_filter_weights(self, query_emb, enhanced_n, narr_mask=None):
        """
        计算 query-aware 软权重（训练时可微）。
        Args:
            query_emb:   (B, d) — 查询句子 embedding
            enhanced_n:  (B, N, d) — 增强后的 narration 特征
            narr_mask:   (B, N) optional — 1=valid, 0=pad
        Returns:
            weights: (B, N) — softmax 权重
        """
        # cosine similarity
        q = F.normalize(query_emb, p=2, dim=-1).unsqueeze(1)  # (B, 1, d)
        n = F.normalize(enhanced_n, p=2, dim=-1)               # (B, N, d)
        sim = (q * n).sum(dim=-1)  # (B, N)

        if narr_mask is not None:
            sim = sim.masked_fill(~narr_mask.bool(), float('-inf'))

        weights = F.softmax(sim, dim=-1)
        return weights

    @torch.no_grad()
    def nucleus_select(self, weights, threshold_p=0.4):
        """
        Nucleus 硬截断（推理用）。
        Args:
            weights:     (B, N) or (N,) — softmax 权重
            threshold_p: 累积概率阈值
        Returns:
            selected_indices: list of list — 每个样本选中的 narration 索引
            selected_weights:  list of Tensor — 每个样本选中的归一化权重
        """
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)

        B, N = weights.shape
        all_indices = []
        all_weights = []

        for b in range(B):
            w = weights[b]
            sorted_w, sorted_idx = torch.sort(w, descending=True)
            cumsum = torch.cumsum(sorted_w, dim=0)
            # 选中直到累积超过 threshold_p
            cutoff = (cumsum > threshold_p).nonzero(as_tuple=True)[0]
            if len(cutoff) == 0:
                k = N
            else:
                k = cutoff[0].item() + 1  # 包含刚好超过阈值的那个
            k = max(k, 1)  # 至少保留 1 个

            sel_idx = sorted_idx[:k]
            sel_w = sorted_w[:k]
            sel_w = sel_w / sel_w.sum()  # 归一化

            all_indices.append(sel_idx)
            all_weights.append(sel_w)

        return all_indices, all_weights
