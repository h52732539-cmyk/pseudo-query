# Pseudo-Query Video Retrieval — 完整流程与数学推导

> 基于**跨模态核过滤 + 查询-伪查询映射学习 + 推理时 Sinkhorn-Knopp 原型构建 + 粗筛精排两阶段检索**的视频检索方法。  
> 核心范式：VLM 密集伪查询 → **跨模态核过滤门控**（Co-Attention + Temporal Block，参考 NarVid）去除噪声/幻觉描述 → CLIP 文本编码 → 训练阶段学习查询→过滤后伪查询的映射（句子级 Adapter + Token 级 Reranker 联合训练） → 推理时 Per-Query 核过滤 → Sinkhorn-Knopp 聚类构建 Query-Specific 原型 → 粗筛（Adapter 输出 vs 原型）→ 精排（Token 级 Cross-Attention）→ Text→Video 检索。

---

## 目录

- [1. 整体架构概览](#1-整体架构概览)
- [2. Phase 0 — 数据准备与预处理](#2-phase-0--数据准备与预处理)
- [3. Phase 1 — 密集伪查询生成（离线）](#3-phase-1--密集伪查询生成离线)
- [4. Phase 1.5 — 跨模态核过滤门控（NarVid-Style）](#4-phase-15--跨模态核过滤门控narvid-style)
- [5. Phase 2 — 训练：查询-伪查询映射学习](#5-phase-2--训练查询-伪查询映射学习)
- [6. Phase 3 — 推理时原型构建（Per-Query 过滤 + Sinkhorn-Knopp 聚类）](#6-phase-3--推理时原型构建per-query-过滤--sinkhorn-knopp-聚类)
- [7. Phase 4 — 粗筛：原型级快速检索](#7-phase-4--粗筛原型级快速检索)
- [8. Phase 5 — 精排：Token 级 Cross-Attention 重排序](#8-phase-5--精排token-级-cross-attention-重排序)
- [9. 损失函数与训练优化](#9-损失函数与训练优化)
- [10. 端到端流程总览](#10-端到端流程总览)
- [11. 评估方案](#11-评估方案)
- [12. 可训练模块与超参数总结](#12-可训练模块与超参数总结)
- [13. 代码-模块对应关系](#13-代码-模块对应关系)
- [附录 A — 与旧 Pipeline 的对比](#附录-a--与旧-pipeline-的对比)

---

## 1. 整体架构概览

```
┌────────────────────────────────────────────────────────────────────────────┐
│        Pseudo-Query Video Retrieval Pipeline (v2 + Nucleus Filter)         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌───────────── 离线阶段 (已预完成) ────────────────┐                     │
│  │  VLM 密集描述 ──→ MSRVTT_narration.json           │                     │
│  │  CLIP ViT ──→ msrvtt_clip_frames/{vid}.pt (帧特征) │                     │
│  └───────────────────────────────────────────────────┘                     │
│                                                                            │
│  ┌──────── 训练阶段：核过滤 + 查询→伪查询映射学习 ────────┐              │
│  │                                                          │              │
│  │  frame_feats (B,K,d)     narr_sent (B,N,d)              │              │
│  │       │                       │                          │              │
│  │       └───── Co-Attention ────┘   ← 可训练 ~3.7M        │              │
│  │               │           │                              │              │
│  │          enhanced_v    enhanced_n                         │              │
│  │               │           │                              │              │
│  │          TemporalBlock  TemporalBlock  ← 可训练 ~12.7M  │              │
│  │               │           │              (共享权重)       │              │
│  │          temporal_v    temporal_n                         │              │
│  │                           │                              │              │
│  │   q_sent ──→ cos(q, temporal_n) → softmax → weights(B,N)│              │
│  │                           │                              │              │
│  │   filtered_centroid = Σ weights · narr_sent (原始CLIP)   │              │
│  │   ─────── 视觉特征到此丢弃，下游纯文本空间 ──────────   │              │
│  │                           │                              │              │
│  │   q_sent → QueryAdapter → adapted_q                     │              │
│  │                 │              │                          │              │
│  │         L_coarse ← InfoNCE(adapted_q, filtered_centroid) │              │
│  │                                                          │              │
│  │   q_tokens → FineGrainedReranker ← narr_tokens           │              │
│  │                    │                                      │              │
│  │            L_fine ← InfoNCE(score_matrix)                │              │
│  │                                                          │              │
│  │   L_total = L_coarse + β · L_fine                        │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                            │
│  ┌──────── 推理阶段：Per-Query 核过滤 + 两阶段检索 ───────┐              │
│  │                                                          │              │
│  │  Step 0: 预增强 — Co-Attn + Temporal → enhanced_n (一次) │              │
│  │  Step 1: Per-Query 核过滤 → 筛选高相关 narrations        │              │
│  │  Step 2: Sinkhorn-Knopp 聚类 → Query-Specific 原型       │              │
│  │  Step 3: Adapter 输出 vs 原型 → 候选视频集 (粗筛)        │              │
│  │  Step 4: Reranker(q_tokens, filtered_narr_tokens) (精排)  │              │
│  │                          │                                │              │
│  │                  最终检索结果                               │              │
│  └──────────────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────┘
```

**核心设计理念：**

1. **跨模态核过滤门控**：引入 NarVid-style Co-Attention + Temporal Block，让视频帧和伪查询 narrations 互相增强，检测并过滤幻觉/噪声描述。训练时软权重、推理时硬截断。**过滤后丢弃视觉特征，下游全部在文本空间进行。**
2. **训练目标为映射学习**：不再学习原型参数，而是学习「查询→过滤后伪查询」的映射关系——使真实查询 embedding 逼近其对应视频过滤后伪查询的加权聚合表示。
3. **推理时 Per-Query 原型构建**：每条查询独立进行核过滤 → Sinkhorn-Knopp 聚类 → 构建 Query-Specific 原型。原型天然反映查询相关的语义分布。
4. **粗筛→精排两阶段**：粗筛用句子级 Adapter 快速锁定候选集，精排用 Token 级 Cross-Attention 在过滤后 narrations 上精确打分。
5. **联合训练**：Co-Attention、Temporal Block、Adapter、Reranker 端到端联合优化，梯度从 loss 经 filtered_centroid → soft_weights → enhanced_n → co-attention/temporal 反向传播。

---

## 2. Phase 0 — 数据准备与预处理

### 2.1 数据源

| 文件 | 内容 | 格式 |
|------|------|------|
| `MSR_VTT.json` | Ground-truth annotations | `{"annotations": [{"image_id": "videoX", "caption": "..."}]}` |
| `MSRVTT_narration.json` | VLM 密集伪查询 | `[{"video_file": "videoX", "caption_1": "...", ..., "caption_N": "..."}]` |
| `data/msrvtt_clip_frames/{vid}.pt` | CLIP 视觉帧特征（预提取） | `torch.Tensor (K, d)` — K=12 帧, d=512 |

### 2.2 数据划分（MSR-VTT 标准）

| Split | 范围 | 数量 |
|-------|------|------|
| Train | video0 ~ video6512 | 6,513 |
| Val   | video6513 ~ video7009 | 497 |
| Test  | video7010 ~ video9999 | 2,990 |

### 2.3 训练数据组织

训练时每条样本为 **(query, narrations, frame_features)** 三元组：

```
训练集中的一条样本：
  query_text:      "a man is surfing in the ocean"        ← GT annotation
  narrations:      ["A surfer rides...", "Ocean waves...", ...]  ← 同视频的所有 VLM 伪查询
  frame_features:  Tensor (K, d)                          ← 预提取的 CLIP 视觉帧特征
  video_id:        "video123"
```

**不再需要**多视图分割（View1/View2），因为不再使用 SwAV 交叉预测。

> **实现文件**：`data/preprocess.py` — `load_narrations()`, `load_gt_annotations()`, `get_split_video_ids()`, `build_retrieval_pairs()`  
> **实现文件**：`data/dataset.py` — `QueryNarrationDataset`, `query_narration_collate_fn()`  
> **实现文件**：`scripts/extract_frame_features.py` — 帧特征预提取脚本

---

## 3. Phase 1 — 密集伪查询生成（离线）

此步骤为**离线预处理**，已预先完成。

- 使用 VLM（如 LLaVA、Video-LLaMA）对 MSR-VTT 10,000 个视频逐帧/逐关键片段生成密集文本描述。
- 每个视频产出 10~30+ 条 caption，保留最大细粒度的视觉细节。
- 产出存储在 `MSRVTT_narration.json` 中。

---

## 4. Phase 1.5 — 跨模态核过滤门控（NarVid-Style）

### 4.1 设计动机

VLM 生成的伪查询 narrations 数量多（每视频 10~30+ 条）且包含噪声和幻觉（hallucination）。直接使用全部 narrations 构建 centroid 或参与聚类，会引入不相关或错误的语义信号。

借鉴 NarVid（arXiv:2503.05186v4）的核过滤机制，引入**跨模态交互**（Co-Attention + Temporal Block）增强 narration 表示，使其包含视觉一致性信息，然后通过 nucleus sampling 策略筛选高价值 narrations。

**关键设计**：跨模态交互仅用于计算过滤权重。过滤完成后，丢弃视觉特征，下游全部使用**原始 CLIP 文本特征 + 过滤权重**，保持文本空间检索的一致性。

### 4.2 帧特征预提取

使用 CLIP ViT-B/32 视觉编码器（冻结）从每个视频均匀采样 $K = 12$ 帧，提取帧级特征：

$$
F_i = \{\mathbf{f}_1, \ldots, \mathbf{f}_K\} \in \mathbb{R}^{K \times d}, \quad d = 512
$$

预提取保存为 `data/msrvtt_clip_frames/{video_id}.pt`，训练和推理时直接加载。

> **实现文件**：`scripts/extract_frame_features.py`

### 4.3 FrameLevelCoAttention（可训练，~3.7M）

帧特征和 narration 句子特征通过 Co-Attention 互相增强：

$$
\text{enhanced\_v}, \text{enhanced\_n} = \text{CoAttention}(F_i, N_i)
$$

其中 $F_i \in \mathbb{R}^{K \times d}$ 为帧特征，$N_i \in \mathbb{R}^{N \times d}$ 为 narration 句子 embedding。

**具体机制**：
- $\text{Attn}_{v \leftarrow n}$：帧 attend narrations（帧从 narrations 中提取相关文本信息）
- $\text{Attn}_{n \leftarrow v}$：narrations attend 帧（narrations 从帧中获取视觉证据）
- 每个方向使用 `nn.MultiheadAttention(d, 8 heads)`
- 残差连接 + LayerNorm

**Hallucination 检测原理**：与视频内容不一致的 narration 在 $\text{Attn}_{n \leftarrow v}$ 中获得低注意力权重，其 enhanced 表示偏向原始值而非视觉增强值。

**参数量**：$2 \times (4d^2 + 2d) + 2d_{\text{LN}} \approx 3.7\text{M}$

### 4.4 TemporalBlock（可训练，~12.7M，共享权重）

增强后的帧/narration 序列通过 Temporal Block 建模时序关系：

$$
\text{temporal\_v} = \text{TempBlock}(\text{enhanced\_v}) \in \mathbb{R}^{K \times d}
$$
$$
\text{temporal\_n} = \text{TempBlock}(\text{enhanced\_n}) \in \mathbb{R}^{N \times d}
$$

**架构**（CLIP4Clip 风格）：
- 可学习位置编码 $\text{PE} \in \mathbb{R}^{L_{\max} \times d}$
- 4 层 Transformer Encoder（`nn.TransformerEncoderLayer`, d=512, 8 heads, FFN=2048）
- 残差连接 + L2 归一化

帧的**时序顺序** + narration 的**叙述顺序**均被建模，使 narration 表示融入序列上下文信息。

**参数量**：$4 \times (4d^2 \cdot 3 + d) + \text{PE} \approx 12.7\text{M}$（视频和 narration 共享同一套权重）

### 4.5 NucleusFilter — 核过滤策略

#### 4.5.1 训练时：软权重（可微）

训练时不做硬截断，使用 softmax 产出权重：

$$
w_k = \frac{\exp(\cos(\mathbf{q}_{\text{sent}},\; \tilde{\mathbf{n}}_k))}{\sum_{j=1}^{N} \exp(\cos(\mathbf{q}_{\text{sent}},\; \tilde{\mathbf{n}}_j))}, \quad \tilde{\mathbf{n}} = \text{temporal\_n}
$$

所有 narrations 参与但权重不等。过滤后的加权 centroid：

$$
\mathbf{c}_i^{\text{filtered}} = \frac{\sum_{k=1}^{N} w_k \cdot \mathbf{n}_k^{\text{clip}}}{\left\|\sum_{k=1}^{N} w_k \cdot \mathbf{n}_k^{\text{clip}}\right\|_2}
$$

注意：权重 $w_k$ 来自 temporal_n（增强后），但加权的目标 $\mathbf{n}_k^{\text{clip}}$ 是**原始 CLIP 句子 embedding**。

**梯度路径**：$\mathcal{L}_{\text{coarse}} \to \mathbf{c}_i^{\text{filtered}} \to w_k \to \cos(\mathbf{q}, \tilde{\mathbf{n}}_k) \to \tilde{\mathbf{n}} \to \text{Temporal} \to \text{CoAttn}$

#### 4.5.2 推理时：硬核过滤

推理时使用 nucleus sampling 策略硬截断：

$$
w_k = \text{softmax}(\cos(\mathbf{q}_{\text{sent}},\; \tilde{\mathbf{n}}_k))
$$

按 $w_k$ 降序排列，累积至 $\sum w_k > p$（默认 $p = 0.4$），选中的 narrations 及其归一化权重用于下游。

**接口**：

```python
# 增强特征（帧+narration 交互）
enhanced_v, enhanced_n = filter.enhance_features(frame_feats, narr_sent_embs)

# 训练时：软权重（完全可微）
weights = filter.compute_filter_weights(query_emb, enhanced_n)  # (B, N)

# 推理时：硬核过滤
selected_indices, selected_weights = filter.nucleus_select(weights, threshold_p=0.4)
```

> **实现文件**：`models/nucleus_filter.py` — `FrameLevelCoAttention`, `TemporalBlock`, `NucleusFilter`

---

## 5. Phase 2 — 训练：查询-伪查询映射学习

### 5.1 设计动机

传统方案（SwAV/Hybrid Pipeline）将原型作为可学习参数端到端训练，推理时也使用同一套原型。这导致：
- 原型必须在训练时见过足够分布才能泛化到测试集
- 训练目标（SwAV 交叉预测 + 逐原型对比损失）间接且复杂

新方案将训练目标简化为：**让真实查询的 embedding 在 CLIP 文本空间中逼近其对应视频经核过滤后伪查询的加权聚合表示**。核过滤确保 centroid 质量，映射学习直观且高效。

### 5.2 CLIP 编码（冻结，不参与训练优化）

#### 5.2.1 查询编码

对 GT 查询文本 $T$ 同时产出两种表示：

**句子级**（粗筛训练用）：

$$
\mathbf{q}_{\text{sent}} = \text{CLIP}_{\text{sentence}}(T) \in \mathbb{R}^d
$$

取 CLIP 的 EOS pooled output 再经过 text_projection，L2 归一化。

**Token级**（精排训练用）：

$$
E_T = \text{CLIP}_{\text{tokens}}(T) = \{e_1, e_2, \ldots, e_L\} \in \mathbb{R}^{L \times d}
$$

取 last_hidden_state 经 text_projection 后 L2 归一化，附带 attention_mask。

#### 5.2.2 伪查询编码（核过滤后加权聚合）

对视频 $V_i$ 的 $N_i$ 条伪查询 narrations：

**句子级聚合（Filtered Centroid）**—— 每条 narration 编码为句子 embedding，经核过滤加权聚合：

$$
\mathbf{c}_i = \frac{\sum_{j=1}^{N_i} w_j \cdot \text{CLIP}_{\text{sentence}}(\text{narr}_{i,j})}{\left\|\sum_{j=1}^{N_i} w_j \cdot \text{CLIP}_{\text{sentence}}(\text{narr}_{i,j})\right\|_2} \in \mathbb{R}^d
$$

其中 $w_j$ 为核过滤模块产出的软权重（训练时可微，见 Phase 1.5）。此 centroid 是该视频在过滤后伪查询空间中的「代表向量」。

**Token级**（精排训练用）—— 所有 narration 的 token 拼接截断：

$$
T_i = [t_{1,1}, \ldots, t_{1,L_1}, \ldots, t_{N_i, L_{N_i}}] \in \mathbb{R}^{M_i \times d}, \quad M_i \leq 512
$$

### 5.3 QueryAdapter — 句子级查询适配器

轻量 MLP 将真实查询的 CLIP 句子 embedding 投影到伪查询空间：

$$
\mathbf{q}_{\text{adapted}} = \text{QueryAdapter}(\mathbf{q}_{\text{sent}})
$$

具体架构：

$$
\begin{aligned}
\mathbf{h} &= \text{GELU}(W_1 \cdot \text{LayerNorm}(\mathbf{q}_{\text{sent}}) + b_1) & W_1 \in \mathbb{R}^{4d \times d} \\
\mathbf{q}_{\text{adapted}} &= \frac{\mathbf{q}_{\text{sent}} + W_2 \mathbf{h} + b_2}{\|\mathbf{q}_{\text{sent}} + W_2 \mathbf{h} + b_2\|_2} & W_2 \in \mathbb{R}^{d \times 4d}
\end{aligned}
$$

残差连接确保初始化时 Adapter 近似恒等映射（$W_2$ 零初始化），训练初期保留 CLIP 原始语义。

**参数量**：$(d \times 4d + 4d) + (4d \times d + d) + d_{\text{LN}} \approx 4 \times 512^2 \approx 1\text{M}$

> **实现文件**：`models/query_adapter.py` — `QueryAdapter`

### 5.4 FineGrainedReranker — Token 级精排器

查询 token 序列通过 Cross-Attention 关注 narration token 序列，产出匹配分数：

$$
\text{score} = \text{FineReranker}(E_T, T_i)
$$

具体架构：

**Cross-Attention 层**：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

其中 $Q = E_T$（查询 token 作为 query），$K = V = T_i$（narration token 作为 key/value）。使用 8 头多头注意力（`nn.MultiheadAttention`）。

**残差 + LayerNorm**：

$$
\hat{E}_T = \text{LayerNorm}(E_T + \text{Attn}(E_T, T_i, T_i))
$$

**Masked Mean Pooling → 标量分数**：

$$
\mathbf{r} = \frac{\sum_{l=1}^{L} m_l \cdot \hat{e}_l}{\sum_{l=1}^{L} m_l} \in \mathbb{R}^d
$$

$$
\text{score} = W_s \cdot \mathbf{r} + b_s \in \mathbb{R}
$$

其中 $m_l$ 为查询的 attention_mask，$W_s \in \mathbb{R}^{1 \times d}$。

**参数量**：$\text{MHA}(3 \times d^2 + d^2) + \text{LN}(2d) + \text{Score}(d + 1) \approx 4 \times 512^2 \approx 1\text{M}$

> **实现文件**：`models/fine_reranker.py` — `FineGrainedReranker`

### 5.5 训练时 B×B 分数矩阵构建

对一个 batch 中的 $B$ 个 (query, narrations) 对，需要计算所有 query-narration 的交叉匹配分数矩阵：

**粗筛分数矩阵**（高效，纯矩阵乘法）：

$$
S^{\text{coarse}}_{i,j} = \frac{\mathbf{q}^{\text{adapted}}_i \cdot \mathbf{c}_j}{\tau}, \quad S^{\text{coarse}} \in \mathbb{R}^{B \times B}
$$

**精排分数矩阵**（逐对 Cross-Attention）：

$$
S^{\text{fine}}_{i,j} = \text{FineReranker}(E_{T_i}, T_j), \quad S^{\text{fine}} \in \mathbb{R}^{B \times B}
$$

实现方式：将 $B$ 个 query 各与 $B$ 个 video 的 narration tokens 配对 → 展开为 $B^2$ 对 → 批量计算 → reshape 回 $(B, B)$。

> 因此 batch_size 从原 128 减至 32，以控制 $B^2$ 计算量和显存。

---

## 6. Phase 3 — 推理时原型构建（Per-Query 过滤 + Sinkhorn-Knopp 聚类）

### 6.1 设计动机

原型不再是训练参数，而是**推理时从测试集伪查询动态构建**。与 v1 的关键区别：
- **Per-Query 构建**：每条查询独立进行核过滤 → 聚类，产出 Query-Specific 原型
- 核过滤确保聚类输入是高质量、与查询相关的 narrations
- Sinkhorn-Knopp 等分约束确保原型均匀覆盖

### 6.2 Step 0 — 预增强（一次性，全测试集）

对所有测试集视频，执行 Co-Attention + Temporal Block 增强（一次性预计算）：

$$
\tilde{N}_i = \text{TempBlock}(\text{CoAttn}(F_i, N_i)) \in \mathbb{R}^{N_i \times d}
$$

缓存所有 $\tilde{N}_i$（enhanced_n），后续每条查询直接使用。同时缓存原始 CLIP 句子 embedding $N_i^{\text{clip}}$ 和 token features。

### 6.3 Step 1 — Per-Query 核过滤

对每条查询 $T$：

$$
w_k^{(v)} = \frac{\exp(\cos(\mathbf{q}_{\text{sent}},\; \tilde{\mathbf{n}}_k^{(v)}))}{\sum_j \exp(\cos(\mathbf{q}_{\text{sent}},\; \tilde{\mathbf{n}}_j^{(v)}))}
$$

对每个视频 $v$ 的 narrations，按 $w_k$ 降序累积至 $\sum > p$，选中的 narrations 保留原始 CLIP embedding + 权重。

$$
\text{Filtered}(T, v) = \{(\mathbf{n}_k^{\text{clip}}, w_k) \mid k \in \text{nucleus\_set}\}
$$

### 6.4 Step 2 — Sinkhorn-Knopp 聚类

在该查询的所有过滤后 narrations 上执行聚类。

**输入**：该查询过滤后的所有 narration embeddings $E^{\text{filtered}} = \{\mathbf{e}_1, \ldots, \mathbf{e}_{N'}\} \in \mathbb{R}^{N' \times d}$（$N' \leq N_{\text{total}}$）

**Step 1 — 初始化**：从 $E$ 中随机采样 $K$ 个 embedding 作为初始原型：

$$
P^{(0)}_k = \frac{\mathbf{e}_{\pi(k)}}{\|\mathbf{e}_{\pi(k)}\|_2}, \quad k = 1, \ldots, K
$$

**Step 2 — 迭代优化**（重复 $R$ 轮，默认 $R = 10$）：

**(a) 计算相似度矩阵**：

$$
S = E \cdot (P^{(r)})^\top \in \mathbb{R}^{N \times K}
$$

**(b) Sinkhorn-Knopp 最优传输** — 产出满足等分约束的软分配矩阵 $Q$：

$$
Q^{(0)} = \frac{\exp(S / \varepsilon)}{\sum_{n,k} \exp(S_{n,k} / \varepsilon)}
$$

迭代 $T$ 次（默认 $T = 3$）行列交替归一化：

$$
Q^{(t)} \leftarrow \text{NormalizeRows}\bigl(\text{NormalizeCols}(Q^{(t-1)})\bigr)
$$

最终 $Q \in \mathbb{R}^{N \times K}$，满足列和近似均匀（每个原型被等量分配）。

**(c) 加权更新质心**：

$$
P^{(r+1)}_k = \frac{\sum_{n=1}^{N} Q_{n,k} \cdot \mathbf{e}_n}{\left\|\sum_{n=1}^{N} Q_{n,k} \cdot \mathbf{e}_n\right\|_2}
$$

**输出**：$P^{(R)} \in \mathbb{R}^{K \times d}$（$K$ 个 L2 归一化原型），$Q^{(R)} \in \mathbb{R}^{N \times K}$（软分配矩阵）

### 6.5 构建倒排索引

将软分配硬化为 argmax 归属，建立**原型 → 视频**的倒排索引：

$$
\text{cluster}(j) = \arg\max_k Q_{j,k}
$$

$$
\text{InvertedIndex}[k] = \{V_i \mid \exists\, \text{narr}_j \in V_i \text{ s.t. } \text{cluster}(j) = k\}
$$

一个视频的多条 narrations 可能被分配到**不同原型**，因此同一视频可出现在多个原型的归属视频集中（多标签归属）。

同时缓存每个视频的 narration token features 用于精排阶段。

> **实现文件**：`models/prototype_builder.py` — `sinkhorn()`, `sinkhorn_cluster()`, `build_inverted_index()`, `InferenceIndex`

---

## 7. Phase 4 — 粗筛：原型级快速检索

### 7.1 查询适配

$$
\mathbf{q}_{\text{adapted}} = \text{QueryAdapter}(\text{CLIP}_{\text{sentence}}(T))
$$

使用训练好的 Adapter 将查询映射到伪查询空间。

### 7.2 原型匹配

计算查询与所有 $K$ 个原型的余弦相似度：

$$
s_k = \cos(\mathbf{q}_{\text{adapted}},\; P_k), \quad k = 1, \ldots, K
$$

### 7.3 候选视频集构建

取 top-$M$ 个最相似原型（默认 $M = 10$），合并其归属视频集：

$$
\mathcal{C}(T) = \bigcup_{k \in \text{top-}M(s)} \text{InvertedIndex}[k]
$$

可选：限制总候选数上限（默认 200），超出时按原型得分截断。

粗筛阶段的计算开销仅为 $O(K \cdot d)$ 的矩阵乘法（$K = 256, d = 512$），极其高效。

> **实现文件**：`models/prototype_builder.py` — `InferenceIndex.coarse_retrieve()`

---

## 8. Phase 5 — 精排：Token 级 Cross-Attention 重排序

### 8.1 候选视频的细粒度评分

对粗筛候选集 $\mathcal{C}(T)$ 中的每个视频 $V_j$，使用训练好的 Reranker 计算精排分数：

$$
\text{score}_{\text{fine}}(T, V_j) = \text{FineReranker}(E_T, T_j)
$$

其中 $E_T$ 为查询 token features，$T_j$ 为该视频的 narration token features（从缓存中读取）。

### 8.2 最终排序

一种方案是直接使用精排分数排序：

$$
\text{rank} = \text{argsort}(\text{score}_{\text{fine}}, \text{descending})
$$

也可以考虑加权融合粗筛和精排分数：

$$
\text{score}_{\text{final}} = \lambda \cdot \text{score}_{\text{coarse}} + (1 - \lambda) \cdot \text{score}_{\text{fine}}
$$

其中 $\text{score}_{\text{coarse}}$ 为查询与候选视频 centroid 的余弦相似度，$\lambda$ 可调。**默认 $\lambda = 0$**，即仅用精排分数（粗筛仅起候选召回作用）。

> **注意**：对于不在候选集中的视频，精排分数设为 $-\infty$。

> **实现文件**：`models/fine_reranker.py` — `FineGrainedReranker`  
> **实现文件**：`evaluate.py` — `fine_rerank_all()`

---

## 9. 损失函数与训练优化

### 9.1 总损失函数

$$
\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{coarse}} + \beta \cdot \mathcal{L}_{\text{fine}}}
$$

其中 $\beta$ 为精排损失权重，默认 $\beta = 0.5$。

### 9.2 粗筛对比损失 $\mathcal{L}_{\text{coarse}}$

在句子级别让适配后的查询 embedding 与对应视频的 narration centroid 对齐。

**分数矩阵**：

$$
S^{\text{coarse}}_{i,j} = \frac{\mathbf{q}^{\text{adapted}}_i \cdot \mathbf{c}_j}{\tau}
$$

**对称 InfoNCE**：

$$
\mathcal{L}_{\text{coarse}} = \frac{1}{2}\left(\mathcal{L}_{\text{q→n}} + \mathcal{L}_{\text{n→q}}\right)
$$

$$
\mathcal{L}_{\text{q→n}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S^{\text{coarse}}_{i,i})}{\sum_{j=1}^{B} \exp(S^{\text{coarse}}_{i,j})}
$$

$$
\mathcal{L}_{\text{n→q}} = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{\exp(S^{\text{coarse}}_{j,j})}{\sum_{i=1}^{B} \exp(S^{\text{coarse}}_{i,j})}
$$

### 9.3 精排对比损失 $\mathcal{L}_{\text{fine}}$

在 token 级别，让查询与对应视频的 narrations 获得最高匹配分数。

**分数矩阵**：

$$
S^{\text{fine}}_{i,j} = \text{FineReranker}(E_{T_i}, T_j) / \tau
$$

对 batch 中所有 $B \times B$ 对计算精排分数后，同样使用**对称 InfoNCE**：

$$
\mathcal{L}_{\text{fine}} = \frac{1}{2}\left(\mathcal{L}^{\text{fine}}_{\text{q→n}} + \mathcal{L}^{\text{fine}}_{\text{n→q}}\right)
$$

公式形式与 $\mathcal{L}_{\text{coarse}}$ 完全一致，仅分数矩阵不同。

### 9.4 训练细节

| 项目 | 配置 |
|------|------|
| 优化器 | AdamW (lr=1e-4, weight_decay=0.01) |
| 可训练模块 | CoAttention + TemporalBlock + QueryAdapter + FineReranker + log_τ |
| 总可训练参数 | ~18.4M |
| 学习率调度 | Warmup (10%) + Cosine Decay |
| 梯度裁剪 | max_norm = 1.0 |
| 混合精度 | FP16 (AMP) |
| Batch Size | 32 (因 B² 精排开销) |
| Epochs | 30 |
| $\beta$ (精排损失权重) | 0.5 |
| $\tau$ (温度) | 0.07, 可学习 |
| 早停 | 基于验证集 loss |

### 9.5 对比：新旧训练目标

| 维度 | 旧Pipeline (SwAV/Hybrid) | 新Pipeline (PQ) |
|------|--------------------------|-----------------|
| 原型 | 可学习 `nn.Parameter`，训练时更新 | 不存在，推理时构建 |
| 视频编码 | PrototypeGuidedAttention → $\mu \in \mathbb{R}^K$ | 无独立视频编码器 |
| 查询编码 | CrossAttention → $s_T \in \mathbb{R}^K$ | Adapter → $\mathbb{R}^d$ + Reranker → 标量 |
| 损失 | $L_{\text{match}} + \alpha \cdot L_{\text{swav}}$ | $L_{\text{coarse}} + \beta \cdot L_{\text{fine}}$ |
| 多视图 | ✅ 必需 | ❌ 不需要 |
| 训练信号 | 间接（原型→对比） | 直接（query embedding → narration embedding） |

> **实现文件**：`models/scoring.py` — `symmetric_infonce()`  
> **实现文件**：`models/pipeline_pq.py` — `PseudoQueryPipeline.compute_loss()`  
> **实现文件**：`train.py` — `main()`

---

## 9. 端到端流程总览

### 9.1 训练时前向传播

```
输入: batch of (video_ids, query_texts, narration_texts)
             │                  │
      CLIP Encoder (冻结)       │
             │                  │
    ┌────────┼──────────────────┤
    │        │                  │
    │   encode_sentence()    encode_sentence()      encode_tokens()
    │        │                  │                        │
    │   q_sent (B,d)      narr_sent (B,N,d)        q_tokens (B,L,d)
    │        │              → mean → L2norm            q_mask   (B,L)
    │        │              centroid (B,d)              │
    │        │                  │                        │
    │        ▼                  │                  encode_tokens()
    │   QueryAdapter            │                  (拼接截断)
    │   (MLP+残差)              │                        │
    │        │                  │                  narr_tokens (B,M,d)
    │        ▼                  ▼                  narr_mask   (B,M)
    │   adapted_q (B,d)   centroid (B,d)                │
    │        │                  │                        │
    │        └─── cosine ───────┘                        │
    │        S_coarse (B,B) / τ                          │
    │             │                                      │
    │    L_coarse = symmetric_infonce(S_coarse)          │
    │                                                    │
    │    q_tokens ──→ FineGrainedReranker ←── narr_tokens
    │               (B² 对 Cross-Attention)
    │                        │
    │                  S_fine (B,B) / τ
    │                        │
    │              L_fine = symmetric_infonce(S_fine)
    │                        │
    └────────────────────────┘
                  │
       L_total = L_coarse + β · L_fine
       反向传播 → 更新 Adapter + Reranker + log_τ
```

### 9.2 推理时流程

```
═══════════════════════════════════════════════════════════════════
                    Step 1: 构建原型索引 (一次性)
═══════════════════════════════════════════════════════════════════

测试集所有视频的 narrations
        │
   CLIP encode_sentence()
        │
   所有 narration embeddings  E ∈ R^{N_total × d}
        │
   Sinkhorn-Knopp 聚类 (K=256, R=10 轮)
        │
   ┌────┴────┐
   │         │
   P (K,d)   Q (N,K)  ← K 个原型 + 软分配矩阵
   原型矩阵   │
              ▼
         argmax 硬化 → 倒排索引  {prototype_k → {video_ids}}
                       narration token 缓存

═══════════════════════════════════════════════════════════════════
                    Step 2: 粗筛 (每查询 O(Kd))
═══════════════════════════════════════════════════════════════════

查询文本 T
    │
CLIP encode_sentence() → q_sent (d,)
    │
QueryAdapter → q_adapted (d,)
    │
cosine(q_adapted, P) → (K,)  相似度
    │
top-M 原型 → 合并倒排索引 → 候选视频集 C(T)
    (M=10)                    (|C| ≤ 200)

═══════════════════════════════════════════════════════════════════
                    Step 3: 精排 (每查询 O(|C|·L·M·d))
═══════════════════════════════════════════════════════════════════

对每个候选视频 Vj ∈ C(T):
    │
CLIP encode_tokens(T) → query_tokens (L, d)
    │
缓存读取 → narr_tokens_j (M_j, d)
    │
FineReranker(query_tokens, narr_tokens_j) → score_j
    │
按 score 降序排序 → 最终检索结果
```

---

## 11. 评估方案

### 11.1 标准指标

| 指标 | 说明 | 方向 |
|------|------|------|
| R@1 | Top-1 召回率 | ↑ |
| R@5 | Top-5 召回率 | ↑ |
| R@10 | Top-10 召回率 | ↑ |
| MdR | Median Rank（中位排名） | ↓ |
| MnR | Mean Rank（平均排名） | ↓ |

### 11.2 测试模式

- **Full Test**：2,990 个测试视频，每个 GT caption 作为独立查询
- **1K-A Test**：MSR-VTT 1K-A 标准 split（video7010~video8009，每视频取第 1 条 GT caption，1,000 对）

### 11.3 两阶段诊断指标

| 指标 | 说明 |
|------|------|
| **粗筛召回率** $\text{Recall}_{\text{coarse}}@M$ | GT 视频在粗筛候选集中的比例 |
| **平均候选集大小** | 粗筛返回的平均视频数 |
| **精排提升** | 对比仅粗筛 vs 粗筛+精排的 R@1/5/10 差异 |
| **原型利用率** | 非空原型比例、各原型归属视频分布 |
| **Sinkhorn 聚类质量** | 原型间余弦相似度分布、聚类紧致度 |
| **核过滤统计** | 平均保留 narration 数、过滤比例 |
| **Per-Query 聚类效率** | 平均聚类时间、原型利用率 |

### 11.4 分析输出

评估脚本产出 JSON 分析文件，包含：
- 每条查询的粗筛候选集、精排排名、GT 视频信息
- 聚类统计摘要
- 错误案例分析（GT 不在候选集 → 粗筛失败 / GT 在候选集但排名低 → 精排失败）

> **实现文件**：`evaluate.py` — `build_inference_index()`, `coarse_retrieve_all()`, `fine_rerank_all()`, `evaluate_two_stage()`, `compute_metrics()`

---

## 12. 可训练模块与超参数总结

### 12.1 模块一览

| 模块 | 参数量级 | 可训练 | 初始化方式 |
|------|---------|--------|-----------|
| CLIP Text Encoder | ~63M | ❄️ 冻结 | 预训练权重 |
| CLIP Visual Encoder | ~87M | ❄️ 冻结（仅帧特征预提取） | 预训练权重 |
| FrameLevelCoAttention | ~3.7M | ✅ | PyTorch 默认初始化 |
| TemporalBlock (4层 Transformer) | ~12.7M | ✅ | PyTorch 默认初始化 |
| QueryAdapter (LayerNorm + 2层MLP) | ~1M | ✅ | $W_2$ 零初始化（残差），$W_1$ Xavier |
| FineGrainedReranker (MHA + LN + Score Head) | ~1M | ✅ | PyTorch 默认初始化 |
| 可学习温度 $\log\tau$ | 1 | ✅ | $\ln(0.07)$ |

**总可训练参数：约 18.4M**（CoAttention 3.7M + Temporal 12.7M + Adapter 1M + Reranker 1M + τ）

### 12.2 关键超参数

| 超参 | 默认值 | 说明 |
|------|--------|------|
| $d$ (特征维度) | 512 | CLIP ViT-B/32 |
| $\tau$ (温度) | 0.07（可学习） | 对比损失温度 |
| $\beta$ (精排损失权重) | 0.5 | $\mathcal{L}_{\text{fine}}$ 权重 |
| Adapter 隐藏层倍数 | 4 | $W_1 \in \mathbb{R}^{4d \times d}$ |
| Reranker 注意力头数 | 8 | Multi-Head Attention |
| Temporal Block 层数 | 4 | Transformer Encoder |
| Temporal Block FFN | 2048 | Feed-Forward 隐藏维度 |
| 核过滤阈值 $p$ (训练) | — | 训练时不做硬截断，用软权重 |
| 核过滤阈值 $p$ (推理) | 0.4 | Nucleus 累积概率阈值 |
| $K$ (原型数) | 256 | 推理时聚类数 |
| $\varepsilon$ (Sinkhorn) | 0.05 | 最优传输温度 |
| $T_{\text{sink}}$ (Sinkhorn 迭代) | 3 | 每轮行列归一化迭代次数 |
| $R$ (聚类轮数) | 10 | 质心-分配交替更新轮数 |
| $M$ (粗筛 Top-M 原型) | 10 | 粗筛取前 M 个原型 |
| 最大候选数 | 200 | 粗筛候选视频上限 |
| 帧采样数 $K_{\text{frame}}$ | 12 | 每视频均匀采样帧数 |
| Batch Size | 32 | B² 精排约束 |
| Learning Rate | $10^{-4}$ | AdamW + warmup + cosine decay |
| Max Narr Tokens | 512 | 每视频 narration token 截断长度 |

---

## 13. 代码-模块对应关系

| 流程阶段 | 代码文件 | 核心类/函数 |
|----------|----------|------------|
| 数据加载 | `data/preprocess.py` | `load_narrations()`, `load_gt_annotations()`, `build_retrieval_pairs()` |
| 训练数据集 | `data/dataset.py` | `QueryNarrationDataset`, `query_narration_collate_fn()` |
| 评估数据集 | `data/dataset.py` | `PseudoQueryEvalDataset`, `eval_collate_fn()` |
| CLIP 文本编码 | `models/clip_encoder.py` | `CLIPTextEncoder.encode_tokens()`, `.encode_sentence()` |
| 帧特征预提取 | `scripts/extract_frame_features.py` | `extract_and_save()` |
| 核过滤门控 | `models/nucleus_filter.py` | `FrameLevelCoAttention`, `TemporalBlock`, `NucleusFilter` |
| 查询适配器（粗筛） | `models/query_adapter.py` | `QueryAdapter` |
| 精排器 | `models/fine_reranker.py` | `FineGrainedReranker` |
| 推理原型构建 | `models/prototype_builder.py` | `sinkhorn()`, `sinkhorn_cluster()`, `InferenceIndex` |
| 对比损失 | `models/scoring.py` | `symmetric_infonce()`, `coarse_prototype_score()` |
| 完整 Pipeline | `models/pipeline_pq.py` | `PseudoQueryPipeline` |
| 训练 | `train.py` | `main()`, `encode_video_captions()` |
| 评估 | `evaluate.py` | `precompute_enhanced_narrations()`, `per_query_filter_and_cluster()`, `evaluate_two_stage()` |
| 冒烟测试 | `scripts/smoke_test.py` | `test_pipeline()` |
| 配置文件 | `configs/default_pq.yaml` | 所有超参数 |

---

## 附录 A — 与旧 Pipeline 的对比

| 维度 | 旧 (SwAV/Hybrid) | 新 (PQ v2 + NucleusFilter) |
|------|-------------------|------------|
| **核过滤** | ❌ 无 | ✅ Co-Attention + Temporal + Nucleus（NarVid-style） |
| **原型来源** | 可学习参数，训练时更新 | 推理时 Per-Query 过滤后聚类构建 |
| **原型学习** | SwAV + Sinkhorn + EMA + 死原型重初始化 | Sinkhorn-Knopp 离线聚类（无参数） |
| **视频编码** | PrototypeGuidedAttention → $\mu \in \mathbb{R}^K$ | 无独立视频编码器（帧特征仅用于核过滤） |
| **查询编码** | CrossAttention + MaxPool → $s_T \in \mathbb{R}^K, \tilde{q} \in \mathbb{R}^{K \times d}$ | Adapter → $\mathbb{R}^d$ + Reranker → 标量 |
| **训练损失** | $L_{\text{match}} + \alpha \cdot L_{\text{swav}}$ | $L_{\text{coarse}} + \beta \cdot L_{\text{fine}}$ |
| **训练意义** | 间接：通过原型空间拉近 query 和 video | 直接：query embedding 逼近过滤后 narration embedding |
| **多视图** | ✅ 必需，用于 SwAV | ❌ 不需要 |
| **可训练参数** | ~2M | ~18.4M（含核过滤模块） |
| **检索流程** | 单阶段：$s_T \cdot \mu_i$ 全局排序 | 两阶段：Per-Query 核过滤 → 聚类 → 粗筛 → 精排 |
| **检索粒度** | 句子级（$K$ 维压缩表示） | 粗筛句子级 + 精排 Token 级 |
| **推理效率** | $O(N \cdot K)$ 全库扫描 | 粗筛 $O(K \cdot d)$ + 精排 $O(|\mathcal{C}| \cdot L \cdot M \cdot d)$ |

### 删除的模块

| 模块 | 旧文件 | 原因 |
|------|--------|------|
| PrototypeLibrary / EMAPrototypeLibrary | `models/prototype.py` | 原型不再是可学习参数 |
| SwAVPrototypeLoss | `models/prototype.py` | 不再使用 SwAV 交叉预测 |
| PrototypeGuidedAttention / VideoEncoder | `models/video_encoder.py` | 无独立视频编码器 |
| QueryAssembly | `models/query_assembly.py` | 替换为 QueryAdapter + FineReranker |
| SwAVPipelineModel | `models/pipeline_swav.py` | 替换为 PseudoQueryPipeline |
| HybridPipelineModel | `models/pipeline_hybrid.py` | 替换为 PseudoQueryPipeline |
| prototype_anchored_contrastive_loss | `models/scoring.py` | 替换为 symmetric_infonce |

### 保留/复用的模块

| 模块 | 文件 | 说明 |
|------|------|------|
| CLIPTextEncoder | `models/clip_encoder.py` | 冻结编码器，完全复用 |
| sinkhorn() | `models/prototype.py` → `models/prototype_builder.py` | 迁移到新文件 |
| cosine_retrieval_score() | `models/scoring.py` | 保留，粗筛评分可用 |
| load_narrations() 等 | `data/preprocess.py` | 数据加载完全复用 |
| PseudoQueryEvalDataset | `data/dataset.py` | 评估数据集保留 |
| encode_video_captions() | `train.py` | token 拼接截断逻辑复用 |
