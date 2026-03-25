# Pseudo-Query Video Retrieval — 完整流程与数学推导

> 基于**逐原型对比学习 + SwAV 在线原型学习**的纯文本空间视频检索方法。  
> 核心范式：VLM 密集伪查询 → CLIP 文本编码 + 在线原型学习（无离线聚类） → 确定性编码器映射为逐原型聚合特征 → Cross-Attention 查询组装 + 逐原型对比评分 → Text→Video 检索。

---

## 目录

- [1. 整体架构概览](#1-整体架构概览)
- [2. Phase 0 — 数据准备与预处理](#2-phase-0--数据准备与预处理)
- [3. Phase 1 — 密集伪查询生成（离线）](#3-phase-1--密集伪查询生成离线)
- [4. Phase 2 — 在线原型学习](#4-phase-2--在线原型学习)
- [5. Phase 3 — 视频编码（逐原型聚合）](#5-phase-3--视频编码逐原型聚合)
- [6. Phase 4 — 查询驱动的细粒度语义动态组装](#6-phase-4--查询驱动的细粒度语义动态组装)
- [7. Phase 5 — 检索评分](#7-phase-5--检索评分)
- [8. 损失函数与训练优化](#8-损失函数与训练优化)
- [9. 端到端前向传播流程](#9-端到端前向传播流程)
- [10. 评估方案](#10-评估方案)
- [11. 可训练模块与超参数总结](#11-可训练模块与超参数总结)
- [12. 代码-模块对应关系](#12-代码-模块对应关系)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│               Pseudo-Query Retrieval Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────── 离线阶段 ────────────────────┐              │
│  │                                                    │              │
│  │  VLM 密集描述 ──→ MSRVTT_narration.json            │              │
│  │  （无需离线聚类，原型完全在线学习）                 │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌────────────────── 训练阶段（多视图） ────────────┐              │
│  │                                                    │              │
│  │  伪查询 (随机二分为 View1, View2)   GT 查询文本    │              │
│  │       │            │                    │          │              │
│  │  CLIP Token     CLIP Token         CLIP Token      │              │
│  │  Encoder        Encoder            Encoder         │              │
│  │       │            │                    │          │              │
│  │  PrototypeGuided  PrototypeGuided  CrossAttention  │              │
│  │  Attention         Attention       + MaxPooling    │              │
│  │       │            │                    │          │              │
│  │  h₁, μ₁ (B,K)  h₂, μ₂ (B,K)     s_T (B,K)      │              │
│  │       │            │              q̃ (B,K,d)       │              │
│  │       └─── SwAV ───┘                   │          │              │
│  │       │                                │          │              │
│  │       └──── 逐原型对比损失 ────────────┘          │              │
│  │                    │                               │              │
│  │         L = L_match + α · L_swav                   │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌──────────────────── 检索阶段 ────────────────────┐              │
│  │                                                    │              │
│  │  预计算所有视频 μ_i ∈ R^K                          │              │
│  │  实时编码查询 → s_T ∈ R^K                          │              │
│  │  Score = cosine(s_T, μ_i) / τ → Top-K             │              │
│  └──────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**核心创新点：**

1. **纯文本空间检索**：通过 VLM 伪查询将视觉信息完全转换为文本，消除跨模态鸿沟。
2. **在线原型学习**：移除离线 K-Means/GMM 聚类，原型作为可学习参数端到端优化（SwAV Sinkhorn 等分约束）。
3. **逐原型对比损失**：在每个原型的 $\mathbb{R}^d$ 子空间中独立计算视频-查询相似度，再以查询激活加权聚合。
4. **多视图训练**：随机将伪查询分为两个视图，利用 SwAV 交叉预测促进原型多样性和均匀使用。
5. **确定性编码**：移除变分推断（σ², KL散度），简化为确定性映射 μ ∈ R^K。

**两种方案分支：**

| 方案 | 原型学习方式 | 核心差异 |
|------|-------------|---------|
| **方案B (SwAV)** | 纯梯度更新 | 原型为 `nn.Parameter`，端到端训练 |
| **方案C (Hybrid)** | 梯度 + EMA 影子副本 | 视频编码用 EMA 副本（稳定），查询编码用梯度副本；死原型自动重初始化 |

---

## 2. Phase 0 — 数据准备与预处理

### 2.1 数据源

| 文件 | 内容 | 格式 |
|------|------|------|
| `MSR_VTT.json` | Ground-truth annotations | `{"annotations": [{"image_id": "videoX", "caption": "..."}]}` |
| `MSRVTT_narration.json` | VLM 密集伪查询 | `[{"video_file": "videoX", "caption_1": "...", ..., "caption_N": "..."}]` |

### 2.2 数据划分（MSR-VTT 标准）

| Split | 范围 | 数量 |
|-------|------|------|
| Train | video0 ~ video6512 | 6,513 |
| Val   | video6513 ~ video7009 | 497 |
| Test  | video7010 ~ video9999 | 2,990 |

### 2.3 Multi-View 数据集

训练时，每个视频的伪查询 captions 被随机等分为两个视图：

```
captions [c1, c2, c3, ..., cN]
    ↓ random shuffle + split
View1: [c3, c7, ...]    View2: [c1, c5, ...]
```

两个视图的 caption 分别编码后送入同一个 VideoEncoder，产生两组独立的视频表示 (h₁, μ₁) 和 (h₂, μ₂)，用于 SwAV 交叉预测。

> **实现文件**：`data/preprocess.py` — `load_narrations()`, `load_gt_annotations()`, `get_split_video_ids()`, `build_retrieval_pairs()`  
> **实现文件**：`data/dataset.py` — `PseudoQueryMultiViewDataset`, `multiview_collate_fn()`

---

## 3. Phase 1 — 密集伪查询生成（离线）

此步骤为**离线预处理**，已预先完成。

- 使用 VLM（如 LLaVA、Video-LLaMA）对 MSR-VTT 10,000 个视频逐帧/逐关键片段生成密集文本描述。
- 每个视频产出 10~30+ 条 caption，保留最大细粒度的视觉细节。
- 产出存储在 `MSRVTT_narration.json` 中。

---

## 4. Phase 2 — 在线原型学习

### 4.1 原型库定义

原型库 $P \in \mathbb{R}^{K \times d}$ 为可学习 `nn.Parameter`，Xavier 初始化，前向传播时 L2 归一化：

$$
P_k = \frac{P_k^{\text{raw}}}{\|P_k^{\text{raw}}\|_2}, \quad k = 1, \ldots, K
$$

**方案B**：纯 `nn.Parameter`，通过 SwAV loss 和匹配 loss 的梯度端到端更新。

**方案C**：额外维护 EMA 影子副本 $\bar{P}$，每步更新：

$$
\bar{P} \leftarrow \gamma \cdot \bar{P} + (1 - \gamma) \cdot P
$$

其中 $\gamma = 0.999$ 为 EMA 衰减系数。视频编码使用 $\bar{P}$（稳定），查询编码使用 $P$（保留梯度路径）。

### 4.2 Sinkhorn-Knopp 等分约束

给定样本-原型相似度 $S \in \mathbb{R}^{N \times K}$，通过 Sinkhorn 迭代产出满足等分约束的软分配矩阵 $Q$：

$$
Q^{(0)} = \frac{\exp(S / \varepsilon)}{\sum_{n,k} \exp(S_{n,k} / \varepsilon)}
$$

迭代（$t = 1, \ldots, T$）：

$$
Q^{(t)} \leftarrow \text{NormalizeRows}( \text{NormalizeCols}( Q^{(t-1)} ) )
$$

约束条件：$Q \mathbf{1} = \frac{1}{K}\mathbf{1}$（每列等分），$Q^\top \mathbf{1} = \frac{1}{N}\mathbf{1}$（每行等分）。

默认 $\varepsilon = 0.05$，$T = 3$。

### 4.3 死原型重初始化（方案C）

追踪每个原型的连续未使用步数。当某原型连续 $T_{\text{dead}} = 100$ 步激活值低于阈值时，从当前 batch 的 token 特征中随机采样一个 token 进行重初始化：

$$
P_k^{\text{dead}} \leftarrow \text{Normalize}(t_{\text{random}})
$$

> **实现文件**：`models/prototype.py` — `PrototypeLibrary`, `EMAPrototypeLibrary`, `sinkhorn()`, `SwAVPrototypeLoss`

---

## 5. Phase 3 — 视频编码（逐原型聚合）

### 5.1 输入：视频的密集 token 特征

对视频 $V_i$ 的伪查询 caption 进行 **token-level** 编码：

$$
\text{caption}_j \xrightarrow{\text{CLIP Token Encoder}} \{t_{j,1}, t_{j,2}, \ldots, t_{j,L_j}\} \in \mathbb{R}^{L_j \times d}
$$

将所有 caption 的 token 拼接、截断至最大 512 tokens：

$$
T_i = [t_{1,1}, \ldots, t_{1,L_1}, \ldots, t_{N_i, L_{N_i}}] \in \mathbb{R}^{M_i \times d}
$$

### 5.2 Prototype-Guided Attention

以 $K$ 个原型作为 query，attend 视频所有 token，为每个原型产出独立的聚合特征：

$$
Q = W_Q \cdot P \in \mathbb{R}^{K \times d}, \quad K_{\text{feat}} = W_K \cdot T_i \in \mathbb{R}^{M \times d}, \quad V = W_V \cdot T_i \in \mathbb{R}^{M \times d}
$$

多头注意力：

$$
\text{Attn}^{(h)} = \text{softmax}\left(\frac{Q^{(h)} (K_{\text{feat}}^{(h)})^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{K \times M}
$$

$$
h_i = W_O \cdot \text{Concat}(\text{head}^{(1)}, \ldots, \text{head}^{(H)}) \in \mathbb{R}^{K \times d}
$$

### 5.3 逐原型标量投影

对每个原型的聚合特征映射为标量激活值：

$$
\mu_{i,k} = W_\mu \cdot h_i[k] + b_\mu, \quad \mu_i \in \mathbb{R}^K
$$

L2 归一化：

$$
\mu_i \leftarrow \frac{\mu_i}{\|\mu_i\|_2}
$$

**注意**：不再有方差头（σ²），不再有重参数化采样。编码为确定性映射。

> **实现文件**：`models/video_encoder.py` — `PrototypeGuidedAttention`, `VideoEncoder`

---

## 6. Phase 4 — 查询驱动的细粒度语义动态组装

### 6.1 查询的 Token-Level 编码

$$
T \xrightarrow{\text{CLIP Token Encoder}} E_T = \{e_1, e_2, \ldots, e_L\} \in \mathbb{R}^{L \times d}
$$

### 6.2 Cross-Attention 原型激活

查询 token 作为 query，原型库 $P$ 作为 key，计算激活矩阵 $A \in \mathbb{R}^{L \times K}$：

$$
A_{l,k} = \frac{\exp\left(\frac{e_l \cdot P_k}{\tau}\right)}{\sum_{j=1}^{K} \exp\left(\frac{e_l \cdot P_j}{\tau}\right)}
$$

### 6.3 双重输出

**Max-Pooling 全局激活** — 每个原型被最相关词激活的最大强度：

$$
s_{T,k} = \max_{l \in [1, L]} A_{l,k}, \quad s_T \in \mathbb{R}^K
$$

$$
s_T \leftarrow \frac{s_T}{\|s_T\|_2}
$$

**逐原型查询语义** — 通过 $A^\top$ 将查询 token 聚合到各原型上：

$$
\tilde{q}_k = \sum_{l=1}^{L} A_{l,k} \cdot e_l, \quad \tilde{q} \in \mathbb{R}^{K \times d}
$$

$\tilde{q}$ 表示查询在每个原型方向上的细粒度语义，用于逐原型对比损失。

> **实现文件**：`models/query_assembly.py` — `QueryAssembly`

---

## 7. Phase 5 — 检索评分

推理时，使用简单的余弦相似度评分：

$$
\text{Score}(T, V_i) = \frac{s_T \cdot \mu_i}{\tau}
$$

由于 $s_T$ 和 $\mu_i$ 均已 L2 归一化，上式等价于缩放后的 cosine similarity。

**检索流程：**

1. **离线**：预计算所有 gallery 视频的 $\mu_i \in \mathbb{R}^K$
2. **在线**：编码查询 → $s_T \in \mathbb{R}^K$，计算 Score 对所有视频
3. **排序**：降序排列，取 Top-K 结果

> **实现文件**：`models/scoring.py` — `cosine_retrieval_score()`

---

## 8. 损失函数与训练优化

### 8.1 总损失函数

$$
\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{match}} + \alpha \cdot \mathcal{L}_{\text{swav}}}
$$

### 8.2 逐原型对比损失 $\mathcal{L}_{\text{match}}$

在每个原型 $k$ 的 $\mathbb{R}^d$ 空间中独立计算视频-查询相似度，然后加权聚合：

**逐原型相似度矩阵（$B \times B \times K$）：**

$$
\text{sim}_{i,j,k} = \cos(h_{i,k},\; \tilde{q}_{j,k})
$$

使用 `einsum('bkd, ckd -> bck')` 高效计算，无需 K 循环。

**加权聚合为匹配分数（$B \times B$）：**

$$
\text{Score}_{i,j} = \frac{1}{\tau} \sum_{k=1}^{K} w_{j,k} \cdot \text{sim}_{i,j,k}
$$

其中 $w_{j,k}$ 为查询 $j$ 在原型 $k$ 上的 L1 归一化激活权重：

$$
w_{j,k} = \frac{s_{T,j,k}}{\sum_{k'} s_{T,j,k'}}
$$

**对称 InfoNCE：**

$$
\mathcal{L}_{\text{match}} = \frac{1}{2}\left(\mathcal{L}_{\text{t2v}} + \mathcal{L}_{\text{v2t}}\right)
$$

$$
\mathcal{L}_{\text{t2v}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{Score}_{i,i})}{\sum_{j=1}^{B} \exp(\text{Score}_{i,j})}
$$

> **实现文件**：`models/scoring.py` — `prototype_anchored_contrastive_loss()`

### 8.3 SwAV 交叉预测损失 $\mathcal{L}_{\text{swav}}$

两个视图的 μ 通过 Sinkhorn 产生伪标签 $Q$，交叉预测对方的 softmax 分布：

$$
Q_1 = \text{Sinkhorn}(\mu_1), \quad Q_2 = \text{Sinkhorn}(\mu_2)
$$

$$
p_1 = \text{softmax}(\mu_1 / t_s), \quad p_2 = \text{softmax}(\mu_2 / t_s)
$$

$$
\mathcal{L}_{\text{swav}} = -\frac{1}{2}\left(\langle Q_2, \log p_1 \rangle + \langle Q_1, \log p_2 \rangle\right)
$$

其中 $t_s = 0.1$ 为 SwAV 温度。

> **实现文件**：`models/prototype.py` — `SwAVPrototypeLoss`

### 8.4 训练细节

| 项目 | 配置 |
|------|------|
| 优化器 | AdamW (lr=1e-4, weight_decay=0.01) |
| 学习率调度 | Warmup (10%) + Cosine Decay |
| 梯度裁剪 | max_norm = 1.0 |
| 混合精度 | FP16 (AMP) |
| Batch Size | 128 |
| Epochs | 30 |
| $\alpha$ (SwAV权重) | 0.5 |
| 早停 | 基于验证集 loss |

**已移除的组件：**

- ~~β-annealing / KL 散度~~ — 不再需要变分推断
- ~~方差头 σ²~~ — 确定性编码
- ~~free-bits~~ — 无 KL 损失
- ~~Memory Bank~~ — 改用多视图 SwAV
- ~~离线聚类 (K-Means / GMM)~~ — 原型在线学习

> **实现文件**：`train.py` — `main()`, `build_model()`

---

## 9. 端到端前向传播流程

### 9.1 训练时前向传播

```
输入: batch of (video_ids, captions_v1, captions_v2, query_texts)
             │              │               │
         CLIP Token      CLIP Token     CLIP Token
         Encoder         Encoder        Encoder
             │              │               │
      v1_feats (B,M,d)  v2_feats (B,M,d)  q_feats (B,L,d)
      v1_mask  (B,M)    v2_mask  (B,M)    q_mask  (B,L)
             │              │               │
             ▼              ▼               │
     PrototypeLibrary.forward()             │
       → P (K, d) L2-norm                  │
             │                              │
     ┌───────┤──────┐                      │
     ▼       │      ▼                      ▼
  VideoEncoder   VideoEncoder     QueryAssembly.forward()
  (View 1)       (View 2)              │
     │              │           logits = q @ P^T / τ
  h₁(B,K,d)     h₂(B,K,d)     A = softmax(logits)   (B,L,K)
  μ₁(B,K)       μ₂(B,K)       s_T = max_pool(A)     (B,K)
     │              │           q̃ = A^T @ q_feats     (B,K,d)
     │              │                   │
     ├──── SwAV ────┤                   │
     │  L_swav = CrossPred(μ₁, μ₂)     │
     │              │                   │
     └── h_avg ─────┘                   │
     (h₁+h₂)/2                         │
          │                             │
          └──── L_match ────────────────┘
          prototype_anchored_contrastive(h_avg, q̃, s_T)
                         │
              L_total = L_match + α · L_swav

[方案C附加] post_step: EMA更新 + 使用追踪 + 死原型重初始化
```

### 9.2 评估/推理时

```
[视频侧 — 离线预计算]
  所有视频的伪查询 → CLIP Token Encoder → (B,M,d)
  → model.get_video_repr() → μ_i ∈ R^K  for i=1..N

[查询侧 — 在线编码]
  查询文本 T → CLIP Token Encoder → (1,L,d)
  → model.get_query_repr() → s_T ∈ R^K

[评分与排序]
  scores(T, V_i) = cosine(s_T, μ_i) / τ   for all i
  排名 = argsort(scores, descending)
  返回 Top-K 视频
```

---

## 10. 评估方案

### 10.1 标准指标

| 指标 | 说明 | 方向 |
|------|------|------|
| R@1 | Top-1 召回率 | ↑ |
| R@5 | Top-5 召回率 | ↑ |
| R@10 | Top-10 召回率 | ↑ |
| MdR | Median Rank（中位排名） | ↓ |
| MnR | Mean Rank（平均排名） | ↓ |

### 10.2 测试模式

- **Full Test**：2,990 个测试视频，每个 GT caption 作为独立查询
- **1K-A Test**：MSR-VTT 1K-A 标准 split（1,000 对）

### 10.3 诊断分析

评估脚本额外输出：

- **原型聚类统计**：非空/空聚类数、各聚类视频分布（min/max/mean/median）
- **GT 视频 vs 预测视频的聚类归属一致性**
- **μ 全局分布统计**
- **逐查询详细分析 JSON**

> **实现文件**：`evaluate.py` — `compute_metrics()`, `print_prototype_statistics()`, `analyze_results()`

---

## 11. 可训练模块与超参数总结

### 11.1 模块一览

| 模块 | 参数量级 | 可训练 | 初始化方式 |
|------|---------|--------|-----------|
| CLIP Text Encoder | ~63M | ❄️ 冻结 | 预训练权重 |
| 原型库 $P \in \mathbb{R}^{K \times d}$ | $K \times d$ | ✅ | Xavier 随机初始化 |
| Prototype-guided Attention ($W_Q, W_K, W_V, W_O$) | $4 \times d^2$ | ✅ | 随机初始化 |
| Mean Head ($W_\mu$) | $d \times 1$ | ✅ | 随机初始化 |
| 可学习温度 $\log\tau$ | 1 | ✅ | $\ln(0.07)$ |
| EMA 影子原型 $\bar{P}$（方案C） | $K \times d$ | ❄️ buffer | 初始=P |

### 11.2 关键超参数

| 超参 | 默认值 | 说明 |
|------|--------|------|
| $K$ (原型数) | 512 | 原型数量 |
| $d$ (特征维度) | 512 | CLIP ViT-B/32 |
| $\tau$ (温度) | 0.07（可学习） | Cross-Attention 和匹配评分温度 |
| $\varepsilon$ (Sinkhorn) | 0.05 | 最优传输温度 |
| $T_{\text{sink}}$ (Sinkhorn 迭代) | 3 | 迭代次数 |
| $t_s$ (SwAV 温度) | 0.1 | SwAV softmax 温度 |
| $\alpha$ (SwAV 权重) | 0.5 | L_swav 损失权重 |
| $\gamma$ (EMA 衰减, 方案C) | 0.999 | EMA 移动平均衰减 |
| $T_{\text{dead}}$ (死原型阈值, 方案C) | 100 | 连续未使用步数阈值 |
| Batch Size | 128 | 对比学习需大 batch |
| Learning Rate | $10^{-4}$ | AdamW + warmup + cosine decay |

---

## 12. 代码-模块对应关系

| 流程阶段 | 代码文件 | 核心类/函数 |
|----------|----------|------------|
| 数据加载 | `data/preprocess.py` | `load_narrations()`, `load_gt_annotations()`, `build_retrieval_pairs()` |
| Multi-View 数据集 | `data/dataset.py` | `PseudoQueryMultiViewDataset`, `multiview_collate_fn()` |
| 文本编码 | `models/clip_encoder.py` | `CLIPTextEncoder.encode_tokens()`, `.encode_sentence()` |
| 原型库 (方案B) | `models/prototype.py` | `PrototypeLibrary`, `sinkhorn()`, `SwAVPrototypeLoss` |
| 原型库 (方案C) | `models/prototype.py` | `EMAPrototypeLibrary` |
| 视频编码器 | `models/video_encoder.py` | `VideoEncoder`, `PrototypeGuidedAttention` |
| 查询组装 | `models/query_assembly.py` | `QueryAssembly` |
| 对比损失 | `models/scoring.py` | `prototype_anchored_contrastive_loss()` |
| 推理评分 | `models/scoring.py` | `cosine_retrieval_score()` |
| 完整 Pipeline (方案B) | `models/pipeline_swav.py` | `SwAVPipelineModel` |
| 完整 Pipeline (方案C) | `models/pipeline_hybrid.py` | `HybridPipelineModel` |
| 训练 | `train.py` | `main()`, `build_model()`, `encode_video_captions()` |
| 评估 | `evaluate.py` | `build_video_representations()`, `build_query_representations()`, `compute_metrics()` |
| 冒烟测试 | `scripts/smoke_test.py` | `test_pipeline()` |
| 配置文件 | `configs/default.yaml` | 所有超参数 |
