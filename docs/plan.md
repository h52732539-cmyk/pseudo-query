# Plan: VIB-based Pseudo-Query Video Retrieval Pipeline

## TL;DR
基于 pq_1.md 的变分信息瓶颈（VIB）伪查询视频检索方案，设计完整的工程实现 pipeline。核心思路：VLM密集描述 → 文本编码+全局聚类建原型 → 变分网络压缩映射 → Cross-Attention查询激活 → 不确定性感知检索。数据集为 MSR-VTT（10,000视频），密集伪查询格式同 MSRVTT_narration.json。

---

## Phase 0: 数据准备与预处理

**Step 0.1 — 数据加载与格式统一**
- 输入：`MSR_VTT.json`（标准元数据 + ground-truth sentences）、`MSRVTT_narration.json`（10K视频完整版，格式: `{"video_file": "videoX", "caption_1": "...", ..., "caption_N": "..."}`）
- 构建统一数据结构：`Dict[video_id → List[str]]`，每视频的伪查询描述列表
- 从 MSR_VTT.json 中提取 ground-truth query（`sentences` 字段），用于训练/评测的 (video, query) 配对
- 划分 train/val/test split（按MSR-VTT标准：6,513 train / 497 val / 2,990 test）

**Step 0.2 — 文本编码器选择与初始化**
- 冻结的预训练文本编码器：CLIP ViT-B/32 的 Text Encoder 或 Sentence-BERT
- 对所有密集伪查询做 token-level 编码，得到每条 caption 的词元特征序列

---

## Phase 1: 密集伪查询生成（Dense Pseudo-Query Generation）— 已完成

- narration.json 即为 VLM（如LLaVA）对视频每帧/关键片段的描述产出
- 每视频 caption 数量不等（10~30+条），保留最大细粒度
- **此步骤为离线预处理，不需训练**

---

## Phase 2: VIB-based Token-level Prototype Reconstruction

### Step 2.1 — 全局词元池构建与聚类（离线，一次性）

1. 将所有10,000个视频的全部caption送入冻结文本编码器
2. **提取词元级特征**：每条caption → token序列特征 $\in \mathbb{R}^{L_j \times d}$（d=512/768取决于编码器）
3. **打散拼接**：将所有视频的所有caption的所有token pooling到一个全局token pool（预估规模：10,000视频 × 平均15条caption × 平均80 tokens ≈ 1200万tokens）
4. **全局聚类**：对全局token pool运行 K-Means（或 Mini-Batch K-Means 以适应大规模数据），提取 $K$ 个聚类中心
   - $K$ 为超参数，建议实验范围：256 / 512 / 1024 / 2048
   - 产出：**最优伪查询原型库** $P \in \mathbb{R}^{K \times d}$
5. **原型初始化后设为可学习参数** `nn.Parameter`，后续在训练中微调

### Step 2.2 — 变分压缩网络（Variational Compression Network）

**网络结构：**
```
输入: 视频 V_i 的所有密集token特征 {t_1, t_2, ..., t_M} (M = 该视频所有caption的总token数)
↓
[Token Aggregation] 平均池化 / 注意力池化 → 视频级聚合特征 h_i ∈ R^d
↓
[Mean Head]  Linear(d, K)  → μ_i ∈ R^K   (均值向量)
[Var Head]   Linear(d, K) + Softplus → σ²_i ∈ R^K  (方差向量，保证非负)
↓
输出: 视频 V_i 的变分表示 N(μ_i, diag(σ²_i))
```

**关键设计决策：**
- Token Aggregation 策略：建议使用 **多头注意力池化**（以K个原型作为query去attend所有token），而非简单的Mean Pooling
- Softplus 激活保证方差非负
- 训练时使用重参数化技巧（Reparameterization Trick）进行采样

---

## Phase 3: Query-Driven Fine-Grained Semantic Assembly

### Step 3.1 — 查询编码

- 真实查询文本 $T$ → 冻结文本编码器 → 词元序列特征 $E_T \in \mathbb{R}^{L \times d}$

### Step 3.2 — Cross-Attention 原型激活

- Query: $E_T$（L × d），Key/Value: $P$（K × d，可学习原型）
- 计算激活矩阵 $A \in \mathbb{R}^{L \times K}$：

  $$A_{l,k} = \frac{\exp(E_{T,l} \cdot P_k / \tau)}{\sum_{j=1}^K \exp(E_{T,l} \cdot P_j / \tau)}$$

- 温度参数 $\tau$ 为可学习标量或固定超参（建议初始 $\tau=0.07$）

### Step 3.3 — Max-Pooling 全局组装

- 沿序列维度 L 做 Max-Pooling，得到查询激活向量 $s_T \in \mathbb{R}^K$：

  $$s_{T,k} = \max_{l \in [1,L]} A_{l,k}$$

- $s_T$ 语义：查询对每个语义原型的最大激活强度（忽略介词/冠词等弱语义token）

---

## Phase 4: Uncertainty-Aware Retrieval & Hubness Optimization

### Step 4.1 — 不确定性感知匹配分数

$$Score(T, V_i) = s_T^\top m_i - \lambda \sum_{k=1}^K s_{T,k} \cdot \Sigma_{i,k}$$

- 第一项：查询激活与视频均值的点积（标准匹配）
- 第二项：方差惩罚项（$\lambda$ 为超参，建议范围 0.01~0.5）
- 高方差 = 语义不确定 → hub视频被惩罚

### Step 4.2 — 检索排序

- 对所有视频计算 Score，降序排列，取 Top-K 作为检索结果

---

## Phase 5: 训练流程

### 总损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \beta \mathcal{L}_{KL}$$

### Step 5.1 — Task InfoNCE Loss

$$\mathcal{L}_{task} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(Score(T_i, V_i)/\tau)}{\sum_{j=1}^B \exp(Score(T_i, V_j)/\tau)}$$

- B = batch size，分子为正样本对，分母为batch内所有样本

### Step 5.2 — KL Divergence Loss

$$\mathcal{L}_{KL} = \frac{1}{B} \sum_{i=1}^B D_{KL}(\mathcal{N}(m_i, \Sigma_i) \| \mathcal{N}(0, I))$$

- 闭式解：$D_{KL} = \frac{1}{2}\sum_{k=1}^K (\Sigma_{i,k} + m_{i,k}^2 - 1 - \ln \Sigma_{i,k})$
- $\beta$ 为信息瓶颈的权衡系数，建议使用 **β-annealing**（从小到大逐步增大）避免posterior collapse

### Step 5.3 — 训练超参建议

| 超参 | 建议值 | 备注 |
|------|--------|------|
| K (原型数) | 512 / 1024 | 消融实验确定 |
| d (特征维度) | 512 (CLIP) / 768 (BERT) | 跟随编码器 |
| τ (温度) | 0.07（可学习） | InfoNCE标准值 |
| β (KL权重) | 1e-4 → 1e-2（annealing） | 避免posterior collapse |
| λ (方差惩罚) | 0.1 | 消融实验确定 |
| Batch Size | 128 / 256 | 对比学习需大batch |
| LR | 1e-4（AdamW） | 带warmup+cosine decay |
| Epochs | 30~50 | 在val上早停 |

---

## Phase 6: 评估方案

### Step 6.1 — 标准评测指标
- **Text→Video Retrieval**：R@1, R@5, R@10, MdR (Median Rank), MnR (Mean Rank)
- 测试集：MSR-VTT 1K-A test split（1,000对）

### Step 6.2 — 消融实验
1. K 的选择（256 vs 512 vs 1024 vs 2048）
2. β-annealing 策略 vs 固定β
3. 方差惩罚项 λ 的影响（0 vs 0.01 vs 0.1 vs 0.5）
4. Token Aggregation：Mean Pooling vs Attention Pooling
5. 有/无 VIB（去掉KL项 + 去掉方差惩罚 = 退化为确定性版本）
6. Max-Pooling vs Mean-Pooling（Step 3.3）

### Step 6.3 — 与Baseline对比
- CLIP4Clip, DRL, Cap4Video, VidLA 等视频检索SOTA
- 纯伪查询 baseline（NarVid 等，直接整句匹配无VIB）

---

## 可训练模块总结

| 模块 | 参数 | 是否可训练 |
|------|------|-----------|
| 文本编码器 | CLIP Text Encoder | ❄️ 冻结 |
| 原型库 P | K × d | ✅ 可学习（K-Means初始化） |
| Token Aggregation | Attention-based | ✅ 可学习 |
| Mean Head | Linear(d, K) | ✅ 可学习 |
| Var Head | Linear(d, K) + Softplus | ✅ 可学习 |
| 温度 τ | 标量 | ✅ 可学习 |

---

## 关键文件清单

| 文件/模块 | 功能 |
|-----------|------|
| `data/preprocess.py` | 加载 narration.json + MSR_VTT.json，构建数据集 |
| `data/dataset.py` | PyTorch Dataset: 返回 (video_token_features, query_text) |
| `models/prototype.py` | 原型库 P 定义 + K-Means初始化逻辑 |
| `models/variational_encoder.py` | 变分压缩网络: Token Aggregation → (μ, σ²) |
| `models/query_assembly.py` | Cross-Attention 激活 + Max-Pooling 组装 |
| `models/scoring.py` | 不确定性感知匹配分数计算 |
| `train.py` | 训练主循环: InfoNCE + KL loss + β-annealing |
| `evaluate.py` | 评估: R@1/5/10, MdR, MnR |
| `scripts/build_prototypes.py` | 离线脚本: 全局token提取 + K-Means聚类 |
| `configs/` | 超参配置文件 |

---

## 决策与排除范围

**已决策（基于 pq_1.md）：**
- 采用方案B（变分分布后验推理），而非方案A/C
- 使用Cross-Attention做查询激活，Max-Pooling做聚合
- 不确定性感知评分公式已确定

**排除范围：**
- 不涉及视频视觉编码器（纯文本空间方法，VLM生成伪查询为离线过程）
- 不涉及 VLM 微调
- 不涉及多数据集泛化（仅MSR-VTT）

---

## 已确认决策

1. **Token Aggregation**：Prototype-guided Attention（用K个原型作为query attend所有视频token）
2. **对称损失**：是（双向 InfoNCE: Text→Video + Video→Text）
3. **实现框架**：PyTorch
