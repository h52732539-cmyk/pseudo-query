# AGC (Adaptive Granularity Clustering) 代码管道详解 — Q-Former 范式

AGC（ Adaptive Granularity Clustering / Pseudo-Query）的核心目标是完成**基于伪查询的视频-文本细粒度检索**。该架构摒弃了粗暴的全局池化，采用 **Q-Former**（Query-Former）范式：通过条件化的可学习元查询，经多层交叉注意力直接从视频帧中聚合信息，生成 $m$ 个连续的语义表征向量，随后与文本 Tokens 展开局部最大相似度匹配（MaxSim）。

> **架构演进说明**：本版本移除了旧版的全局语义密码本 $E$、退火聚类分配 (Phase C) 和显著性加权聚合 (Phase D)。旧版中"元查询路由密码本"的两层间接映射导致了路由崩塌（所有元查询趋向取密码本均值）和聚类坍缩。新版直接让元查询通过交叉注意力从视频帧提取信息，架构更清晰，梯度流更通畅。

## 1. 核心架构流 (Overall Pipeline Architecture)

整个前向传播划分为两个主要阶段：

### 1.1 Phase A: 条件化伪查询生成与交叉注意力聚合 (Conditioned Pseudo-Query Generation & Cross-Attention Aggregation)

1. 通过 Attention Pooling 将输入视频帧聚合为全局摘要 $\bar{x}$，加到**可学习元查询 (Meta-Queries)** $M$ 上进行条件化：$\tilde{M} = M + \bar{x}$。
2. 条件化查询 $\tilde{M}$ 经过 $L$ 层 **QFormerBlock** 进行深度信息聚合。每层包含：
   - **Self-Attention**：伪查询间相互交互，促进多样性
   - **Cross-Attention**：Q=伪查询，K=V=视频帧特征，从视频中提取细粒度信息
   - **FFN + LayerNorm**
3. 输出 $Z_\Psi \in \mathbb{R}^{m \times h}$，直接作为该视频的 $m$ 个连续多向量表示 $C$。

### 1.2 Phase B: Late Interaction 与损失优化 (MaxSim & Loss)

通过冻结的 CLIP Text Encoder 获取文本 Token 特征 $T \in \mathbb{R}^{L \times h}$。视频表示 $C$ 与 $T$ 进行 MaxSim 匹配（每个文本词元找最相似的视频伪查询），计算跨模态 InfoNCE 损失，配合**正交正则化 $\mathcal{L}_{ortho}$** 保证伪查询多样性。

---

## 2. 模块职责解析 (Module Responsibilities)

### 2.1 AGC/models/agc_module.py (模型网络层)
*   **核心组件**：`AttentionPool`, `QFormerBlock`, `PseudoQueryGenerator`, `AGCModel` 主体。
*   **主要职责**：承载条件化元查询生成 → 多层 Q-Former 交叉注意力聚合的完整流程。维护可学习的元查询 $M$ 和 Q-Former 各层参数。输出压缩后的视频表示 $c$ 和辅助信息 `aux`。

### 2.2 AGC/models/clip_encoder.py (文本侧处理)
*   **主要职责**：引入并**完全冻结**预训练的 CLIP Text Encoder。
*   **说明**：主要提供 Token 级特征 `encode_tokens`，保留文本细粒度语义以执行多对多 MaxSim 比对。

### 2.3 AGC/models/losses.py (算分与损失项)
*   **`max_sim_scores`**：利用 ColBERT 思想，将文本中的每一个有效词元去寻找对应匹配度最高的视频伪查询表示（Max），随后求和得到总分，形成匹配矩阵。
*   **`info_nce_loss`**：核心主损失。对称的跨模态 InfoNCE 损失。
*   **`orthogonal_regularization_loss`**：正交正则化损失，强制 $m$ 个伪查询表示相互正交，防止坍缩为同质化表示。

### 2.4 AGC/monitor.py (动态健康诊断系统)
*   **主要职责**：保留为兼容工具。训练循环中改为直接计算伪查询间余弦相似度来监控多样性。

### 2.5 AGC/train.py & AGC/evaluate.py (全管线生命周期)
*   **Train API (`AGC/train.py`)**：实现模型状态恢复 (Checkpoint)，执行 AMP 梯度缩放，定期监控伪查询多样性（平均余弦相似度），定期触发失效分析。
*   **Evaluate API (`AGC/evaluate.py`)**：利用 `precompute_video_representations` 将验证集/测试集视频**离线批处理预先特征化压缩**，然后按文本进行分批快速计算 R@1、R@5、R@10、MdR 和 MnR 等检索核心性能指标。

---

## 3. 核心交互流图景 (Interaction Flow)

1. **数据抓取**：`Dataset/Dataloader` 泵出预训练视觉帧矩阵 $X$ 和文本 Captions。
2. **文本编码**：文本通过被冻结的 `CLIP` 文本编码器处理为 Tokens 序列，全程挂起无需梯度计算。
3. **视频特征压缩**：$X$ 序列送入 `AGCModel`，经过：
   * **注意力池化 + 元查询条件化 (Conditioning)**
   * **多层 Q-Former 交叉注意力聚合 (Cross-Attention Aggregation)**
4. **精细匹配**：生成的 $m$ 个连续视频表示 $C$ 与文本 Tokens 序列进入 `Losses::MaxSim`，计算序列与序列间的相似度，生成匹配打分矩阵。
5. **反向传播**：对比学习损失 `InfoNCE` 与正交正则化 $\mathcal{L}_{ortho}$ 进行反向传播，更新 Q-Former 各层参数和可学习的元查询，以此循环直至收敛。

---

## 4. 完整数学公式推导

以下是 Q-Former 范式 AGC 架构全管道的完整数学公式推导。

### **符号定义**
*   $X \in \mathbb{R}^{n \times h}$: 视频帧提取的稠密特征（$n$ 为帧/Patch数量，$h$ 为特征维度）。
*   $TEXT$: 输入的自然语言文本描述。
*   $M \in \mathbb{R}^{m \times h}$: **可学习的元查询 (Meta-Queries)**，$m$ 为伪查询数量（如32）。

---

### **Phase A: 条件化伪查询生成与多层交叉注意力聚合**

#### Step 1: 视频注意力池化 (Attention Pooling)

定义一个可学习的池化查询向量 $q_{pool} \in \mathbb{R}^h$，计算每个视频帧 $X_i$ 的注意力权重：
$$ a_i = \frac{\exp(q_{pool}^\top X_i / \sqrt{h})}{\sum_{j=1}^n \exp(q_{pool}^\top X_j / \sqrt{h})} $$
得到**视频全局摘要** $\bar{x} \in \mathbb{R}^h$:
$$ \bar{x} = \sum_{i=1}^n a_i X_i $$

#### Step 2: 元查询条件化 (Conditioning)

利用广播机制，将视频摘要附加到每一个元查询上：
$$ \tilde{M} = M + \bar{x} \quad (\tilde{M} \in \mathbb{R}^{m \times h}) $$

#### Step 3: 多层 Q-Former 交叉注意力聚合

初始化 $Q^{(0)} = \tilde{M}$，对视频帧 $X$ 进行归一化得到 $\hat{X} = \text{LayerNorm}(X)$。

对 $l = 1, 2, \dots, L$ 层，每层执行：

**自注意力**（伪查询间交互，促进多样性）：
$$ Q^{(l)}_{sa} = \text{LayerNorm}\left(Q^{(l-1)} + \text{SelfAttn}(Q^{(l-1)}, Q^{(l-1)}, Q^{(l-1)})\right) $$

**交叉注意力**（从视频帧聚合信息）：
$$ Q^{(l)}_{ca} = \text{LayerNorm}\left(Q^{(l)}_{sa} + \text{CrossAttn}(Q^{(l)}_{sa}, \hat{X}, \hat{X})\right) $$

**前馈网络**：
$$ Q^{(l)} = \text{LayerNorm}\left(Q^{(l)}_{ca} + \text{FFN}(Q^{(l)}_{ca})\right) $$

最终输出：
$$ C = Z_\Psi = \text{LayerNorm}(Q^{(L)}) \in \mathbb{R}^{m \times h} $$

这 $m$ 个向量直接作为该视频的 **multi-vector 表示**。

---

### **Phase B: Late Interaction 与损失优化 (MaxSim & Loss)**

通过完全独立的冻结 `CLIP Text Encoder` 处理文本 $TEXT$，输出文本词元特征 $T \in \mathbb{R}^{L \times h}$，并使用 $\text{mask}_l \in \{0,1\}$ 排除 Padding。

#### 1. 细粒度词级最大相似度 (MaxSim)
$$ \text{score}(C, T) = \sum_{l=1}^L \text{mask}_l \left( \max_{j \in [1,m]} \frac{C_j^\top T_l}{\|C_j\|_2 \|T_l\|_2} \right) $$

#### 2. 跨模态对比损失 (InfoNCE Loss)
在一个拥有 $B$ 个视频-文本配对的 Batch 中（引入可学习温度标量 $\gamma$ 放大差异）：
$$ \mathcal{L}_{NCE} = - \frac{1}{2B} \sum_{b=1}^B \left[ \log \frac{\exp(\gamma \cdot \text{score}(C_b, T_b))}{\sum_{b'} \exp(\gamma \cdot \text{score}(C_b, T_{b'}))} + \log \frac{\exp(\gamma \cdot \text{score}(C_b, T_b))}{\sum_{b'} \exp(\gamma \cdot \text{score}(C_{b'}, T_b))} \right] $$

#### 3. 正交正则化损失 (Orthogonal Regularization)
强制 $m$ 个伪查询表示相互正交，防止同质化坍缩：
$$ \mathcal{L}_{ortho} = \frac{1}{B} \sum_{b=1}^B \left\| \hat{C}_b \hat{C}_b^\top - I_m \right\|_F^2 $$
其中 $\hat{C}_b$ 为 $C_b$ 逐行 L2 归一化后的矩阵，$I_m$ 为 $m$ 阶单位矩阵。

#### 4. 总体目标
$$ \mathcal{L}_{Total} = \mathcal{L}_{NCE} + \beta_{ortho} \cdot \mathcal{L}_{ortho} $$
