# AGC (Adaptive Granularity Clustering) 代码管道详解

AGC（ Adaptive Granularity Clustering / Pseudo-Query）的核心目标是完成**基于伪查询的视频-文本细粒度检索**。该架构摒弃了粗暴的全局池化，而是通过交叉注意力“路由”生成一定数量的语义中心（伪查询），以“聚类”的方式将长片段视频的冗余帧特征合理压缩，随后与文本 Tokens 展开局部最大相似度匹配（MaxSim）。

## 1. 核心架构流 (Overall Pipeline Architecture)

整个前向传播高度结构化，被明确地划分为四个阶段 (Phase A - D)：

### 1.1 Phase A: 数据感知伪查询生成 (Data-Aware Pseudo Query Generation)
首先通过 Attention Pooling 将输入视频帧聚合为全局摘要，加到**可学习元查询 (Meta-Queries)**上进行条件化。
条件化查询与**全局可学习语义密码本 (Codebook)** 进行 Cross-Attention 路由，生成具有当前输入感知能力的伪查询 $Q_{\Psi}$。

### 1.2 Phase B: 联合编码与显著性计算 (Joint Encoding & Saliency Computation)
将原始视频 Tokens 和伪查询拼接，送入多层 Transformer 进行交互。
提取最后一层的注意力权重（即伪查询如何“注视”视频帧），将其在聚类维度上取平均，作为视频帧的**显著性分数 $\alpha$**。这用于后续评估每一帧的重要程度。

### 1.3 Phase C: Soft-to-Hard 退火聚类分配 (Soft-to-Hard Annealing Clustering)
计算联合编码后的视频特征 $Z_X$ 与伪查询聚类中心 $Z_{\Psi}$ 的归一化余弦相似度。
受一个线性下降的**退火温度调度器 ($\tau$)**的控制（由 `train.py` 从 $1.0$ 下降到 $0.1$），实施 Softmax。前期偏向软分配（平滑探索），训练后期逼近硬分配（聚类确立）。

### 1.4 Phase D: 显著性加权聚合与残差锚定 (Weighted Aggregation & Residual Anchoring)
将分配权重结合显著性 $\alpha$ 加权汇聚帧特征。
引入受限的残差系数量 $\lambda$，将聚合后的特征与伪查询本身融合，输出规模为 $m \times h$ 的最精简聚类级视频表示 $C$。

---

## 2. 模块职责解析 (Module Responsibilities)

### 2.1 AGC/models/agc_module.py (模型网络层)
*   **核心组件**：`AttentionPool`, `PseudoQueryGenerator`, `JointEncoder`, `AGCModel`主体。
*   **主要职责**：承载并串联上述 Phases A-D，维护两大可学习基底参数 (Codebook $E$ 和 Meta-Queries $M$) 和残差参数 $\lambda$。输出压缩后的视频表示 $c$ 和所有的中间状态（激活值/注意力图）`aux`，以便给辅损失和 Monitor 分析。

### 2.2 AGC/models/clip_encoder.py (文本侧处理)
*   **主要职责**：引入并**完全冻结**预训练的 CLIP Text Encoder。
*   **说明**：主要提供 Token 级特征 `encode_tokens`，保留文本细粒度语义以执行多对多 MaxSim 比对。

### 2.3 AGC/models/losses.py (算分与损失项)
*   **`max_sim_scores`**：利用 ColBERT 思想，将文本中的每一个有效词元去寻找对应匹配度最高的视频聚类中心（Max），随后求和得到总分，形成匹配矩阵。
*   **`info_nce_loss`**：核心主损失。对称的跨模态 InfoNCE 损失。
*   **辅助损失**：包含密码本多样性损失 `codebook_diversity_loss` 和 聚类均衡损失 `cluster_balance_loss`。分别对抗密码本“同质化”以及避免“旱的旱死、涝的涝死”带来的死聚类（Dead Cluster）。

### 2.4 AGC/monitor.py (动态健康诊断系统)
*   **主要职责**：用于决定是否要打开（或调大权重 `beta_div`/`beta_bal`）上述的辅助损失。
*   **监控指标**：
    *   **密码本健康度**：计算两两余弦相似度均值、最大值和特征的有效秩（Effective Rank）。
    *   **聚类均衡度**：统计各聚类中心分配情况，计算**Gini系数**、变异系数和零负载死聚类占比。如果出现告警则在终端输出修改建议。

### 2.5 AGC/train.py & AGC/evaluate.py (全管线生命周期)
*   **Train API (`AGC/train.py`)**：控制总步数中的全局 $\tau$（温度）线性退火，实现模型状态恢复 (Checkpoint)，执行 AMP 梯度缩放，并且定期触发 Monitor 健康检查日志。
*   **Evaluate API (`AGC/evaluate.py`)**：利用 `precompute_video_representations` 将验证集/测试集视频**离线批处理预先特征化压缩**，规避了推断期间视频侧的大量重复运算。然后按文本进行分批快速计算 R@1、R@5、R@10、MdR 和 MnR 等检索核心性能指标。

---

## 3. 核心交互流图景 (Interaction Flow)

整个交互流可以总结为以下步骤：

1. **数据抓取**：`Dataset/Dataloader` 泵出预训练视觉帧矩阵 $X$ 和文本 Captions。
2. **文本编码**：文本通过被冻结的 `CLIP` 文本编码器处理为 Tokens 序列，全程挂起无需梯度计算。
3. **视频特征压缩**：$X$ 序列送入 `AGCModel`，经过：
   * **汇聚条件化(Contextualization)** 
   * **路由密码本得伪查询(Routing Codebook)** 
   * **联合即插即用型Transformer编码(Joint Encoding)** 
   * **退火分配成聚类表示 $C$(Annealed Clustering)**。
4. **精细匹配**：生成的紧凑视频表示 $C$ 与 提取出的文本 Tokens 序列进入 `Losses::MaxSim`，计算序列与序列间的相似度，并生成批次对应的混淆打分矩阵。
5. **反向传播**：对比学习损失 `InfoNCE` 及附加的正则化惩罚项进行反向传播，用于更新 Transformer，及其内部可学习的密码本基底参数和元查询中心，以此循环直至收敛网络。

以下是 AGC (Adaptive Granularity Clustering) 架构全管道的完整数学公式推导。我们将按照前向传播的时间线，从视频帧输入到最终计算损失，分阶段进行严格定义。

### **符号定义**
*   $X \in \mathbb{R}^{n \times h}$: 视频帧提取的稠密特征（$n$ 为帧/Patch数量，$h$ 为特征维度）。
*   $TEXT$: 输入的自然语言文本描述。
*   $M \in \mathbb{R}^{m \times h}$: **可学习的元查询 (Meta-Queries)**，$m$ 为聚类中心的预设数量（如32）。
*   $E \in \mathbb{R}^{K \times h}$: **可学习的全局语义密码本 (Codebook)**，$K$ 为字典大小（如128）。
*   $\tau \in (0, 1]$: **退火温度参数**，随训练步数从 $1.0$ 线性衰减至接近 $0.1$。

---

### **Phase A: 数据感知伪查询生成 (Data-Aware Pseudo-Query Generation)**

首先，为生成适合当前视频的伪查询 $Q_\Psi$，模型提取视频的全局上下文，以条件化元查询并从密码本中路由语义。

1. **视频注意力池化 (Attention Pooling)**
   定义一个可学习的池化查询向量 $q_{pool} \in \mathbb{R}^h$，计算每个视频帧 $X_i$ 的注意力权重：
   $$ a_i = \frac{\exp(q_{pool}^\top X_i / \sqrt{h})}{\sum_{j=1}^n \exp(q_{pool}^\top X_j / \sqrt{h})} $$
   得到**视频全局摘要** $\bar{x} \in \mathbb{R}^h$:
   $$ \bar{x} = \sum_{i=1}^n a_i X_i $$

2. **元查询条件化 (Conditioning)**
   利用广播机制，将视频摘要附加到每一个元查询上：
   $$ \tilde{M} = M + \bar{x} \quad (\tilde{M} \in \mathbb{R}^{m \times h}) $$

3. **密码本路由 (Codebook Routing Cross-Attention)**
   以条件化元查询 $\tilde{M}$ 为 Query，密码本 $E$ 为 Key 和 Value，进行交叉注意力机制，生成**伪查询** $Q_\Psi$：
   $$ Q_\Psi = \text{Softmax}\left( \frac{\tilde{M} E^\top}{\sqrt{h}} \right) E \in \mathbb{R}^{m \times h} $$

---

### **Phase B: 联合编码与显著性计算 (Joint Encoding & Saliency)**

引入多层 Transformer Encoder，建立伪查询与原本视频序列之间的深层交互。

1. **联合自注意力 (Joint Attention)**
   将原始视频特征 $X$ 与伪查询 $Q_\Psi$ 在序列维度拼接，并传入 $L$ 层 Transformer：
   $$ H^{(0)} = [X \parallel Q_\Psi] \in \mathbb{R}^{(n+m) \times h} $$
   $$ H^{(L)} = \text{TransformerEncoder}(H^{(0)}) $$
   分离输出，得到深层编码后的视频特征 $Z_X \in \mathbb{R}^{n \times h}$ 与编码后的伪查询 $Z_\Psi \in \mathbb{R}^{m \times h}$。

2. **显著性分数计算 (Saliency Computation)**
   提取最后一层 Transformer 的注意力矩阵对角块，即 $m$ 个伪查询对 $n$ 个视频帧的注意力权重矩阵 $A_{\Psi \rightarrow X} \in \mathbb{R}^{m \times n}$。
   第 $i$ 帧的**显著性分数** $\alpha_i$ 为其接收到的平均关注度：
   $$ \alpha_i = \frac{1}{m} \sum_{j=1}^m A_{\Psi \rightarrow X}[j, i] $$

---

### **Phase C: Soft-to-Hard 退火聚类分配 (Annealed Clustering)**

计算每帧视频应归属于哪个聚类中心（伪查询）。

1. **余弦相似度矩阵**
   对于第 $i$ 帧特征 $Z_{X,i}$ 与第 $j$ 个伪查询 $Z_{\Psi,j}$：
   $$ S_{i,j} = \frac{Z_{X,i}^\top Z_{\Psi,j}}{\|Z_{X,i}\|_2 \|Z_{\Psi,j}\|_2} $$

2. **退火 Softmax 分配权重**
   使用温度调度器 $\tau$ 实现从软分配到硬分配的平滑过渡，得到分配权重矩阵 $W \in \mathbb{R}^{n \times m}$：
   $$ w_{i,j} = \frac{\exp(S_{i,j} / \tau)}{\sum_{k=1}^m \exp(S_{i,k} / \tau)} $$
   *(其中 $\sum_{j=1}^m w_{i,j} = 1$)*

---

### **Phase D: 显著性加权聚合与残差锚定 (Aggregation)**

将上述得到的分配权重及先前的显著性分数结合，以得出最终的视频紧缩表征。

1. **局部联合加权规范化**
   结合分配权重与该帧的显著性，针对每个聚类中心 $j$，在帧维度上进行规范化：
   $$ \hat{w}_{i,j} = \frac{\alpha_i \cdot w_{i,j}}{\sum_{i'=1}^n \alpha_{i'} \cdot w_{i',j} + \epsilon} $$

2. **聚合特征 (Weighted Aggregation)**
   获取第 $j$ 个中心的中间聚合特征 $V_j \in \mathbb{R}^h$：
   $$ V_j = \sum_{i=1}^n \hat{w}_{i,j} Z_{X,i} $$

3. **残差锚定 (Residual Anchoring)**
   为防止聚合过程中聚类崩塌或过度偏离原义，利用可学习的门控参数 $\lambda \in [0, 1]$ 引入初始生成时的残差 $Q_\Psi$：
   $$ C_j = V_j + \lambda Q_{\Psi, j} $$
   **最终的视频极简聚类集**即为 $C = \{C_1, C_2, \dots, C_m\} \in \mathbb{R}^{m \times h}$。

---

### **Phase E: Late Interaction 与 损失优化 (MaxSim & Loss)**

这里，通过完全独立的冻结 `CLIP Text Encoder` 处理文本 $TEXT$，输出文本词元特征 $T \in \mathbb{R}^{L \times h}$，并使用 $\text{mask}_l \in \{0,1\}$ 排除 Padding。

1. **细粒度词级最大相似度 (MaxSim)**
   视频表征 $C$ 与 文本表征 $T$ 的得分为每个词元对应的**最大视频聚类相似度之和**：
   $$ \text{score}(C, T) = \frac{1}{\sum_{l=1}^L \text{mask}_l} \sum_{l=1}^L \text{mask}_l \left( \max_{j \in [1,m]} \frac{C_j^\top T_l}{\|C_j\|_2 \|T_l\|_2} \right) $$

2. **跨模态对比损失 (InfoNCE Loss)**
   在一个拥有 $B$ 个视频-文本配对的 Batch 中（引入可学习温度标量 $\gamma$ 放大差异）：
   $$ \mathcal{L}_{NCE} = - \frac{1}{B} \sum_{b=1}^B \log \frac{\exp(\gamma \cdot \text{score}(C_b, T_b))}{\sum_{b'=1}^B \exp(\gamma \cdot \text{score}(C_b, T_{b'}))} $$
   *(实际实现中是对称损失，即同时计算 Text-to-Video 和 Video-to-Text 的交叉熵并求均值)*

3. **辅助罚项优化 (Auxiliary Penalties)**
   - **群组均衡损失 (Balance Loss)**: 惩罚使得聚类分配过于集中的情况，计算所有帧到聚类的边际分布概率 $P(j) = \frac{1}{n} \sum_{i=1}^n w_{i,j}$ 的负熵：
     $$ \mathcal{L}_{bal} = \sum_{j=1}^m P(j) \log P(j) $$
   - **总体目标**: $\mathcal{L}_{Total} = \mathcal{L}_{NCE} + \beta_{bal} \cdot \mathcal{L}_{bal} + \text{Diversity Loss}$ (如有)。