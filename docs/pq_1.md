#### Step 1: 密集伪查询生成（Dense Pseudo-Query Generation）

* **操作**：参照 Narvid 等工作，使用 VLM 对视频 $V_i$ 进行极高密度的解析（例如每帧或每个关键片段生成一次描述）。
* **产出**：每个视频获得一个极其密集的文本描述集合，此时保留了最细粒度的视觉细节，但也伴随大量冗余和背景噪声。

#### Step 2: 基于变分信息瓶颈的词级原型重构（VIB-based Token-level Prototype Reconstruction）

* **第一小步（全局聚类构建底座）**：将所有视频的密集伪查询打散为细粒度的词元（Tokens），经过全局聚类，提取出 $K$ 个最具代表性的语义聚类中心，构建出**最优伪查询库（Optimal Pseudo-query Prototypes）** $P \in \mathbb{R}^{K \times d}$。* **第二小步（变分压缩映射）**：针对具体视频 $V_i$ 的密集词元，我们利用变分网络（Variational Network）将其压缩映射到这 $K$ 个原型上。网络输出关于这 $K$ 个原型的均值 $m_i \in \mathbb{R}^K$ 和方差 $\Sigma_i \in \mathbb{R}^K$。
* **物理意义**：视频 $V_i$ 不再是确定的特征向量，而是被表示为一个多变量高斯分布 $\mathcal{N}(m_i, \Sigma_i)$。均值代表对某个原型的归属概率，方差代表该归属的**不确定度（噪声程度）**。

#### Step 3: 查询驱动的细粒度语义动态组装（Query-Driven Fine-Grained Semantic Assembly）

* **文本编码**：真实查询 $T$ 输入文本编码器，得到词元序列特征 $E_T \in \mathbb{R}^{L \times d}$（$L$ 为序列长度，$d$ 为特征维度）。
* **交叉注意力激活**：使用 Cross-Attention 让 $T$ 中的每一个词元去“巡视”并激活 $K$ 个原型。第 $l$ 个词元对第 $k$ 个原型的激活强度为：

$$A_{l,k} = \frac{\exp(E_{T,l} \cdot P_k / \tau)}{\sum_{j=1}^K \exp(E_{T,l} \cdot P_j / \tau)}$$


* **全局组装**：为了提取最具辨识度的核心查询词汇（忽略无意义的介词等），沿序列长度 $L$ 进行**最大池化（Max-Pooling）**，得到真实查询 $T$ 对全局原型的理想激活向量 $s_T \in \mathbb{R}^K$：

$$s_{T, k} = \max_{l \in [1, L]} A_{l,k}$$



#### Step 4: 不确定性感知检索与枢纽问题优化（Uncertainty-Aware Retrieval & Hubness Optimization）

* **匹配过程**：在进行视频找回时，不使用传统的余弦相似度，而是利用变分分布的均值 $m_i$ 和方差 $\Sigma_i$ 计算**不确定性感知的匹配分数（Uncertainty-aware Score）**。
* **方差惩罚（解决枢纽问题）**：贴近原型中心、语义模棱两可的“万金油”视频（Hubs）往往具有极大的方差 $\Sigma_i$。我们将方差作为惩罚项引入：

$$Score(T, V_i) = s_T^\top m_i - \lambda \sum_{k=1}^K s_{T, k} \cdot \Sigma_{i, k}$$


* **结果**：若查询 $T$ 强激活了某个原型（$s_{T,k}$ 很大），但视频在该原型上极不确定（$\Sigma_{i,k}$ 很大），匹配分数将被大幅拉低，从而精准过滤掉高频错误召回。

---

### 二、 损失优化框架（Loss Optimization Framework）

为了实现上述机制，整体模型的优化完全遵循**变分信息瓶颈（VIB）**理论。总损失函数由两部分构成：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \beta \mathcal{L}_{KL}$$

#### 1. 任务损失 / 匹配对齐（Task InfoNCE Loss）

要求压缩后的均值表示 $m_i$ 能够与真实查询激活向量 $s_T$ 形成强匹配。这里采用基于上述不确定性分数的 InfoNCE 损失：


$$\mathcal{L}_{task} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(Score(T_i, V_i) / \tau)}{\sum_{j=1}^{B} \exp(Score(T_i, V_j) / \tau)}$$


*(注：$B$ 为 Batch Size，分子为正样本对，分母为 Batch 内的所有负样本对。)*

#### 2. 信息瓶颈压缩损失（KL Divergence Loss）

为了强制网络过滤密集伪查询中的视觉噪声，并生成具有物理意义的方差 $\Sigma_i$，我们计算视频的变分分布 $\mathcal{N}(m_i, \Sigma_i)$ 与先验分布（通常设为标准正态分布 $\mathcal{N}(0, I)$）之间的 KL 散度：


$$\mathcal{L}_{KL} = \frac{1}{B} \sum_{i=1}^{B} D_{KL}(\mathcal{N}(m_i, \Sigma_i) \parallel \mathcal{N}(0, I))$$

* **作用机制**：如果某个原型的语义是背景噪声，网络为了降低 $\mathcal{L}_{KL}$，会主动将其对应的方差 $\Sigma_{i,k}$ 拉大，并将均值 $m_{i,k}$ 逼近 0。只有对 $\mathcal{L}_{task}$ 极其关键的语义，网络才愿意承受 KL 惩罚，保留高置信度（小方差）和高均值。这在训练阶段实现了“无监督的噪声过滤”。
