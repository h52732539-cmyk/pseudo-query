# AGC 方案二：伪查询作为聚类中心 — 完整流程梳理

> 本文档基于 readme_AGC.md 梳理，不涉及其他项目代码，为全新管线的设计规格。

---

## 0. 整体架构概览

```
输入视频帧
    │
    ▼
[视频 Token 化]  →  X ∈ R^{n×h}（n 个 patch token，维度 h）
    │
    ├──────────────────────────────┐
    ▼                              ▼
[Phase A: 伪查询生成]        [视频摘要 x̄]
    │                              │
    │◄─────────────────────────────┘
    ▼
Q_Ψ ∈ R^{m×h}（m 个伪查询）
    │
    ▼
[Phase B: 联合编码 + 显著性计算]
    │
    ├── Z_X^(L) ∈ R^{n×h}（编码后视频 token）
    ├── Z_Ψ^(L) ∈ R^{m×h}（编码后伪查询 = 聚类中心 μ）
    └── α ∈ R^n（每个视频 token 的显著性分数）
    │
    ▼
[Phase C: 聚类分配（Soft-to-Hard 退火）]
    │
    └── w_{kj}（视频 token j 被分配到聚类 k 的权重）
    │
    ▼
[Phase D: 显著性加权聚合 + 残差锚定]
    │
    └── c_k ∈ R^h（k=1..m，最终压缩表示）
    │
    ▼
[Phase E: 检索匹配 + 损失优化]
    │
    └── MaxSim(c, text) → InfoNCE loss
```

---

## 1. Phase A：数据感知伪查询生成

### 1.1 可学习参数

| 参数 | 形状 | 说明 |
|------|------|------|
| 全局语义密码本 $E$ | $K \times h$ | $K$=128~256，所有视频共享 |
| 元查询 $M$ | $m \times h$ | $m$=32（默认），可学习初始化 |

### 1.2 前向流程

**Step 1 — 视频摘要提取**

对输入视频 token $X \in \mathbb{R}^{n \times h}$ 做注意力池化（attention pooling），得到视频级摘要向量：

$$\bar{x} = \text{AttnPool}(X) \in \mathbb{R}^{h}$$

具体实现：可用一个可学习的 query 向量 $q_{pool} \in \mathbb{R}^h$ 对 $X$ 做单头注意力：
$$a_j = \frac{\exp(q_{pool}^\top X_j / \sqrt{h})}{\sum_{j'} \exp(q_{pool}^\top X_{j'} / \sqrt{h})}, \quad \bar{x} = \sum_j a_j X_j$$

或更简单地使用均值池化 $\bar{x} = \frac{1}{n}\sum_j X_j$（但注意力池化更灵活）。

**Step 2 — 条件化交叉注意力**

$$Q_\Psi = \text{CrossAttn}(M + \bar{x},\ E,\ E) \in \mathbb{R}^{m \times h}$$

- **Query**：$M + \bar{x}$（元查询广播加上视频摘要，使每个元查询感知当前视频内容）
- **Key / Value**：密码本 $E$
- 标准多头交叉注意力，全程可微，无 top-k 操作

**关键点**：
- 广播机制：$\bar{x}$ 是一个向量，加到 $M$ 的每一行上 → 所有元查询共享同一视频条件
- 不同视频产生不同的 $Q_\Psi$，但都从同一密码本 $E$ 中路由，保证跨视频语义一致性
- 计算量轻：$m$ 个 query × $K$ 个 key 的注意力矩阵，$m$=32, $K$≤256

### 1.3 输出

$Q_\Psi \in \mathbb{R}^{m \times h}$：$m$ 个数据感知的伪查询 token，将在 Phase B 与视频 token 联合编码。

---

## 2. Phase B：联合编码 + 显著性计算

### 2.1 前向流程

**Step 1 — 拼接输入**

将伪查询 $Q_\Psi$ 与视频 token $X$ 拼接：

$$\text{Input} = [X,\ Q_\Psi] \in \mathbb{R}^{(n+m) \times h}$$

**Step 2 — Transformer 编码**

送入共享的视觉编码器（如 ViT 的后续层或独立 Transformer 编码器）：

$$[Z_X^{(L)},\ Z_\Psi^{(L)}] = F_{enc}([X,\ Q_\Psi];\ \theta)$$

- $Z_X^{(L)} \in \mathbb{R}^{n \times h}$：编码后视频 token
- $Z_\Psi^{(L)} \in \mathbb{R}^{m \times h}$：编码后伪查询

伪查询通过自注意力与视频 token 交互，吸收视频内容信息。

**Step 3 — 显著性分数计算**

利用最后一层注意力中伪查询对视频 token 的注意力权重：

$$\alpha_j = \frac{1}{|\Psi| \cdot H} \sum_{i \in \Psi} \sum_{\eta=1}^{H} \text{Attn}_{i \to j}^{(L, \eta)}$$

- 对所有伪查询位置 $i$、所有注意力头 $\eta$ 取平均
- $\alpha_j$ 衡量视频 token $j$ 被伪查询"关注"的程度 → 语义显著性
- $\alpha \in \mathbb{R}^n$，所有分量之和不一定为 1（因为是多头多 query 平均）

### 2.2 输出

- $Z_X^{(L)}$：编码后视频 token（带上下文信息）
- $Z_\Psi^{(L)}$：编码后伪查询 → **直接作为聚类中心 $\mu_k$**
- $\alpha$：视频 token 显著性权重

---

## 3. Phase C：聚类分配（Soft-to-Hard 退火）

### 3.1 聚类中心定义

$$\{\mu_k\}_{k=1}^m = Z_\Psi^{(L)}$$

即第 $k$ 个聚类中心就是第 $k$ 个编码后伪查询。不需要额外选择步骤。

### 3.2 分配权重计算

对每个视频 token $j$ 和聚类 $k$，计算 softmax 分配：

$$w_{kj} = \frac{\exp\bigl(\cos(Z_{X,j}^{(L)},\ \mu_k) / \tau\bigr)}{\sum_{k'=1}^{m} \exp\bigl(\cos(Z_{X,j}^{(L)},\ \mu_{k'}) / \tau\bigr)}$$

- $\cos(\cdot, \cdot)$：余弦相似度
- $\tau$：温度参数，控制分配硬度

### 3.3 温度退火策略

| 阶段 | 温度 $\tau$ | 分配行为 |
|------|-------------|----------|
| 训练初期 | 1.0 | 软分配，梯度平滑流动 |
| 训练中期 | 线性退火 1.0 → 0.1 | 逐渐收紧 |
| 训练后期 | 0.1 | 接近硬分配 |
| 推理 | $\tau \to 0$（等效 argmax） | 完全硬分配 |

退火调度建议：

$$\tau(t) = \tau_{start} - (\tau_{start} - \tau_{end}) \cdot \frac{t}{T}$$

其中 $t$ 为当前 step，$T$ 为总 step 数，$\tau_{start}=1.0$，$\tau_{end}=0.1$。

### 3.4 硬分配（推理时）

$$G_k = \{j\ |\ k = \arg\max_{k'} \cos(Z_{X,j}^{(L)},\ \mu_{k'})\}$$

每个视频 token 唯一归属一个聚类。

### 3.5 软硬切换的处理

训练时全程使用软分配（$w_{kj}$ 连续），保证梯度流。推理时切换硬分配（argmax），效率等同原 AGC。

---

## 4. Phase D：显著性加权聚合 + 残差锚定

### 4.1 聚合池化

**软分配下**（训练时）：

$$\text{AggPool}_k = \frac{\sum_{j=1}^{n} w_{kj} \cdot \alpha_j \cdot Z_{X,j}^{(L)}}{\sum_{j=1}^{n} w_{kj} \cdot \alpha_j + \epsilon}$$

- $w_{kj}$：聚类分配权重（软）
- $\alpha_j$：显著性权重
- $\epsilon$：数值稳定项（如 1e-8）

**硬分配下**（推理时）：

$$\text{AggPool}_k = \frac{\sum_{j \in G_k} \alpha_j \cdot Z_{X,j}^{(L)}}{\sum_{j \in G_k} \alpha_j + \epsilon}$$

### 4.2 残差锚定

$$c_k = (1 - \lambda) \cdot \text{AggPool}_k + \lambda \cdot Z_{\Psi,k}^{(L)}$$

- $\lambda$：可学习标量，初始化为 0.1
- **动机**：当某聚类被分配到极少 token 时，纯聚合不稳定（分母接近零），混入伪查询的语义表示作为"锚点"保证输出有意义
- $\lambda = 0$ 时退化为标准加权聚合（无残差锚定）

### 4.3 输出

$\{c_k\}_{k=1}^m \in \mathbb{R}^{m \times h}$：$m$ 个压缩后的视频表示 token。

- 原始 $n$ 个视频 token 被压缩为 $m$ 个（$m$=32 时压缩率约 97.5%）
- 每个 $c_k$ 代表一个语义聚类的加权汇总

---

## 5. Phase E：检索匹配与损失优化

### 5.1 检索匹配 — MaxSim

给定文本查询 $q$ 经文本编码器得到 token 表示 $\{q_i\}$：

$$\text{MaxSim}(c, q) = \sum_{i} \max_{k} \cos(c_k, q_i)$$

- 对每个文本 token，找到与之最相似的视频聚类表示
- 求和得到总匹配分数

### 5.2 核心损失：$L_{retrieval}$

$$L_{retrieval} = \text{InfoNCE}(\text{MaxSim}(c, q))$$

标准 InfoNCE，batch 内负采样：

$$L_{retrieval} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{MaxSim}(c_i, q_i) / \tau_r)}{\sum_{j=1}^{B} \exp(\text{MaxSim}(c_i, q_j) / \tau_r)}$$

- $B$：batch size
- $\tau_r$：InfoNCE 温度参数（与聚类温度 $\tau$ 不同）
- 正样本：匹配的视频-文本对
- 负样本：同 batch 内非匹配对

可选双向 InfoNCE（video→text + text→video 对称）。

### 5.3 辅助损失分析与建议

#### $L_{div}$（密码本多样性损失）

$$L_{div} = \frac{1}{K(K-1)} \sum_{i \neq j} \max\bigl(0,\ \cos(E_i, E_j) - \tau_{div}\bigr)$$

- 作用：防止密码本条目坍塌（多个条目趋向相同向量）
- 建议系数 $\beta_1 = 0.1$，阈值 $\tau_{div} = 0.5$

**是否必须？**

不确定。密码本坍塌是否会在当前配置下发生，取决于：
- 密码本大小 $K$ 与伪查询数 $m$ 的比值（$K/m$ = 4~8）
- 数据多样性
- 学习率与优化动态

**建议策略**：

1. **初始训练不加入** $L_{div}$（即 $\beta_1 = 0$）
2. **在评估/验证阶段监控密码本多样性指标**：
   - 两两余弦相似度矩阵 $S_{ij} = \cos(E_i, E_j)$
   - 报告均值、最大值、以及超过 $\tau_{div}=0.5$ 的比例
   - 有效秩（effective rank）：对 $E$ 做 SVD，计算归一化奇异值的熵
3. **触发条件**：若观测到 $\max_{i \neq j} \cos(E_i, E_j) > 0.8$ 或有效秩大幅低于 $K$，再开启 $L_{div}$
4. 这样避免引入不必要的正则项干扰主任务优化

#### $L_{bal}$（负载均衡损失）

$$L_{bal} = K \cdot \sum_{k=1}^{K} f_k \cdot p_k$$

其中（注意：此处的 $K$ 应为聚类数 $m$，而非密码本大小，因为负载均衡针对的是聚类分配）：

$$f_k = \frac{1}{n} \sum_{j=1}^{n} \mathbb{1}[\arg\max_{k'} w_{k'j} = k]$$
$$p_k = \frac{1}{n} \sum_{j=1}^{n} w_{kj}$$

- $f_k$：第 $k$ 个聚类被硬分配到的 token 比例
- $p_k$：第 $k$ 个聚类的软分配权重均值
- 建议系数 $\beta_2 = 0.01$

**是否必须？**

同样不确定。负载不均可能出现也可能不出现，取决于：
- 视频语义分布
- 伪查询初始化
- 温度退火速度

**建议策略**：

1. **初始训练不加入** $L_{bal}$（即 $\beta_2 = 0$）
2. **在评估/验证阶段监控聚类负载均衡指标**：
   - 每个聚类分配到的 token 数量分布 $\{|G_k|\}_{k=1}^m$
   - Gini 系数：衡量分配不均匀度（0=完全均匀，1=完全集中）
   - 变异系数（CV）：$\text{std}(\{|G_k|\}) / \text{mean}(\{|G_k|\})$
   - "死聚类"数量：$|G_k| = 0$ 的聚类个数
3. **触发条件**：若 Gini > 0.3 或 死聚类 > 10% 的 $m$，再开启 $L_{bal}$

### 5.4 完整损失函数

$$L = L_{retrieval} + \beta_1 L_{div} + \beta_2 L_{bal}$$

**推荐训练策略**：

| 阶段 | $\beta_1$ | $\beta_2$ | 说明 |
|------|-----------|-----------|------|
| 第一阶段（探索） | 0 | 0 | 只用检索损失，观察自然行为 |
| 第二阶段（按需引入） | 0.1（若需要） | 0.01（若需要） | 根据监控指标决定是否开启 |

---

## 6. 完整训练流程

### 6.1 单步前向传播

```
输入：视频帧 → 视觉 Token X ∈ R^{n×h}，文本 → 文本编码 {q_i}

1. x̄ = AttnPool(X)                            # 视频摘要
2. Q_Ψ = CrossAttn(M + x̄, E, E)              # 伪查询生成 [Phase A]
3. [Z_X, Z_Ψ] = F_enc([X, Q_Ψ]; θ)           # 联合编码 [Phase B]
4. α = mean_over_heads_and_queries(Attn_last)  # 显著性分数 [Phase B]
5. μ = Z_Ψ                                     # 聚类中心 = 编码后伪查询 [Phase C]
6. w = softmax(cos(Z_X, μ) / τ)               # 软分配 [Phase C]
7. AggPool_k = Σ(w_kj · α_j · Z_Xj) / Σ(w_kj · α_j)  # 聚合 [Phase D]
8. c_k = (1-λ) · AggPool_k + λ · Z_Ψk         # 残差锚定 [Phase D]
9. score = MaxSim(c, q)                         # 检索打分 [Phase E]
10. L = InfoNCE(score) [+ β1·L_div + β2·L_bal] # 损失 [Phase E]
```

### 6.2 反向传播梯度流路径

```
L_retrieval
  ↓
MaxSim(c, q)
  ↓
c_k = (1-λ)·AggPool_k + λ·Z_Ψk
  ↓                        ↓
  ├── ∂L/∂AggPool_k       ├── ∂L/∂Z_Ψk → 编码器 θ → 伪查询 Q_Ψ → 密码本 E, 元查询 M
  │     ↓
  │   ∂L/∂w_kj (通过 softmax, 可微)
  │     ↓
  │   ∂L/∂μ_k = ∂L/∂Z_Ψk (同上)
  │   ∂L/∂Z_Xj → 编码器 θ
  │     ↓
  │   ∂L/∂α_j (通过注意力权重, 可微)
  │     ↓
  └── 编码器 θ → Q_Ψ → CrossAttn → E, M, x̄ → AttnPool → X
```

**全程可微**：没有 top-k、argmax 或其他不连续操作在训练路径中。

### 6.3 可学习参数清单

| 参数 | 形状 | 初始化建议 |
|------|------|------------|
| 密码本 $E$ | $K \times h$ | 正态分布 $\mathcal{N}(0, 1/\sqrt{h})$ |
| 元查询 $M$ | $m \times h$ | 正态分布 $\mathcal{N}(0, 1/\sqrt{h})$ |
| 残差系数 $\lambda$ | 标量 | 0.1 |
| 注意力池化 query $q_{pool}$（若使用） | $h$ | 正态分布 |
| 交叉注意力层参数 $W_Q, W_K, W_V, W_O$ | 标准 | Xavier/Kaiming |
| 编码器 $F_{enc}$ 参数 $\theta$ | 预训练 | 从 CLIP ViT 加载 |

### 6.4 温度退火实现

每个 training step 更新：

$$\tau = \max\bigl(\tau_{end},\ \tau_{start} - (\tau_{start} - \tau_{end}) \cdot \frac{\text{step}}{\text{total\_steps}}\bigr)$$

不是可学习参数，是调度器控制的超参数。

---

## 7. 推理流程

```
输入：视频帧 → X

1. x̄ = AttnPool(X)
2. Q_Ψ = CrossAttn(M + x̄, E, E)
3. [Z_X, Z_Ψ] = F_enc([X, Q_Ψ]; θ)
4. α = 显著性分数
5. G_k = {j | k = argmax_k' cos(Z_Xj, Z_Ψk')}   # 硬分配
6. AggPool_k = Σ_{j∈G_k} α_j·Z_Xj / Σ_{j∈G_k} α_j
7. c_k = (1-λ)·AggPool_k + λ·Z_Ψk
8. 存储 {c_k} 作为视频索引
```

查询时：$\text{score} = \text{MaxSim}(c, q)$，按分数排序返回结果。

额外开销 vs 原 AGC：仅多一次交叉注意力（Step 2），$m \times K$ 规模，可忽略不计。

---

## 8. 评估阶段监控指标（用于决定是否引入辅助损失）

### 8.1 密码本健康度（决定是否需要 $L_{div}$）

| 指标 | 计算方式 | 健康阈值 | 报警阈值 |
|------|----------|----------|----------|
| 两两余弦相似度均值 | $\frac{1}{K(K-1)}\sum_{i \neq j}\cos(E_i, E_j)$ | < 0.3 | > 0.5 |
| 两两余弦最大值 | $\max_{i \neq j}\cos(E_i, E_j)$ | < 0.6 | > 0.8 |
| 有效秩 | $\exp(-\sum_i \hat{\sigma}_i \log \hat{\sigma}_i)$，$\hat{\sigma}$ 为归一化奇异值 | > 0.5$K$ | < 0.3$K$ |
| 高相似度对比例 | $\cos > 0.5$ 的 pair 占比 | < 5% | > 20% |

### 8.2 聚类均衡度（决定是否需要 $L_{bal}$）

| 指标 | 计算方式 | 健康阈值 | 报警阈值 |
|------|----------|----------|----------|
| Gini 系数 | 标准 Gini 公式 on $\{|G_k|\}$ | < 0.2 | > 0.3 |
| 变异系数 (CV) | $\text{std}/\text{mean}$ of $\{|G_k|\}$ | < 0.5 | > 1.0 |
| 死聚类比例 | $\frac{|\{k : |G_k|=0\}|}{m}$ | 0% | > 10% |
| 最大聚类占比 | $\max_k |G_k| / n$ | < $3/m$ | > $5/m$ |

### 8.3 监控频率

- 每 N 个 epoch（如 N=1~5）在验证集上计算上述指标
- 记录趋势：若指标持续恶化，才引入对应辅助损失
- 可视化：密码本 t-SNE / PCA + 聚类分配热力图

---

## 9. 超参数汇总

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| $m$（伪查询数/聚类数） | 32 | ~97.5% 压缩率 |
| $K$（密码本大小） | 128~256 | $4\text{-}8 \times m$ |
| $h$（特征维度） | 取决于视觉编码器 | 如 CLIP ViT-B/32 为 512 |
| $\tau_{start}$（聚类初始温度） | 1.0 | |
| $\tau_{end}$（聚类终止温度） | 0.1 | |
| $\tau_r$（InfoNCE 温度） | 待定 | 通常 0.07 或可学习 |
| $\lambda$（残差锚定系数） | 0.1（可学习） | |
| $\beta_1$（$L_{div}$ 系数） | 0（按需开启，建议值 0.1） | |
| $\beta_2$（$L_{bal}$ 系数） | 0（按需开启，建议值 0.01） | |
| $\tau_{div}$（多样性阈值） | 0.5 | |

---

## 10. 消融实验计划（后续验证用）

| 实验 | 修改 | 验证目标 |
|------|------|----------|
| 无密码本 | 去掉 $E$，$Q_\Psi = M + \bar{x}$ | 密码本的价值 |
| 无残差锚定 | $\lambda = 0$ 固定 | 残差锚定的稳定性 |
| 无退火 | $\tau$ 固定为 1.0 或 0.1 | 退火策略的贡献 |
| 无显著性 | $\alpha_j = 1\ \forall j$ | 显著性权重的作用 |
| 硬分配训练 | 训练时也用 argmax（straight-through） | soft-to-hard 的优势 |
| 加 $L_{div}$ | $\beta_1 = 0.1$ | 多样性损失的影响 |
| 加 $L_{bal}$ | $\beta_2 = 0.01$ | 负载均衡损失的影响 |
