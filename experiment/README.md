# AGC 伪查询实验

## 目标

验证 **VLM 生成的伪查询（MSRVTT_narration）能否完全替代视频帧数据**，通过将每个视频的伪查询拼接为纯文本文档，使用 [OmniColPress](https://github.com/hanxiangqin/omni-col-press) 框架的 **AGC（Attention-Guided Clustering）** 完整流程进行 text-to-text 检索，在 MSR-VTT 测试集上评估召回率。

### 核心对比

| 方法 | 文档输入 | 编码器 | 压缩 | R@1 | R@10 | nDCG@10 |
|------|---------|--------|------|-----|------|---------|
| AGC (论文 Table 4) | 24帧视频 | Qwen2.5-VL-3B | 32 tok | 56.9 | 87.0 | 71.5 |
| **AGC (本实验)** | **伪查询文本** | **Qwen2.5-VL-3B** | **32 tok** | **?** | **?** | **?** |

## 参考论文

> **Multi-Vector Index Compression in Any Modality**
> Hanxiang Qin, Alexander Martin, Rohan Jha, Chunsheng Zuo, Reno Kriz, Benjamin Van Durme
> arXiv:2602.21202 — https://arxiv.org/abs/2602.21202v1

AGC 方法三步走：
1. **Attention-based Centroid Selection** — 可学习的 Universal Query tokens 通过注意力机制识别文档中语义最显著的 token 作为聚类中心
2. **Hard Clustering** — 将所有其他 token 分配到最近的中心，消除冗余
3. **Weighted Aggregation** — 按显著性权重聚合每个簇内的 token，生成压缩后的多向量表示

检索使用 ColBERT-style **Late Interaction (MaxSim)**：
$$s(q,d) = \sum_{i=1}^{n_q} \max_{1 \le j \le m} \langle \mathbf{q}_i, \mathbf{c}_j \rangle$$

## 文件结构

```
experiment/
├── README.md               # 本文件
├── prepare_data.py         # 数据准备脚本
├── validate_data.py        # 数据验证脚本
├── configs/
│   ├── train_data.yaml     # OmniColPress 训练数据集配置
│   └── train_data_with_val.yaml
├── scripts/
│   ├── run_train.sh        # 训练启动脚本
│   ├── run_eval.sh         # 评估启动脚本 (索引构建 + 检索)
│   └── run_ablation.sh     # 消融实验 (不同压缩预算)
├── data/                   # (运行 prepare_data.py 后生成)
│   ├── train_corpus.jsonl  # 训练语料 (7010 docs)
│   ├── test_corpus.jsonl   # 测试语料 (2990 docs)
│   ├── train.jsonl         # 训练 query-doc 对 (~140k pairs)
│   ├── test_queries.csv    # 测试查询 (1000 queries, MSR-VTT 1k-A)
│   ├── test_qrels.jsonl    # 相关性标注 (1000 qrels)
│   └── stats.json          # 数据统计
└── outputs/                # (训练/评估后生成)
    └── agc_pq_b32/
        ├── (model checkpoint)
        ├── index_multivec/
        └── results/
```

## 运行步骤

### 前置条件

1. 克隆 OmniColPress：
```bash
git clone https://github.com/hanxiangqin/omni-col-press.git
```

2. 安装依赖：
```bash
conda create -n omnicolpress python=3.11
conda activate omnicolpress
pip install torch torchvision
pip install transformers deepspeed ninja peft librosa numpy
pip install qwen-vl-utils[decord] qwen-omni-utils[decord] -U
pip install flash-attn --no-build-isolation
pip install fast-plaid
conda install -c pytorch -c nvidia faiss-gpu
```

### Step 0: 准备数据

```bash
cd /path/to/pseudo-query
python experiment/prepare_data.py
python experiment/validate_data.py   # 验证数据格式
```

### Step 1: 训练

```bash
export OMNI_COL_PRESS_DIR=/path/to/omni-col-press
export NUM_GPUS=4

# 默认 budget=32
bash experiment/scripts/run_train.sh

# 或指定不同 budget
NUM_REPR_VECTORS=128 NUM_APPENDING_TOKENS=128 bash experiment/scripts/run_train.sh
```

### Step 2: 评估

```bash
bash experiment/scripts/run_eval.sh
```

### 消融实验（可选）

```bash
# 依次跑 budget=5, 32, 128
bash experiment/scripts/run_ablation.sh
```

## 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 编码器 | Qwen2.5-VL-3B | 与论文一致，公平对比 |
| 文档侧模态 | text-only | 实验核心：用伪查询替代视频 |
| 伪查询拼接 | 所有 caption 句号分隔 | 保留全部信息，让 AGC 自动学习筛选 |
| 测试集 | MSR-VTT 1k-A | 1000 query-video 对，与论文评估一致 |
| Hard negatives | 无（in-batch） | 简化实验，与论文 setup 对齐 |
| passage_max_len | 1024 | 伪查询拼接后 p95=1173 词，1024 token 覆盖大部分文档 |

## 关键参数

```
# AGC 压缩
--pooling select
--num_repr_vectors 32       # 压缩后的 token 数
--num_appending_token 32    # Universal Query tokens 数量
--use_parametric_appending_tokens
--use_cluster_pooling
--use_attn_weight_cluster_pooling

# 模态控制 (关键：禁用视频)
--encode_modalities '{"default": {"text": true, "image": false, "video": false, "audio": false}}'

# 训练超参 (对齐论文)
--learning_rate 1e-5
--num_train_epochs 2
--per_device_train_batch_size 28
--gradient_accumulation_steps 4
--bf16
```

## 预期结果与分析

### 如果伪查询接近视频帧效果 (e.g. R@1 > 50)
→ 伪查询可以作为视频的有效文本替代，大幅降低存储和计算成本

### 如果伪查询显著低于视频帧 (e.g. R@1 < 40)
→ 伪查询丢失了视频的视觉细节，需要考虑更好的文本生成策略或混合方案

### 需要关注的点
1. **passage_max_len 的影响**：伪查询拼接后约 150-300 words，截断是否丢失关键信息
2. **AGC 是否有效压缩文本冗余**：伪查询之间有大量语义重复，AGC 的注意力机制应能识别并去冗余
3. **与论文 baseline 的对比**：论文的 uncompressed baseline（1318 tok）→ R@1=55.7，AGC(32)=56.9（压缩后反而更好）
