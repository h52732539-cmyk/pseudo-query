# VIB-based Pseudo-Query Video Retrieval

基于**变分信息瓶颈（VIB）**的伪查询视频检索 pipeline。核心思路：利用 VLM 生成的密集伪查询描述，通过全局聚类构建语义原型库，再经变分压缩网络映射为概率分布，最终在检索时通过 Cross-Attention 激活与不确定性感知评分完成 Text→Video 检索。

## 项目结构

```
pseudo query/
├── configs/
│   └── default.yaml              # 超参数配置文件
├── data/
│   ├── __init__.py
│   ├── preprocess.py             # 数据加载：MSR_VTT.json + narration.json 解析、split 划分
│   └── dataset.py                # PyTorch Dataset / DataLoader / collate_fn
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py           # CLIP Text Encoder 封装（冻结，自动下载）
│   ├── prototype.py              # 原型库 P (K×d)：K-Means 初始化 + nn.Parameter
│   ├── variational_encoder.py    # 变分压缩网络：Prototype-guided Attention → (μ, σ²)
│   ├── query_assembly.py         # Cross-Attention 查询激活 + Max-Pooling 组装
│   ├── scoring.py                # 不确定性感知评分、InfoNCE loss、KL divergence
│   └── pipeline.py               # 完整 VIBPseudoQueryModel（组装所有模块）
├── scripts/
│   ├── build_prototypes.py       # 离线脚本：全局 token 提取 → K-Means → 保存原型
│   └── smoke_test.py             # 快速验证脚本：端到端 pipeline 正确性检查
├── checkpoints/                  # 模型检查点保存目录
├── docs/
│   ├── plan.md                   # 完整工程设计方案
│   └── pq_1.md                   # 理论设计文档
├── MSR_VTT.json                  # MSR-VTT 标准元数据 + ground-truth 标注
├── MSRVTT_narration.json         # VLM 生成的密集伪查询描述（10K 视频）
├── train.py                      # 训练主循环
├── evaluate.py                   # 评估脚本：R@1/5/10, MdR, MnR
├── requirements.txt              # Python 依赖
└── README.md
```

## 环境配置

```bash
conda create -n pq python=3.10
conda activate pq
pip install -r requirements.txt
```

依赖列表：
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.2.0`
- `pyyaml >= 6.0`
- `tqdm >= 4.65.0`

> CLIP 模型（`openai/clip-vit-base-patch32`）首次运行会自动从 HuggingFace 下载。若网络受限，代码会自动使用 `hf-mirror.com` 镜像。也可手动设置：`export HF_ENDPOINT=https://hf-mirror.com`

## 数据准备

确保项目根目录下存在以下两个文件：

| 文件 | 说明 |
|------|------|
| `MSR_VTT.json` | MSR-VTT 标准数据集元数据，包含 `annotations` 字段（每条含 `image_id` 和 `caption`） |
| `MSRVTT_narration.json` | VLM 密集伪查询，每个视频包含 `video_file` 和 `caption_1` ~ `caption_N` |

数据划分遵循 MSR-VTT 标准：

| Split | 范围 | 数量 |
|-------|------|------|
| Train | video0 ~ video6512 | 6,513 |
| Val   | video6513 ~ video7009 | 497 |
| Test  | video7010 ~ video9999 | 2,990 |

## 使用指南

### Step 1: 快速验证（Smoke Test）

首次使用建议先运行 smoke test，验证环境和 pipeline 正确性：

```bash
python scripts/smoke_test.py
```

该脚本会依次测试：数据加载 → CLIP 编码 → 小规模 K-Means → 模型前向/反向传播 → 评估推理。看到 `ALL TESTS PASSED` 表示一切正常。

### Step 2: 构建原型库（离线，一次性）

从所有伪查询中提取 token 级特征，运行 Mini-Batch K-Means 聚类生成原型库：

```bash
python scripts/build_prototypes.py --config configs/default.yaml
```

可选参数：
- `--batch_size 256`：编码时的 batch size（根据 GPU 显存调整）
- `--device cuda`：使用 GPU 加速编码

产出文件：`checkpoints/prototypes.pt`（形状 `K × 512`，默认 K=512）

> 全量 10K 视频约 1200 万 tokens，在 GPU 上编码约需 10~30 分钟。

### Step 3: 训练

```bash
python train.py --config configs/default.yaml
```

可选参数：
- `--device cuda`：指定设备
- `--resume checkpoints/latest_model.pt`：从断点恢复训练

训练行为：
- **β-annealing**：KL 权重从 `1e-4` 线性增长至 `1e-2`，避免 posterior collapse
- **学习率**：AdamW + warmup + cosine decay
- **混合精度**：GPU 上自动启用 FP16
- **检查点**：每个 epoch 保存 `latest_model.pt`，验证集最优保存 `best_model.pt`

训练日志示例：
```
Epoch 1/30 [Train]: loss=4.8532 task=4.8531 kl=1.0023 beta=0.000100 tau=0.0700
Epoch 1: train_loss=4.2156, val_loss=3.8901
  ✓ Best model saved (val_loss=3.8901)
```

### Step 4: 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

可选参数：
- `--test_mode full`：全量 test set（2,990 视频，每视频所有 GT caption 作为查询）
- `--test_mode 1k-A`：MSR-VTT 1K-A split（video7010~video8009，每视频取第一条 GT caption）
- `--test_mode both`：同时运行两种评估
- `--device cuda`：指定设备

输出指标：

| 指标 | 说明 |
|------|------|
| R@1 | Top-1 召回率 (%) |
| R@5 | Top-5 召回率 (%) |
| R@10 | Top-10 召回率 (%) |
| MdR | 中位排名（越小越好） |
| MnR | 平均排名（越小越好） |

## 核心超参数

在 `configs/default.yaml` 中配置：

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `prototype.num_prototypes` | 512 | 原型数 K |
| `model.aggregation` | `attention` | Token 聚合方式（`attention` / `mean`） |
| `model.temperature_init` | 0.07 | 温度 τ 初始值（可学习） |
| `model.variance_penalty` | 0.1 | 方差惩罚系数 λ |
| `training.batch_size` | 128 | 对比学习需大 batch |
| `training.lr` | 1e-4 | 学习率 |
| `training.epochs` | 30 | 训练轮数 |
| `training.beta_start` | 1e-4 | β-annealing 起始值 |
| `training.beta_end` | 1e-2 | β-annealing 终止值 |

## 方法概述

```
┌──────────────────────────────────────────────────────────┐
│                    训练阶段                               │
│                                                          │
│  视频伪查询 captions ──→ CLIP Text Encoder (冻结)         │
│       ↓                                                  │
│  Token 特征 {t₁,...,tₘ} ──→ Prototype-guided Attention   │
│       ↓                                                  │
│  聚合特征 h ──→ Mean Head → μ ∈ ℝᴷ                       │
│              └→ Var Head  → σ² ∈ ℝᴷ (Softplus)          │
│       ↓                                                  │
│  重参数化采样 m = μ + ε·σ                                 │
│                                                          │
│  GT 查询文本 ──→ CLIP Text Encoder → Cross-Attention      │
│       ↓            with 原型库 P                          │
│  Max-Pooling → 查询激活 s_T ∈ ℝᴷ                         │
│       ↓                                                  │
│  Score = s_Tᵀ m - λ · s_Tᵀ σ²   (不确定性感知评分)        │
│       ↓                                                  │
│  Loss = InfoNCE(双向) + β · KL(N(μ,σ²) ‖ N(0,I))        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    检索阶段                               │
│                                                          │
│  输入查询 → encode_query → s_T                            │
│  所有视频 → encode_video → (μ, σ²)   [可离线预计算]       │
│  Score(T, Vᵢ) = s_Tᵀ μᵢ - λ Σₖ s_{T,k}·σ²_{i,k}       │
│  按 Score 降序排列 → Top-K 检索结果                        │
└──────────────────────────────────────────────────────────┘
```

## 可训练模块

| 模块 | 参数量级 | 说明 |
|------|---------|------|
| 原型库 P | K × d | K-Means 初始化，训练中微调 |
| Prototype-guided Attention | ~4d² | 多头注意力聚合 |
| Mean Head | d × K | 线性映射 → μ |
| Var Head | d × K | 线性映射 + Softplus → σ² |
| 温度 τ | 1 | 可学习标量 |
| **CLIP Text Encoder** | **~63M** | **冻结，不参与训练** |
