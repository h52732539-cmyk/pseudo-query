# Pseudo-Query Video Retrieval

基于**逐原型对比学习 + SwAV 在线原型学习**的伪查询视频检索 pipeline。核心思路：利用 VLM 生成的密集伪查询描述，通过在线学习的语义原型库进行 Prototype-guided Attention 聚合，再经逐原型对比损失和 SwAV 交叉预测训练，最终在检索时通过 cosine similarity 完成 Text→Video 检索。

支持两种方案分支：
- **方案B (SwAV)**：原型为 `nn.Parameter`，端到端梯度更新 + Sinkhorn 等分约束
- **方案C (Hybrid)**：梯度原型 + EMA 影子副本（稳定视频编码）+ 死原型自动重初始化

## 项目结构

```
pseudo-query/
├── configs/
│   └── default.yaml              # 超参数配置文件
├── data/
│   ├── __init__.py
│   ├── preprocess.py             # 数据加载：MSR_VTT.json + narration.json 解析、split 划分
│   └── dataset.py                # Multi-View Dataset / DataLoader / collate_fn
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py           # CLIP Text Encoder 封装（冻结，自动下载）
│   ├── prototype.py              # 原型库：PrototypeLibrary / EMAPrototypeLibrary / Sinkhorn / SwAVLoss
│   ├── video_encoder.py          # 视频编码器：Prototype-guided Attention → (h, μ)
│   ├── query_assembly.py         # Cross-Attention 查询激活 + Max-Pooling → (s_T, q̃)
│   ├── scoring.py                # 逐原型对比损失 + cosine 推理评分
│   ├── pipeline_swav.py          # 方案B 完整 Pipeline（SwAV Sinkhorn）
│   └── pipeline_hybrid.py        # 方案C 完整 Pipeline（Hybrid EMA + SwAV）
├── scripts/
│   └── smoke_test.py             # 快速验证脚本：SwAV + Hybrid 双 pipeline 正确性检查
├── checkpoints/                  # 模型检查点保存目录
├── docs/
│   ├── pipeline.md               # 完整流程与数学推导
│   ├── plan.md                   # 工程设计方案
│   └── pq_1.md                   # 理论设计文档
├── MSR_VTT.json                  # MSR-VTT 标准元数据 + ground-truth 标注
├── MSRVTT_narration.json         # VLM 生成的密集伪查询描述（10K 视频）
├── train.py                      # 训练主循环（多视图 + SwAV）
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

该脚本会依次测试：数据加载 → CLIP 编码 → Multi-View 数据集 → SwAV Pipeline 前向/反向 → Hybrid Pipeline 前向/反向 → 评估推理。看到 `ALL SMOKE TESTS PASSED` 表示一切正常。

### Step 2: 训练

无需离线构建原型库，原型完全在线学习。直接开始训练：

```bash
# 方案B: SwAV（默认）
python train.py --config configs/default.yaml

# 方案C: Hybrid EMA + SwAV
# 修改 configs/default.yaml 中 scheme: "hybrid"，然后运行：
python train.py --config configs/default.yaml
```

可选参数：
- `--device cuda`：指定设备
- `--resume checkpoints/latest_model.pt`：从断点恢复训练

训练行为：
- **多视图训练**：每个视频的伪查询随机分为两组，SwAV 交叉预测促进原型多样性
- **损失函数**：L_match（逐原型对比） + α · L_swav（交叉预测）
- **学习率**：AdamW + warmup + cosine decay
- **混合精度**：GPU 上自动启用 FP16
- **检查点**：每个 epoch 保存 `latest_model.pt`，验证集最优保存 `best_model.pt`
- **方案C 附加**：每步执行 EMA 更新 + 死原型检测与重初始化

训练日志示例：
```
Epoch 1/30 [Train]: loss=4.8532 match=4.2156 swav=1.2752 tau=0.0700
Epoch 1: train_loss=4.2156, val_loss=3.8901
  ✓ Best model saved (val_loss=3.8901)
```

### Step 3: 评估

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
| `scheme` | `swav` | 方案选择（`swav` / `hybrid`） |
| `prototype.num_prototypes` | 512 | 原型数 K |
| `model.aggregation` | `attention` | Token 聚合方式（`attention` / `mean`） |
| `model.temperature_init` | 0.07 | 温度 τ 初始值（可学习） |
| `swav.sinkhorn_eps` | 0.05 | Sinkhorn 正则化系数 |
| `swav.temperature` | 0.1 | SwAV 交叉预测温度 |
| `training.swav_alpha` | 0.5 | L_swav 损失权重 α |
| `training.batch_size` | 128 | 对比学习需大 batch |
| `training.lr` | 1e-4 | 学习率 |
| `training.epochs` | 30 | 训练轮数 |
| `ema.decay` | 0.999 | EMA 衰减系数（方案C） |
| `ema.dead_proto_threshold` | 100 | 死原型判定步数（方案C） |

## 方法概述

```
┌──────────────────────────────────────────────────────────┐
│                    训练阶段（多视图）                      │
│                                                          │
│  视频伪查询 → 随机二分为 View1, View2                     │
│       ↓                     ↓                            │
│  CLIP Token Encoder    CLIP Token Encoder  (冻结)        │
│       ↓                     ↓                            │
│  Prototype-guided      Prototype-guided                  │
│  Attention             Attention                         │
│       ↓                     ↓                            │
│  h₁, μ₁ ∈ ℝᴷ          h₂, μ₂ ∈ ℝᴷ                     │
│       └──── SwAV ────────────┘                           │
│              (Sinkhorn 交叉预测)                          │
│                                                          │
│  GT 查询文本 → CLIP → Cross-Attention + MaxPool           │
│       ↓                                                  │
│  s_T ∈ ℝᴷ (查询激活)    q̃ ∈ ℝᴷˣᵈ (逐原型查询语义)       │
│       ↓                                                  │
│  L_match = 逐原型对比损失(h_avg, q̃, s_T)                 │
│  L_total = L_match + α · L_swav                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    检索阶段                               │
│                                                          │
│  输入查询 → get_query_repr → s_T ∈ ℝᴷ                    │
│  所有视频 → get_video_repr → μ ∈ ℝᴷ   [可离线预计算]      │
│  Score(T, Vᵢ) = cosine(s_T, μᵢ) / τ                     │
│  按 Score 降序排列 → Top-K 检索结果                        │
└──────────────────────────────────────────────────────────┘
```

## 可训练模块

| 模块 | 参数量级 | 说明 |
|------|---------|------|
| 原型库 P | K × d | Xavier 随机初始化，在线学习 |
| Prototype-guided Attention | ~4d² | 多头注意力聚合 |
| Mean Head | d → 1 | 线性映射 → μ |
| 温度 τ | 1 | 可学习标量 |
| EMA 影子原型 (方案C) | K × d | buffer，不参与梯度 |
| **CLIP Text Encoder** | **~63M** | **冻结，不参与训练** |
