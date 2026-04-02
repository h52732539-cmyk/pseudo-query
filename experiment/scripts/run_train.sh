#!/bin/bash
# ══════════════════════════════════════════════════════════════
# AGC 伪查询实验 — Step 1: 训练
#
# 使用 OmniColPress 框架训练 AGC 压缩模型
# 文档侧输入: 伪查询文本 (text-only, 无视频帧)
# 查询侧输入: GT caption (text)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 配置 ─────────────────────────────────────────────────────
# OmniColPress 仓库路径 (克隆后修改此路径)
OMNI_COL_PRESS_DIR="${OMNI_COL_PRESS_DIR:-/path/to/omni-col-press}"

# 项目根目录 (pseudo-query)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# GPU 数量
NUM_GPUS="${NUM_GPUS:-4}"

# 压缩预算 (对齐论文 Table 4 的 budget=32)
NUM_REPR_VECTORS="${NUM_REPR_VECTORS:-32}"
NUM_APPENDING_TOKENS="${NUM_APPENDING_TOKENS:-32}"

# passage_max_len: 伪查询拼接后文本长度
# 伪查询拼接后 p95=1173 词, 对应约 1500+ token
# Qwen2.5-VL 文本 token 化约 1.3x 词数, 1024 覆盖大部分
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-1024}"

# 输出目录
OUTPUT_DIR="${PROJECT_DIR}/experiment/outputs/agc_pq_b${NUM_REPR_VECTORS}"

# ─── 检查 ─────────────────────────────────────────────────────
if [ ! -d "${OMNI_COL_PRESS_DIR}/src" ]; then
    echo "ERROR: OmniColPress not found at ${OMNI_COL_PRESS_DIR}"
    echo "Please clone: git clone https://github.com/hanxiangqin/omni-col-press.git"
    echo "Then set OMNI_COL_PRESS_DIR environment variable."
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/experiment/data/train.jsonl" ]; then
    echo "ERROR: Training data not found. Run prepare_data.py first:"
    echo "  cd ${PROJECT_DIR} && python experiment/prepare_data.py"
    exit 1
fi

# ─── 训练 ─────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "AGC 伪查询实验 — 训练"
echo "  Model:            Qwen/Qwen2.5-VL-3B-Instruct"
echo "  Pooling:          AGC (select)"
echo "  Budget:           ${NUM_REPR_VECTORS} tokens"
echo "  Appending tokens: ${NUM_APPENDING_TOKENS}"
echo "  Passage max len:  ${PASSAGE_MAX_LEN}"
echo "  GPUs:             ${NUM_GPUS}"
echo "  Output:           ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════"

cd "${OMNI_COL_PRESS_DIR}"

deepspeed --num_gpus=${NUM_GPUS} src/train.py \
    --deepspeed deepspeed/ds_zero3_config.json \
    \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    \
    --train_yaml "${PROJECT_DIR}/experiment/configs/train_data.yaml" \
    \
    --encode_modalities '{"default": {"text": true, "image": false, "video": false, "audio": false}}' \
    \
    --query_prefix "Query: " \
    --passage_prefix "Passage: " \
    --query_max_len 77 \
    --passage_max_len ${PASSAGE_MAX_LEN} \
    --train_group_size 2 \
    \
    --pooling select \
    --num_repr_vectors ${NUM_REPR_VECTORS} \
    --num_appending_token ${NUM_APPENDING_TOKENS} \
    --use_parametric_appending_tokens \
    --use_cluster_pooling \
    --use_attn_weight_cluster_pooling \
    --normalize \
    \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 28 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    \
    --logging_steps 10 \
    --logging_first_step \
    --save_strategy epoch \
    --report_to wandb \
    --run_name "agc_pseudoquery_b${NUM_REPR_VECTORS}"

echo ""
echo "训练完成！模型保存在: ${OUTPUT_DIR}"
echo "下一步: 运行 run_eval.sh 进行评估"
