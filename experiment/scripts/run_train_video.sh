#!/bin/bash
# ══════════════════════════════════════════════════════════════
# AGC 视频原始实验 — Step 1: 训练
#
# 与伪查询实验的核心差异：
#   - 文档侧输入: 原始 MP4 视频 (24帧, Qwen2.5-VL-3B 编码)
#   - encode_modalities: video=true, text=false
#   - assets_path 指向 MSRVTT 视频目录
#   - passage_max_len: 2048 (视频约 1296 tokens + 冗余)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 配置 ─────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OMNI_COL_PRESS_DIR="${PROJECT_DIR}/omni-col-press"

# GPU
NUM_GPUS="${NUM_GPUS:-4}"
GPU_ID="${GPU_ID:-4,5,6,7}"

# 压缩预算 (对齐论文 Table 4 的 budget=32)
NUM_REPR_VECTORS="${NUM_REPR_VECTORS:-32}"
NUM_APPENDING_TOKENS="${NUM_APPENDING_TOKENS:-32}"

# passage_max_len:
#   Qwen2.5-VL 编码 24 帧视频约产生 1296 visual tokens
#   再加少量文本 token (prefix 等), 设 2048 留足余量
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-2048}"

# MSRVTT 视频资产目录
VIDEO_ASSETS_DIR="/hpc2hdd/home/yuxuanzhao/xuhaodong/MSRVTT/videos/all"

# 输出目录 (独立于 text 实验)
OUTPUT_DIR="${PROJECT_DIR}/outputs/agc_video_b${NUM_REPR_VECTORS}"

# ─── 检查 ─────────────────────────────────────────────────────
if [ ! -d "${OMNI_COL_PRESS_DIR}/src" ]; then
    echo "ERROR: OmniColPress not found at ${OMNI_COL_PRESS_DIR}"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/data-video/train.jsonl" ]; then
    echo "ERROR: 训练数据不存在，请先运行:"
    echo "  cd ${PROJECT_DIR}/.. && python experiment/prepare_data_video.py"
    exit 1
fi

if [ ! -d "${VIDEO_ASSETS_DIR}" ]; then
    echo "ERROR: 视频目录不存在: ${VIDEO_ASSETS_DIR}"
    exit 1
fi

# ─── 训练 ─────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "AGC 视频原始实验 — 训练"
echo "  Model:            Qwen/Qwen2.5-VL-3B-Instruct"
echo "  文档侧:            原始 MP4 视频 (24 帧)"
echo "  Pooling:          AGC (select)"
echo "  Budget:           ${NUM_REPR_VECTORS} tokens"
echo "  Appending tokens: ${NUM_APPENDING_TOKENS}"
echo "  Passage max len:  ${PASSAGE_MAX_LEN}"
echo "  GPUs:             ${NUM_GPUS}"
echo "  Output:           ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════"

cd "${OMNI_COL_PRESS_DIR}"
export PYTHONPATH="${OMNI_COL_PRESS_DIR}:${PYTHONPATH:-}"

deepspeed --include localhost:${GPU_ID} src/train.py \
    --deepspeed deepspeed/ds_zero3_config.json \
    \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    \
    --train_yaml "${PROJECT_DIR}/configs/train_data_video.yaml" \
    \
    --encode_modalities '{"default": {"text": false, "image": false, "video": true, "audio": false}}' \
    \
    --query_prefix "Query: " \
    --passage_prefix "" \
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 28 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --remove_unused_columns false \
    \
    --logging_steps 10 \
    --logging_first_step \
    --save_strategy epoch \
    --report_to wandb \
    --run_name "agc_video_b${NUM_REPR_VECTORS}"

echo ""
echo "训练完成！模型保存在: ${OUTPUT_DIR}"
echo "下一步: 运行 run_eval_video.sh 进行评估"
