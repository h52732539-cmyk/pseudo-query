#!/bin/bash
# ══════════════════════════════════════════════════════════════
# AGC 视频原始实验 — Step 2: 构建索引 + 检索评估
#
# 两步流程:
#   1. build_index: 编码 test corpus (视频) → 构建多向量索引
#   2. evaluate:    编码 test queries (文本) → 检索 + 计算指标
#
# 与伪查询实验的核心差异：
#   - 文档编码使用 video 模态
#   - assets_path 指向 MSRVTT 视频目录
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 配置 ─────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OMNI_COL_PRESS_DIR="${OMNI_COL_PRESS_DIR:-${PROJECT_DIR}/omni-col-press}"

NUM_GPUS="${NUM_GPUS:-4}"
NUM_REPR_VECTORS="${NUM_REPR_VECTORS:-32}"
NUM_APPENDING_TOKENS="${NUM_APPENDING_TOKENS:-32}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-2048}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES

# MSRVTT 视频资产目录
VIDEO_ASSETS_DIR="/hpc2hdd/home/yuxuanzhao/xuhaodong/MSRVTT/videos/all"

# 训练好的模型检查点路径
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/outputs/agc_video_b${NUM_REPR_VECTORS}}"

# 索引类型
INDEX_TYPE="${INDEX_TYPE:-multivec}"

INDEX_DIR="${MODEL_PATH}/index_${INDEX_TYPE}"
RESULT_DIR="${MODEL_PATH}/results"
MASTER_PORT=$((RANDOM % 10000 + 20000))

# ─── 检查 ─────────────────────────────────────────────────────
if [ ! -d "${OMNI_COL_PRESS_DIR}/src" ]; then
    echo "ERROR: OmniColPress not found at ${OMNI_COL_PRESS_DIR}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: 模型检查点不存在: ${MODEL_PATH}"
    echo "请先运行 run_train_video.sh"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/data-video/test_corpus.jsonl" ]; then
    echo "ERROR: 测试数据不存在，请先运行 prepare_data_video.py"
    exit 1
fi

if [ ! -d "${VIDEO_ASSETS_DIR}" ]; then
    echo "ERROR: 视频目录不存在: ${VIDEO_ASSETS_DIR}"
    exit 1
fi

cd "${OMNI_COL_PRESS_DIR}"
export PYTHONPATH="${OMNI_COL_PRESS_DIR}:${PYTHONPATH:-}"

# ─── Step 1: 构建视频索引 ─────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "Step 1/2: 构建视频索引"
echo "  Model:      ${MODEL_PATH}"
echo "  Corpus:     test_corpus.jsonl (视频, ~2990 docs)"
echo "  Index type: ${INDEX_TYPE}"
echo "  Budget:     ${NUM_REPR_VECTORS} tokens"
echo "════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=${NUM_GPUS} --master-port=${MASTER_PORT} -m src.build_index \
    --model_name_or_path "${MODEL_PATH}" \
    \
    --corpus_path "${PROJECT_DIR}/data-video/test_corpus.jsonl" \
    --dataset_name json \
    --assets_path "${VIDEO_ASSETS_DIR}" \
    \
    --encode_modalities '{"default": {"text": false, "image": false, "video": true, "audio": false}}' \
    --passage_prefix "" \
    --passage_max_len ${PASSAGE_MAX_LEN} \
    \
    --pooling select \
    --num_repr_vectors ${NUM_REPR_VECTORS} \
    --num_appending_token ${NUM_APPENDING_TOKENS} \
    --use_parametric_appending_tokens \
    --use_cluster_pooling \
    --use_attn_weight_cluster_pooling \
    --normalize \
    \
    --index_output_path "${INDEX_DIR}" \
    --index_type ${INDEX_TYPE} \
    --batch_size 2 \
    --dtype bfloat16

echo ""
echo "索引构建完成: ${INDEX_DIR}"

# ─── Step 2: 检索与评估（查询侧仍为文本）────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 2/2: 检索与评估"
echo "  Queries:  test_queries.csv (1000 queries, 文本)"
echo "  Qrels:    test_qrels.jsonl"
echo "  Top-K:    1, 5, 10"
echo "════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=${NUM_GPUS} --master-port=$((MASTER_PORT+1)) -m src.evaluate \
    --model_name_or_path "${MODEL_PATH}" \
    \
    --query_path "${PROJECT_DIR}/data-video/test_queries.csv" \
    --qrels_path "${PROJECT_DIR}/data-video/test_qrels.jsonl" \
    --dataset_name csv \
    \
    --encode_is_query \
    --encode_modalities '{"default": {"text": true, "image": false, "video": false, "audio": false}}' \
    --query_prefix "Query: " \
    --query_max_len 77 \
    \
    --pooling select \
    --num_repr_vectors ${NUM_REPR_VECTORS} \
    --num_appending_token ${NUM_APPENDING_TOKENS} \
    --use_parametric_appending_tokens \
    --use_cluster_pooling \
    --use_attn_weight_cluster_pooling \
    --normalize \
    \
    --index_path "${INDEX_DIR}" \
    --index_type ${INDEX_TYPE} \
    --output_path "${RESULT_DIR}" \
    --batch_size 8 \
    --top_k 1 5 10 \
    --dtype bfloat16

echo ""
echo "════════════════════════════════════════════════════════════"
echo "评估完成！结果保存在: ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════════"
