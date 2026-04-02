#!/bin/bash
# ══════════════════════════════════════════════════════════════
# AGC 伪查询实验 — Step 2: 构建索引 + 检索评估
#
# 两步流程:
#   1. build_index: 编码 test corpus → 构建多向量索引
#   2. evaluate:    编码 test queries → 检索 + 计算指标
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 配置 ─────────────────────────────────────────────────────
OMNI_COL_PRESS_DIR="${OMNI_COL_PRESS_DIR:-/path/to/omni-col-press}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

NUM_GPUS="${NUM_GPUS:-4}"
NUM_REPR_VECTORS="${NUM_REPR_VECTORS:-32}"
NUM_APPENDING_TOKENS="${NUM_APPENDING_TOKENS:-32}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-1024}"

# 训练好的模型检查点路径
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/experiment/outputs/agc_pq_b${NUM_REPR_VECTORS}}"

# 索引类型: "multivec" (PyTorch brute-force) 或 "fast-plaid"
INDEX_TYPE="${INDEX_TYPE:-multivec}"

# 输出路径
INDEX_DIR="${MODEL_PATH}/index_${INDEX_TYPE}"
RESULT_DIR="${MODEL_PATH}/results"

# ─── 检查 ─────────────────────────────────────────────────────
if [ ! -d "${OMNI_COL_PRESS_DIR}/src" ]; then
    echo "ERROR: OmniColPress not found at ${OMNI_COL_PRESS_DIR}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model checkpoint not found at ${MODEL_PATH}"
    echo "Please run run_train.sh first."
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/experiment/data/test_corpus.jsonl" ]; then
    echo "ERROR: Test data not found. Run prepare_data.py first."
    exit 1
fi

cd "${OMNI_COL_PRESS_DIR}"

# ─── Step 1: Build Index ──────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "Step 1/2: 构建索引"
echo "  Model:      ${MODEL_PATH}"
echo "  Corpus:     test_corpus.jsonl (2990 docs)"
echo "  Index type: ${INDEX_TYPE}"
echo "  Budget:     ${NUM_REPR_VECTORS} tokens"
echo "════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=${NUM_GPUS} -m src.build_index \
    --model_name_or_path "${MODEL_PATH}" \
    \
    --corpus_path "${PROJECT_DIR}/experiment/data/test_corpus.jsonl" \
    --dataset_name json \
    \
    --encode_modalities '{"default": {"text": true, "image": false, "video": false, "audio": false}}' \
    --passage_prefix "Passage: " \
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
    --batch_size 8 \
    --bf16

echo ""
echo "索引构建完成: ${INDEX_DIR}"

# ─── Step 2: Retrieve & Evaluate ─────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 2/2: 检索与评估"
echo "  Queries:  test_queries.csv (1000 queries)"
echo "  Qrels:    test_qrels.jsonl"
echo "  Top-K:    1, 5, 10"
echo "════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=${NUM_GPUS} -m src.evaluate \
    --model_name_or_path "${MODEL_PATH}" \
    \
    --query_path "${PROJECT_DIR}/experiment/data/test_queries.csv" \
    --qrels_path "${PROJECT_DIR}/experiment/data/test_qrels.jsonl" \
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
    --bf16

echo ""
echo "════════════════════════════════════════════════════════════"
echo "评估完成！结果保存在: ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════════"
