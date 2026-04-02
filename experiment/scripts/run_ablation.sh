#!/bin/bash
# ══════════════════════════════════════════════════════════════
# AGC 伪查询实验 — 消融实验: 不同压缩预算
#
# 分别训练和评估 budget = 5, 32, 128 以及无压缩基线 (colbert)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "══════════════════════════════════════════════════════════"
echo "AGC 伪查询消融实验 — 不同压缩预算"
echo "══════════════════════════════════════════════════════════"

# ─── AGC budget=5 ─────────────────────────────────────────────
echo ""
echo ">>> [1/4] AGC budget=5 (99.6% compression)"
NUM_REPR_VECTORS=5 NUM_APPENDING_TOKENS=5 \
    bash "${SCRIPT_DIR}/run_train.sh"
NUM_REPR_VECTORS=5 NUM_APPENDING_TOKENS=5 \
    bash "${SCRIPT_DIR}/run_eval.sh"

# ─── AGC budget=32 ────────────────────────────────────────────
echo ""
echo ">>> [2/4] AGC budget=32 (default, ~93.8% compression)"
NUM_REPR_VECTORS=32 NUM_APPENDING_TOKENS=32 \
    bash "${SCRIPT_DIR}/run_train.sh"
NUM_REPR_VECTORS=32 NUM_APPENDING_TOKENS=32 \
    bash "${SCRIPT_DIR}/run_eval.sh"

# ─── AGC budget=128 ───────────────────────────────────────────
echo ""
echo ">>> [3/4] AGC budget=128 (~75.2% compression)"
NUM_REPR_VECTORS=128 NUM_APPENDING_TOKENS=128 \
    bash "${SCRIPT_DIR}/run_train.sh"
NUM_REPR_VECTORS=128 NUM_APPENDING_TOKENS=128 \
    bash "${SCRIPT_DIR}/run_eval.sh"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "全部消融实验完成"
echo "══════════════════════════════════════════════════════════"
