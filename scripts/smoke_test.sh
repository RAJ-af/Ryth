#!/usr/bin/env bash
# CPU Smoke Test for 30M Prototype.
# Trains ryth_30m on CPU for 20 steps and reports loss progression.
#
# Usage:
#   scripts/smoke_test.sh [data_dir] [out_dir]

set -euo pipefail

DATA_DIR="${1:-rds_out}"
OUT_DIR="${2:-runs/ryth_smoke}"

echo "============================================================"
echo " Running CPU Smoke Test (ryth_30m, 20 steps)"
echo " Data Dir: ${DATA_DIR}"
echo " Out Dir : ${OUT_DIR}"
echo "============================================================"

# Ensure data_dir exists, or prepare sample data if needed
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
    echo ">> '${DATA_DIR}' not found or empty. Preparing dataset..."
    scripts/prepare_30m_data.sh raw_repos "${DATA_DIR}"
fi

ryth-train \
    --data_dir "${DATA_DIR}" \
    --val_data_dir "${DATA_DIR}" \
    --model_preset ryth_30m \
    --seq_len 1024 \
    --micro_batch_size 1 \
    --grad_accum_steps 1 \
    --warmup_steps 2 \
    --max_steps 20 \
    --device cpu \
    --dtype fp32 \
    --log_every 1 \
    --eval_every 20 \
    --save_every 20 \
    --out_dir "${OUT_DIR}"

echo "============================================================"
echo " Smoke Test Complete!"
echo " Log file: ${OUT_DIR}/metrics.jsonl"
echo "============================================================"
