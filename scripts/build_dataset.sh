#!/usr/bin/env bash
# Convenience wrapper: train a tokenizer, then build + verify an RDS dataset.
#
# Usage:
#   scripts/build_dataset.sh <raw_repos_dir> <output_dir> [vocab] [seq_len]
#
# Example:
#   scripts/build_dataset.sh raw_repos rds_out 8000 1024
set -euo pipefail

ROOT="${1:?usage: build_dataset.sh <raw_repos_dir> <output_dir> [vocab] [seq_len]}"
OUT="${2:?output dir required}"
VOCAB="${3:-8000}"
SEQ_LEN="${4:-1024}"
TOK_DIR="${OUT}_tokenizer"

echo ">> Training tokenizer (vocab=${VOCAB}) on ${ROOT}/**/*.py"
ryth-tokenizer train --files "${ROOT}/**/*.py" --vocab "${VOCAB}" --out "${TOK_DIR}"

echo ">> Building RDS dataset (seq_len=${SEQ_LEN}) -> ${OUT}"
ryth-rde build "${ROOT}" "${OUT}" \
    --tokenizer "${TOK_DIR}/tokenizer.json" --seq_len "${SEQ_LEN}"

echo ">> Verifying shards"
ryth-rde verify "${OUT}"

echo ">> Done. Report: ${OUT}/report.html"
