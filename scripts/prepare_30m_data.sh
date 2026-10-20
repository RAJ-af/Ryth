#!/usr/bin/env bash
# Prepare data for 30M prototype model end-to-end.
#
# Usage:
#   scripts/prepare_30m_data.sh [input_dir] [out_dir]
#
# Defaults:
#   input_dir = raw_repos
#   out_dir   = rds_out
#
# Flow:
#   1. Ensure non-empty input_dir. If missing/empty:
#      Try downloading a small permissive corpus using ryth-corpus / python.
#      If offline/network unavailable, fall back to sample fixture build.
#   2. ryth-corpus build -> corpus_out
#   3. ryth-tokenizer train (vocab 16000) -> tok/
#   4. ryth-rde build (seq_len 1024) -> out_dir

set -euo pipefail

INPUT_DIR="${1:-raw_repos}"
OUT_DIR="${2:-rds_out}"
TOK_DIR="tok"
CORPUS_OUT="corpus_out"
VOCAB_SIZE=16000
SEQ_LEN=1024

echo "============================================================"
echo " Preparing 30M Data Pipeline"
echo " Input Dir : ${INPUT_DIR}"
echo " Output Dir: ${OUT_DIR}"
echo " Tok Dir   : ${TOK_DIR}"
echo "============================================================"

has_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            return 0
        fi
    fi
    return 1
}

if ! has_files "${INPUT_DIR}"; then
    echo ">> '${INPUT_DIR}' is missing or empty. Attempting to fetch permissive-license corpus..."
    mkdir -p "${INPUT_DIR}"

    python3 -c "
import sys, os
from corpus.sources import default_sources
from corpus.download import resolve_downloader

sources = [s for s in default_sources().enabled() if s.kind in ('github', 'http')]
stage_dir = '${INPUT_DIR}'
ok = 0
for s in sources[:2]:  # Top small sources
    dl = resolve_downloader(s.kind)
    dl.timeout = 10.0
    try:
        staged = dl.fetch(s, stage_dir)
        print(f'  [+] Staged {staged.repo} -> {staged.root}')
        ok += 1
    except Exception as e:
        print(f'  [-] Failed {s.id}: {e}')

if ok == 0:
    sys.exit(1)
" || {
        echo "============================================================"
        echo " WARNING: Network/download failed or offline."
        echo " Falling back to generating small sample fixture dataset."
        echo "============================================================"
        python3 -c "from tests.sample_data import build_sample; build_sample('${INPUT_DIR}')"
    }
fi

echo ">> 1. Running ryth-corpus build..."
ryth-corpus build --input "${INPUT_DIR}" --out "${CORPUS_OUT}"

# Search for python files to train tokenizer on
TOKENIZER_GLOB="${INPUT_DIR}/**/*.py"
if ! python3 -c "import sys, glob; sys.exit(0 if glob.glob('${TOKENIZER_GLOB}', recursive=True) else 1)"; then
    TOKENIZER_GLOB="${INPUT_DIR}/**/*"
fi

echo ">> 2. Training BPE tokenizer (vocab=${VOCAB_SIZE}) -> ${TOK_DIR}..."
ryth-tokenizer train --files "${TOKENIZER_GLOB}" --vocab "${VOCAB_SIZE}" --out "${TOK_DIR}"

echo ">> 3. Building RDS dataset (seq_len=${SEQ_LEN}) -> ${OUT_DIR}..."
ryth-rde build "${INPUT_DIR}" "${OUT_DIR}" \
    --tokenizer "${TOK_DIR}/tokenizer.json" \
    --seq_len "${SEQ_LEN}"

echo ">> 4. Verifying RDS dataset..."
ryth-rde verify "${OUT_DIR}"

echo "============================================================"
echo " 30M Data Preparation Complete!"
echo " Dataset Shards: ${OUT_DIR}"
echo " Tokenizer     : ${TOK_DIR}/tokenizer.json"
echo " Report        : ${OUT_DIR}/report.html"
echo "============================================================"
