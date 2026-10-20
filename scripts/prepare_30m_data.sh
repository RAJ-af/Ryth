#!/usr/bin/env bash
# Prepare data for 30M prototype training.
#
# Pipeline:
#   1. ryth-corpus build -> cleans/scores permissive raw code into corpus records
#   2. ryth-corpus export -> exports records into a cleaned raw tree
#   3. ryth-tokenizer train -> trains byte-level BPE tokenizer (vocab 16000)
#   4. ryth-rde build -> packs dataset into RDS format (seq_len 1024)
#
# Usage:
#   scripts/prepare_30m_data.sh [RAW_DIR] [CORPUS_DIR] [TOK_DIR] [RDS_DIR]
#
# If RAW_DIR is not supplied and raw_repos/ does not exist, synthetic permissively
# licensed code will be generated so the pipeline runs out-of-the-box.

set -euo pipefail

RAW_DIR="${1:-raw_repos}"
CORPUS_DIR="${2:-data/corpus_30m}"
TOK_DIR="${3:-data/tok_16k}"
RDS_DIR="${4:-data/rds_30m}"
VOCAB_SIZE=16000
SEQ_LEN=1024

echo "============================================================"
echo " Ryth 30M Prototype Data Preparation"
echo "============================================================"
echo " Raw directory:     ${RAW_DIR}"
echo " Corpus directory:  ${CORPUS_DIR}"
echo " Tokenizer output:  ${TOK_DIR}"
echo " RDS output:        ${RDS_DIR}"
echo " Vocab size:        ${VOCAB_SIZE}"
echo " Sequence length:   ${SEQ_LEN}"
echo "============================================================"

# If RAW_DIR doesn't exist or is empty, create synthetic permissively-licensed repos
if [ ! -d "${RAW_DIR}" ] || [ -z "$(ls -A "${RAW_DIR}" 2>/dev/null)" ]; then
    echo ">> [1/4] RAW_DIR '${RAW_DIR}' missing or empty. Generating synthetic permissive corpus..."
    python -c "
import os
raw_dir = '${RAW_DIR}'
os.makedirs(os.path.join(raw_dir, 'repo_synth'), exist_ok=True)
lic = 'MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy...'
with open(os.path.join(raw_dir, 'repo_synth', 'LICENSE'), 'w') as f:
    f.write(lic)
with open(os.path.join(raw_dir, 'repo_synth', 'README.md'), 'w') as f:
    f.write('# Synthetic Repo\nPermissive sample for 30M prep.\n')

for i in range(20):
    code = f'''\"\"\"Module {i} for 30M prototype training.\"\"\"

def process_item_{i}(val: int) -> int:
    \"\"\"Return processed item {i}.\"\"\"
    res = val * {i + 1}
    if res > 100:
        return res % 100
    return res

class Handler_{i}:
    def __init__(self, name: str):
        self.name = name
        self.count = {i}

    def execute(self) -> str:
        return self.name + str(self.count)
'''
    with open(os.path.join(raw_dir, 'repo_synth', f'mod_{i}.py'), 'w') as f:
        f.write(code)
    with open(os.path.join(raw_dir, 'repo_synth', f'test_mod_{i}.py'), 'w') as f:
        f.write(f'from mod_{i} import process_item_{i}\ndef test_{i}(): assert process_item_{i}(2) >= 0\n')
"
fi

# Step 1: ryth-corpus build
echo ">> [1/4] Running ryth-corpus build..."
ryth-corpus build --input "${RAW_DIR}" --out "${CORPUS_DIR}" --min-quality 0

# Step 2: ryth-corpus export (extract cleaned code tree for tokenizer & RDE)
CLEANED_RAW_DIR="${CORPUS_DIR}/raw"
echo ">> [2/4] Exporting cleaned corpus to '${CLEANED_RAW_DIR}'..."
ryth-corpus export --records "${CORPUS_DIR}/records.jsonl" --format raw --out "${CLEANED_RAW_DIR}"

# Step 3: ryth-tokenizer train
echo ">> [3/4] Running ryth-tokenizer train (vocab ${VOCAB_SIZE})..."
ryth-tokenizer train --files "${CLEANED_RAW_DIR}/**/*.py" --vocab "${VOCAB_SIZE}" --out "${TOK_DIR}"

# Step 4: ryth-rde build
echo ">> [4/4] Running ryth-rde build (seq_len ${SEQ_LEN})..."
ryth-rde build "${CLEANED_RAW_DIR}" "${RDS_DIR}" \
    --tokenizer "${TOK_DIR}/tokenizer.json" \
    --seq_len "${SEQ_LEN}"

echo ">> Verifying RDS dataset..."
ryth-rde verify "${RDS_DIR}"

echo "============================================================"
echo " 30M Data Preparation Complete!"
echo " Tokenizer: ${TOK_DIR}/tokenizer.json"
echo " RDS Shards: ${RDS_DIR}"
echo "============================================================"
