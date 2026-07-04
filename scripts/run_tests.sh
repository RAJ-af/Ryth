#!/usr/bin/env bash
# Run the full test suite. Works with or without pytest installed.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v pytest >/dev/null 2>&1; then
    pytest -q
else
    echo "pytest not found — running test files directly"
    python tests/test_tokenizer.py
    python tests/test_rde.py
fi
