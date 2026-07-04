"""Corpus helpers + a reusable training function for the BPE tokenizer.

Do corpus sources supported:
  * JSONL folder — RDE-style formats (instruction/reasoning/fim/chat.jsonl)
  * Plain text files — koi bhi .txt/.py/... files (raw content)
"""

from __future__ import annotations

import glob
import json
import os

from .bpe import BPETokenizer
from . import DEFAULT_SPECIAL_TOKENS


def iter_jsonl_corpus(data_dir: str):
    """RDE dataset ke jsonl formats se text stream banao."""
    def _read(name):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    for ex in _read("instruction.jsonl"):
        yield (ex.get("instruction", "") + "\n" + ex.get("output", "")).strip()
    for ex in _read("reasoning.jsonl"):
        yield (ex.get("problem", "") + "\n" + ex.get("reasoning", "") + "\n"
               + ex.get("solution", "") + "\n" + "\n".join(ex.get("tests", []))).strip()
    for ex in _read("fim.jsonl"):
        yield ex.get("psm", "")
    for ex in _read("chat.jsonl"):
        yield "\n".join(m.get("content", "") for m in ex.get("messages", []))


def iter_text_files(pattern: str):
    """Glob pattern se plain text/code files padho (recursive)."""
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            if text.strip():
                yield text


def train_tokenizer(texts, vocab_size: int, out_dir: str, *,
                    special_tokens=None, verbose: bool = False) -> BPETokenizer:
    """Train a BPE tokenizer on `texts` and save to `out_dir/tokenizer.json`."""
    special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
    tok = BPETokenizer()
    tok.train(list(texts), vocab_size=vocab_size, verbose=verbose)
    tok.add_special_tokens(special_tokens)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    if verbose:
        print(f"learned {len(tok.merges)} merges | vocab_size = {tok.vocab_size}")
        print(f"saved -> {path}")
    return tok
