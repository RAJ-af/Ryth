"""Unit tests for the sft package — fake teacher, zero network.

Run:  python -m pytest tests/test_sft.py -v
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from sft.schema import (Example, Seed, read_jsonl, validate_example,
                        write_jsonl)


def _tok(vocab_size=360):
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=vocab_size, verbose=False)
    return tok


def test_seed_validate_delegates_to_check():
    s = Seed(id="s1", task="t", language="python", user_prompt="u",
             check=lambda text: ([] if "ok" in text else ["not ok"]))
    assert s.validate("ok code") == []
    assert s.validate("bad") == ["not ok"]
    assert Seed(id="s2", task="t", language="", user_prompt="u").validate("") == []


def test_example_to_row_with_tokenizer():
    from evals.chat_template import CHAT_TOKENS   # tuple of token STRINGS
    tok = _tok()
    e = Example(task="t", messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}])
    row = e.to_row(tok)
    assert set(CHAT_TOKENS) <= set(tok.special_tokens)
    assert row["text"].startswith("<|system|>sys<|end|>")
    assert isinstance(row["token_ids"], list)
    assert all(isinstance(i, int) for i in row["token_ids"])
    # tokenizer ke bina: text hai, token_ids nahi (provisional-tok trap)
    row2 = e.to_row(None)
    assert "token_ids" not in row2 and row2["text"].startswith("<|system|>")


def test_example_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "rows.jsonl")
    rows = [{"task": "t", "messages": [], "text": "x"}]
    assert write_jsonl(rows, p) == 1
    assert read_jsonl(p) == rows


def test_validate_example_flags_structure():
    good = {"task": "t", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}], "text": "..."}
    assert validate_example(good) == []
    assert validate_example({"task": "t"}) != []              # missing keys
    bad_roles = dict(good, messages=[{"role": "user", "content": "q"}])
    assert any("role" in r for r in validate_example(bad_roles))
