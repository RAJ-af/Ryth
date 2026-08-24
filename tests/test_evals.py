"""Unit tests for the evals package.

Chat-template + metrics + execution pure stdlib; generation/ppl need torch.
Run:  python -m pytest tests/test_evals.py -v
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe import BPETokenizer
from evals.chat_template import (CHAT_TOKENS, extract_assistant,
                                 register_chat_tokens, render)


def _tok() -> BPETokenizer:
    """Tokenize karne layak fresh tokenizer (encode/decode roundtrip kaam kare)."""
    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=350, verbose=False)
    return tok


def test_register_and_render_roundtrip():
    tok = _tok()
    ids = register_chat_tokens(tok)
    assert set(ids) == {"<|system|>", "<|user|>", "<|assistant|>", "<|end|>"}
    # idempotent: dobara register karne par ids na badlein
    assert register_chat_tokens(tok) == ids
    msg = [{"role": "user", "content": "hello"}]
    s = render(msg, add_generation_prompt=True)
    assert s == "<|user|>hello<|end|><|assistant|>"
    # special tokens encode me intact rehne chahiye
    enc = tok.encode(s)
    assert "<|end|>" in tok.decode(enc)


def test_render_system_and_assistant_roles():
    msgs = [
        {"role": "system", "content": "You are Ryth."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert render(msgs) == ("<|system|>You are Ryth.<|end|>"
                            "<|user|>hi<|end|>"
                            "<|assistant|>hello<|end|>")


def test_extract_assistant_cuts_at_end():
    full = "<|user|>q<|end|><|assistant|>def f():\n    return 1<|end|>"
    assert extract_assistant(full) == "def f():\n    return 1"
    # bina <|end|> ke poora tail milta hai
    assert extract_assistant("x<|assistant|>abc") == "abc"
    assert extract_assistant("no markers here") == ""


# --------------------------------------------------------------------- #
# metrics — pass@k
# --------------------------------------------------------------------- #

from evals.metrics import aggregate, pass_at_k


def test_pass_at_k_known_values():
    assert pass_at_k(100, 100, 1) == 1.0          # sab correct
    assert pass_at_k(100, 0, 1) == 0.0            # kuch correct nahi
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9   # c/n
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9   # simple: 1-(n-c)/n
    # monotone: zyada k => zyada ya barabar chance
    assert pass_at_k(20, 5, 5) >= pass_at_k(20, 5, 1)


def test_aggregate_mean_over_tasks():
    res = [{"task_id": "a", "n": 10, "n_passed": 10},
           {"task_id": "b", "n": 10, "n_passed": 5}]
    out = aggregate(res, ks=(1, 2))
    assert abs(out["pass@1"] - 0.75) < 1e-9       # (1.0 + 0.5)/2
    assert out["pass@2"] >= out["pass@1"]