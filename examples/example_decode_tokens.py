"""Example: encode text to tokens and decode back (round-trip).

Run:  python examples/example_decode_tokens.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS


def main():
    # train a tiny tokenizer on a few code lines
    corpus = [
        "def add(a, b):\n    return a + b",
        "def mul(a, b):\n    return a * b",
        "for i in range(10):\n    print(i)",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x",
    ] * 5
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=350, verbose=False)
    tok.add_special_tokens(DEFAULT_SPECIAL_TOKENS)

    samples = [
        "def add(a, b): return a + b",
        "print('नमस्ते')",              # non-ASCII: byte fallback, no unknown token
        "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    pass",
    ]
    for text in samples:
        ids = tok.encode(text)
        back = tok.decode(ids)
        ratio = len(text) / max(1, len(ids))
        print(f"text     : {text!r}")
        print(f"  tokens : {len(ids)}  ({ratio:.2f} chars/token)")
        print(f"  ids    : {ids[:20]}{' ...' if len(ids) > 20 else ''}")
        print(f"  decode : {back!r}")
        print(f"  exact roundtrip: {back == text}\n")

    # special tokens are single ids
    for t in ["<|fim_prefix|>", "<|assistant|>", "<|endoftext|>"]:
        enc = tok.encode(t)
        print(f"{t:16s} -> {enc}  ({'single token' if len(enc)==1 else 'MULTI'})")


if __name__ == "__main__":
    main()
