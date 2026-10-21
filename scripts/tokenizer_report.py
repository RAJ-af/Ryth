"""Tokenizer Report Script — calculates and prints tokens-per-word and bytes-per-token
for English, Hindi, Hinglish, and Python code samples.
"""

from __future__ import annotations

import argparse
import re
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tokenizer import BPETokenizer, build_special_tokens
from tokenizer.train import train_tokenizer, extend_tokenizer

SAMPLES = {
    "English": (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence and machine learning are transforming the software development industry."
    ),
    "Hindi": (
        "भारत एक विशाल और सुंदर देश है। दिल्ली भारत की राजधानी है। "
        "यहाँ कई भाषाएँ और संस्कृतियाँ एक साथ शांति से रहती हैं।"
    ),
    "Hinglish": (
        "Aaj ka mausam bohot achha hai. Bro, kya tum kal office aa rahe ho? "
        "Mujhe ek naya project start karna hai aur code review complete karna hai."
    ),
    "Python": (
        "def calculate_factorial(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * calculate_factorial(n - 1)\n\n"
        "class DataProcessor:\n"
        "    def __init__(self, data: list):\n"
        "        self.data = data\n"
        "        self.cleaned = [x.strip() for x in data if x]\n"
    )
}


def analyze_tokenizer(tok: BPETokenizer) -> dict[str, dict[str, float | int]]:
    results = {}
    for domain, text in SAMPLES.items():
        raw_bytes = text.encode("utf-8")
        n_bytes = len(raw_bytes)
        words = re.findall(r"\S+", text)
        n_words = len(words) or 1
        tokens = tok.encode(text)
        n_tokens = len(tokens) or 1

        tokens_per_word = n_tokens / n_words
        bytes_per_token = n_bytes / n_tokens

        results[domain] = {
            "n_bytes": n_bytes,
            "n_words": n_words,
            "n_tokens": n_tokens,
            "tokens_per_word": round(tokens_per_word, 3),
            "bytes_per_token": round(bytes_per_token, 3),
        }
    return results


def print_report(results: dict[str, dict[str, float | int]]):
    print("=" * 72)
    print(" RYTH BPE TOKENIZER METRICS REPORT ")
    print("=" * 72)
    print(f"{'Domain':<12} | {'Words':<8} | {'Bytes':<8} | {'Tokens':<8} | {'Tok/Word':<10} | {'Bytes/Tok':<10}")
    print("-" * 72)
    for domain, m in results.items():
        print(f"{domain:<12} | {m['n_words']:<8} | {m['n_bytes']:<8} | {m['n_tokens']:<8} | {m['tokens_per_word']:<10.3f} | {m['bytes_per_token']:<10.3f}")
    print("=" * 72)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tokenizer Report for English, Hindi, Hinglish & Python code")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (if omitted, trains a demo tokenizer)")
    args = parser.parse_args(argv)

    if args.tokenizer and os.path.exists(args.tokenizer):
        tok = BPETokenizer.load(args.tokenizer)
        print(f"Loaded tokenizer from {args.tokenizer} (vocab_size={tok.vocab_size}, version={tok.version})")
    else:
        print("No tokenizer path provided. Training base tokenizer + extending on multilingual sample...")
        base_texts = [SAMPLES["English"], SAMPLES["Python"]] * 10
        tok_base = BPETokenizer()
        tok_base.train(base_texts, vocab_size=500)
        tok_base.add_special_tokens(build_special_tokens(reserved=256))

        ext_texts = [SAMPLES["Hindi"], SAMPLES["Hinglish"]] * 10
        tok = BPETokenizer()
        tok.merges = dict(tok_base.merges)
        tok.vocab = dict(tok_base.vocab)
        tok.special_tokens = dict(tok_base.special_tokens)
        tok._inv_special = dict(tok_base._inv_special)
        tok.version = tok_base.version
        tok.extend(ext_texts, add_vocab=100)
        print(f"Trained demo tokenizer (vocab_size={tok.vocab_size}, version={tok.version})")

    results = analyze_tokenizer(tok)
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
