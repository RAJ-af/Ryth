"""Ryth scratch Byte-Level BPE tokenizer.

A from-scratch Byte-Pair Encoding tokenizer in pure Python (no external
libraries). Byte-level, so any language / script is representable with no
"unknown" token. Supports training, encoding, decoding, save/load, and
special tokens.

Public API:
    BPETokenizer  — the tokenizer (train / encode / decode / save / load)
"""

from .bpe import BPETokenizer

# FIM + chat special tokens. Ye RDE ke FIM sentinels ke saath match karte hain
# taaki model unhe single-token samjhe.
DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>",
    "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]

__all__ = ["BPETokenizer", "DEFAULT_SPECIAL_TOKENS"]
