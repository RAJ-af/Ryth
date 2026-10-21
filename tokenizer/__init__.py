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


def build_special_tokens(reserved: int = 256,
                         extra_specials: list[str] | None = None) -> list[str]:
    """Build list of special tokens starting with <|reserved_0|>...<|reserved_{reserved-1}|>."""
    res = [f"<|reserved_{i}|>" for i in range(reserved)]
    if extra_specials is not None:
        res.extend(extra_specials)
    else:
        res.extend(DEFAULT_SPECIAL_TOKENS)
    return res


__all__ = ["BPETokenizer", "DEFAULT_SPECIAL_TOKENS", "build_special_tokens"]
