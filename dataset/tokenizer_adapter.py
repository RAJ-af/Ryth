"""Module 6 bridge — RDE ko kisi bhi tokenizer se jodo.

RDE ko sirf ek chhota interface chahiye:
    .encode(text) -> list[int]
    .decode(ids)  -> str
    .vocab_size, .eos_id, .bos_id, .pad_id, .version

`ByteTokenizer` ek self-contained fallback hai (byte-level, hamesha chalta hai),
taaki RDE bina kisi external file ke test ho sake. Production me apna scratch BPE
tokenizer (`tokenizer/bpe.py`) plug karo — usme bhi yehi methods hain.
"""

from __future__ import annotations


class ByteTokenizer:
    """Zero-dependency byte-level tokenizer (256 bytes + special tokens)."""

    def __init__(self):
        self._special = {"<|eos|>": 256, "<|pad|>": 257, "<|bos|>": 258}
        self.vocab_size = 259
        self.version = 0

    @property
    def eos_id(self) -> int: return self._special["<|eos|>"]

    @property
    def pad_id(self) -> int: return self._special["<|pad|>"]

    @property
    def bos_id(self) -> int: return self._special["<|bos|>"]

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def load_bpe_tokenizer(path: str, version: int = 1):
    """Aapka scratch BPE tokenizer (`tokenizer/bpe.py`) load karo.

    Isse RDE ke saath use karne ke liye ye chhota wrapper deta hai jo eos/bos/pad
    ids expose karta hai. Agar special tokens missing ho to fallback ids use hote.
    """
    from tokenizer.bpe import BPETokenizer              # noqa: import on demand

    tok = BPETokenizer.load(path)

    class _Wrapped:
        def __init__(self, t):
            self._t = t
            self.version = version
            self.vocab_size = t.vocab_size
            sp = t.special_tokens
            self.eos_id = sp.get("<|endoftext|>", sp.get("<|eos|>", 0))
            self.bos_id = sp.get("<|endoftext|>", self.eos_id)
            self.pad_id = sp.get("<|pad|>", self.eos_id)

        def encode(self, text): return self._t.encode(text)
        def decode(self, ids): return self._t.decode(list(ids))

    return _Wrapped(tok)
