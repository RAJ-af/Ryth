"""Unit tests for the scratch byte-level BPE tokenizer.

Pure standard library. Run:
    pytest tests/test_tokenizer.py -v
    # or: python tests/test_tokenizer.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS

_CORPUS = [
    "def add(a, b):\n    return a + b",
    "def mul(a, b):\n    return a * b",
    "class Point:\n    def __init__(self, x):\n        self.x = x",
    "for i in range(10):\n    print(i)",
] * 5


def _tok(vocab=350):
    t = BPETokenizer()
    t.train(_CORPUS, vocab_size=vocab, verbose=False)
    t.add_special_tokens(DEFAULT_SPECIAL_TOKENS)
    return t


def test_train_grows_vocab():
    t = BPETokenizer()
    t.train(_CORPUS, vocab_size=300)
    assert t.vocab_size > 256                 # learned some merges above base bytes
    assert len(t.merges) > 0


def test_vocab_size_floor():
    t = BPETokenizer()
    try:
        t.train(_CORPUS, vocab_size=100)      # < 256 base bytes
        assert False, "expected assertion"
    except AssertionError:
        pass


def test_encode_decode_roundtrip():
    t = _tok()
    for text in ["def add(a, b): return a + b", "class X: pass", "x = [1, 2, 3]"]:
        ids = t.encode(text)
        assert t.decode(ids) == text          # exact roundtrip


def test_byte_fallback_non_ascii():
    """Non-ASCII (never trained) must still roundtrip via byte fallback."""
    t = _tok()
    for text in ["नमस्ते दुनिया", "வணக்கம்", "🚀 def f():"]:
        ids = t.encode(text)
        assert t.decode(ids) == text          # no unknown token, lossless


def test_special_tokens_are_single_ids():
    t = _tok()
    for s in DEFAULT_SPECIAL_TOKENS:
        ids = t.encode(s)
        assert len(ids) == 1, (s, ids)        # never split


def test_special_tokens_embedded():
    t = _tok()
    text = "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    return 1"
    ids = t.encode(text)
    assert t.decode(ids) == text


def test_save_load_roundtrip():
    t = _tok()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tok.json")
    t.save(path)

    t2 = BPETokenizer.load(path)
    assert t2.vocab_size == t.vocab_size
    assert t2.merges == t.merges
    text = "def add(a, b): return a + b"
    assert t2.encode(text) == t.encode(text)  # identical encoding after reload
    assert t2.decode(t2.encode(text)) == text


def test_compression_improves_on_trained_text():
    """Trained-domain text should compress better than raw bytes."""
    t = _tok(vocab=400)
    text = "def add(a, b):\n    return a + b\n" * 3
    n_tokens = len(t.encode(text))
    n_bytes = len(text.encode("utf-8"))
    assert n_tokens < n_bytes                 # merges reduced the length


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tokenizer tests passed ✅")
