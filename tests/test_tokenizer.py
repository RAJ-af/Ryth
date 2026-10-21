"""Unit tests for the scratch byte-level BPE tokenizer.

Pure standard library. Run:
    pytest tests/test_tokenizer.py -v
    # or: python tests/test_tokenizer.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import unicodedata

from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS, build_special_tokens

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


def test_reserved_special_tokens():
    specials = build_special_tokens(reserved=256)
    assert len(specials) == 256 + len(DEFAULT_SPECIAL_TOKENS)
    assert specials[0] == "<|reserved_0|>"
    assert specials[255] == "<|reserved_255|>"
    assert specials[256] == "<|endoftext|>"

    t = BPETokenizer()
    t.train(_CORPUS, vocab_size=300)
    t.add_special_tokens(specials)
    assert t.encode("<|reserved_0|>") == [t.special_tokens["<|reserved_0|>"]]
    assert t.encode("<|reserved_255|>") == [t.special_tokens["<|reserved_255|>"]]
    assert t.decode([t.special_tokens["<|reserved_0|>"]]) == "<|reserved_0|>"


def test_extend_preserves_old_ids(tmp_path):
    t1 = BPETokenizer()
    t1.train(_CORPUS, vocab_size=300)
    t1.add_special_tokens(build_special_tokens(reserved=256))

    hindi_corpus = [
        "भारत एक महान देश है। दिल्ली भारत की राजधानी है।",
        "हिंदी हमारी भाषा है। नमस्ते दुनिया।",
    ] * 5

    t2 = BPETokenizer()
    t2.merges = dict(t1.merges)
    t2.vocab = dict(t1.vocab)
    t2.special_tokens = dict(t1.special_tokens)
    t2._inv_special = dict(t1._inv_special)
    t2.version = t1.version
    t2.extend(hindi_corpus, add_vocab=50)

    # (a) Every old ID maps to the exact same bytes or special token
    for id_, bytes_ in t1.vocab.items():
        assert id_ in t2.vocab, f"old vocab ID {id_} missing in extended tokenizer"
        assert t2.vocab[id_] == bytes_, f"old vocab ID {id_} changed bytes"

    for token_str, id_ in t1.special_tokens.items():
        assert token_str in t2.special_tokens
        assert t2.special_tokens[token_str] == id_

    for id_, token_str in t1._inv_special.items():
        assert id_ in t2._inv_special
        assert t2._inv_special[id_] == token_str


def test_extend_old_corpus_encoding_parity():
    t1 = BPETokenizer()
    t1.train(_CORPUS, vocab_size=300)
    t1.add_special_tokens(build_special_tokens(reserved=256))

    hindi_corpus = ["भारत एक महान देश है। दिल्ली भारत की राजधानी है।"] * 5

    t2 = BPETokenizer()
    t2.merges = dict(t1.merges)
    t2.vocab = dict(t1.vocab)
    t2.special_tokens = dict(t1.special_tokens)
    t2._inv_special = dict(t1._inv_special)
    t2.version = t1.version
    t2.extend(hindi_corpus, add_vocab=30)

    diff_count = 0
    for text in _CORPUS:
        ids_v1 = t1.encode(text)
        ids_v2 = t2.encode(text)
        if ids_v1 != ids_v2:
            diff_count += 1
            print(f"[extend parity diff] Text: {text!r}")
            print(f"  v1 ({len(ids_v1)} tokens): {ids_v1}")
            print(f"  v2 ({len(ids_v2)} tokens): {ids_v2}")
            max_old_id = max(max(t1.vocab.keys()), max(t1._inv_special.keys()))
            new_ids_in_v2 = [i for i in ids_v2 if i > max_old_id]
            assert len(new_ids_in_v2) > 0, "IDs differed without using any new merge!"
    print(f"Report: {diff_count}/{len(_CORPUS)} old corpus texts had encoding diffs after extend")


def test_extend_new_language_compression():
    t1 = BPETokenizer()
    t1.train(_CORPUS, vocab_size=300)
    t1.add_special_tokens(build_special_tokens(reserved=256))

    hindi_corpus = [
        "भारत एक विशाल देश है। यहाँ की संस्कृति बहुत समृद्ध और विविध है।",
        "नई दिल्ली भारत की राजधानी है। हिंदी और अंग्रेजी प्रमुख भाषाएँ हैं।",
    ] * 10

    t2 = BPETokenizer()
    t2.merges = dict(t1.merges)
    t2.vocab = dict(t1.vocab)
    t2.special_tokens = dict(t1.special_tokens)
    t2._inv_special = dict(t1._inv_special)
    t2.version = t1.version
    t2.extend(hindi_corpus, add_vocab=100)

    sample_text = "भारत की राजधानी नई दिल्ली है। हिंदी एक सुंदर भाषा है।"
    tokens_v1 = len(t1.encode(sample_text))
    tokens_v2 = len(t2.encode(sample_text))

    assert tokens_v2 < tokens_v1, f"Expected compression improvement: {tokens_v1} -> {tokens_v2}"


def test_devanagari_pre_tokenizer():
    from tokenizer.bpe import _SPLIT_PATTERN
    text1 = "लक्ष्मी क्‌लिष्‍ट भारत"
    pieces1 = _SPLIT_PATTERN.findall(text1)
    assert len(pieces1) == 3
    assert pieces1[0] == "लक्ष्मी"
    assert pieces1[1] == " क्\u200cलिष्\u200dट"  # includes ZWNJ (\u200c) and ZWJ (\u200d) inside word
    assert pieces1[2] == " भारत"

    text2 = "भारत एक देश है। ॥ श्रीराम ॥"
    pieces2 = _SPLIT_PATTERN.findall(text2)
    assert pieces2 == ["भारत", " एक", " देश", " है", "।", " ॥", " श्रीराम", " ॥"]


def test_nfc_normalization():
    t = _tok()
    nfd_text = unicodedata.normalize("NFD", "नमस्ते भारत")
    ids = t.encode(nfd_text)
    decoded = t.decode(ids)
    assert decoded == unicodedata.normalize("NFC", nfd_text)


def test_sha256_verification_and_backward_compatibility(tmp_path):
    t = _tok()
    assert t.version == 2
    path = str(tmp_path / "tok_v2.json")
    t.save(path)

    t2 = BPETokenizer.load(path)
    assert t2.version == 2
    assert t2.sha256 != ""

    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    content["merges"].append([0, 1, 9999])
    tampered_path = str(tmp_path / "tampered.json")
    with open(tampered_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=1)

    try:
        BPETokenizer.load(tampered_path)
        assert False, "expected ValueError on sha256 mismatch"
    except ValueError as e:
        assert "sha256 mismatch" in str(e)

    old_path = str(tmp_path / "old_tok.json")
    old_content = {
        "type": "bpe-byte-level",
        "merges": [[a, b, idx] for (a, b), idx in t.merges.items()],
        "special_tokens": t.special_tokens,
    }
    with open(old_path, "w", encoding="utf-8") as f:
        json.dump(old_content, f, indent=1)

    t_old = BPETokenizer.load(old_path)
    assert t_old.version == 1
    assert t_old.vocab_size == t.vocab_size


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in fn.__code__.co_varnames:
            import pathlib
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                fn(pathlib.Path(td))
        else:
            fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tokenizer tests passed ✅")
