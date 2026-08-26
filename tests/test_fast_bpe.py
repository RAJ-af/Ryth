"""fast_bpe differential tests — naive trainer ke BIT-IDENTICAL merges proof.

W1-revision directive point 6: CPU-only BPE bottleneck ka sound redesign.
Fast impl tabhi trustworthy hai jab merge SEQUENCE (order samet) naive jaisi
ho — differential tests har edge-case me ye assert karte hain.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe import BPETokenizer                    # noqa: E402
from tokenizer.fast_bpe import train_fast                 # noqa: E402


def _assert_identical(tok_naive: BPETokenizer, tok_fast: BPETokenizer):
    """Merge sequence EXACT (order samet) + vocab bytes EXACT."""
    seq_n = list(tok_naive.merges.items())
    seq_f = list(tok_fast.merges.items())
    assert len(seq_n) == len(seq_f), f"{len(seq_n)} vs {len(seq_f)} merges"
    for i, (a, b) in enumerate(zip(seq_n, seq_f)):
        assert a == b, f"merge #{i} diverges: naive={a} fast={b}"
    assert tok_naive.vocab == tok_fast.vocab


REAL_MIXED = [
    # hindi wikipedia-jaisa
    "भारत एक देश है। यह बहुत बड़ा देश है। दिल्ली भारत की राजधानी है। " * 8,
    # bengali + tamil
    "বাংলা ভাষা একটি সুন্দর ভাষা। " * 6,
    "தமிழ் ஒரு அழகான மொழி. இது தென்னிந்தியாவில் பேசப்படுகிறது. " * 6,
    # english prose
    "The quick brown fox jumps over the lazy dog. " * 10,
    # python
    "def add(a, b):\n    return a + b\n\nclass Foo:\n    pass\n" * 10,
    # c
    "#include <stdio.h>\nint main(void) { printf(\"hi\\n\"); return 0; }\n" * 8,
]


def test_exact_parity_real_mixed_text():
    tok_n = BPETokenizer().train(list(REAL_MIXED), vocab_size=700)
    tok_f = train_fast(list(REAL_MIXED), vocab_size=700)
    _assert_identical(tok_n, tok_f)
    assert len(tok_f.merges) > 200              # kaafi merges hue — real exercise


def test_exact_parity_heavy_ties():
    # do SAME-frequency bigrams — tie-break corpus-order pehle wala chune
    tied = ("AB " * 60 + "CD " * 60) * 3
    tok_n = BPETokenizer().train([tied], vocab_size=400)
    tok_f = train_fast([tied], vocab_size=400)
    _assert_identical(tok_n, tok_f)
    # pehla merge (A,B) hona chahiye — scan-order pehla
    first_pair = next(iter(tok_f.merges))
    assert first_pair == (ord("A"), ord("B"))

    # aur bhi adversarial: barabar counts, interleaved
    adv = "xyxy zxzy wxwx qxqx " * 40
    tok_n2 = BPETokenizer().train([adv], vocab_size=350)
    tok_f2 = train_fast([adv], vocab_size=350)
    _assert_identical(tok_n2, tok_f2)


def test_early_stop_parity_tiny_corpus():
    tiny = ["ha ha ha"]
    tok_n = BPETokenizer().train(tiny, vocab_size=500)   # kabhi 500 nahi pahunchega
    tok_f = train_fast(tiny, vocab_size=500)
    _assert_identical(tok_n, tok_f)
    assert len(tok_f.merges) < 100                   # dono ne wahi early-stop kiya


def test_empty_and_single_piece_inputs():
    for texts in ([], [""], ["x"], ["onlyone"]):
        tok_n = BPETokenizer().train(texts, vocab_size=300)
        tok_f = train_fast(texts, vocab_size=300)
        _assert_identical(tok_n, tok_f)


def test_checkpoint_partial_save(tmp_path):
    cp = tmp_path / "partial.json"
    tok = train_fast(list(REAL_MIXED), vocab_size=700,
                     checkpoint_path=str(cp), checkpoint_every=100)
    assert cp.exists()
    data = json.load(open(cp, encoding="utf-8"))
    assert len(data["merges"]) >= 100               # partial snapshot tha...
    final = json.loads(json.dumps(
        [[a, b, i] for (a, b), i in tok.merges.items()]))
    assert len(final) == len(tok.merges)


def test_fast_result_roundtrips_through_save_load(tmp_path):
    tok = train_fast(list(REAL_MIXED), vocab_size=600)
    p = str(tmp_path / "t.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert list(tok2.merges.items()) == list(tok.merges.items())
    assert tok2.vocab == tok.vocab


def test_exact_parity_larger_vocab_stress():
    """Zyada merges = zyada tie-cascade chances — parity wahan bhi kaape."""
    tok_n = BPETokenizer().train(list(REAL_MIXED), vocab_size=1200)
    tok_f = train_fast(list(REAL_MIXED), vocab_size=1200)
    _assert_identical(tok_n, tok_f)
