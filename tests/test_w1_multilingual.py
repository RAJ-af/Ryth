"""W1-revision tests — multilingual corpus, specials registration, efficiency.

W1-revision directive se aaye fixes ka regression coverage:
  * HF source `name` (config) passthrough — wikimedia/wikipedia type datasets
  * special tokens save se PEHLE register + W3-template ID consistency
  * per-source balanced sampler (multilingual)
  * efficiency harness (tokens/char, bytes/token, chars/token)
"""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from corpus.sources.registry import Source          # noqa: E402
from tokenizer.bpe import BPETokenizer              # noqa: E402
from tokenizer import DEFAULT_SPECIAL_TOKENS        # noqa: E402


# ------------------------------------------------------------------- #
# HF `name` passthrough — wikimedia/wikipedia ko config-name chahiye
# ------------------------------------------------------------------- #

class _NameCaptureDS:
    def __init__(self):
        pass

    def __iter__(self):
        return iter([{"text": "यह हिंदी विकी पाठ है। " * 10,
                      "title": "t", "url": "u", "id": "1"}])


def _install_capturing_datasets(monkeypatch):
    """sys.modules fake jaise hi load_dataset ke saare kwargs capture kare."""
    captured = {}

    def fake_load(location, cfg_name=None, split=None, streaming=None,
                  data_dir=None, revision=None):
        captured.update(location=location, cfg_name=cfg_name,
                        data_dir=data_dir, revision=revision)
        return _NameCaptureDS()

    mod = types.ModuleType("datasets")
    mod.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return captured


def test_hf_source_name_passthrough(tmp_path, monkeypatch):
    from corpus.download.huggingface import HuggingFaceDownloader

    captured = _install_capturing_datasets(monkeypatch)

    src = Source(id="hf:wikipedia-hi", kind="huggingface",
                 location="wikimedia/wikipedia", name="20231101.hi",
                 languages=("hindi",), category="code")
    staged = HuggingFaceDownloader(max_examples=5).fetch(src, str(tmp_path))
    assert captured["location"] == "wikimedia/wikipedia"
    assert captured["cfg_name"] == "20231101.hi"
    assert captured["data_dir"] is None             # name wale me subpath nahi
    files = os.listdir(staged.root)
    assert len(files) == 1 and files[0].endswith(".txt")


def test_hf_name_empty_passes_none(monkeypatch, tmp_path):
    # github-code jaise bina-name sources par behavior unchanged (name=None)
    from corpus.download.huggingface import HuggingFaceDownloader

    captured = _install_capturing_datasets(monkeypatch)

    src = Source(id="hf:ghcode-py", kind="huggingface",
                 location="codeparrot/github-code", subpath="Python-all",
                 languages=("python",), category="code")
    HuggingFaceDownloader(max_examples=5).fetch(src, str(tmp_path))
    assert captured["cfg_name"] is None
    assert captured["data_dir"] == "Python-all"


# ------------------------------------------------------------------- #
# Special tokens — save se PEHLE register, W3 IDs consistent
# ------------------------------------------------------------------- #

@pytest.fixture()
def tiny_tokenizer_file(tmp_path):
    """Script ke asli code-path se chhota tokenizer.json banata hai."""
    from w1_train_tokenizer import main as tok_main

    raw = tmp_path / "stage" / "s1"
    raw.mkdir(parents=True)
    (raw / "a.txt").write_text("hello world def return value " * 120,
                               encoding="utf-8")
    out = tmp_path / "tok" / "tokenizer.json"
    rc = tok_main(["--raw", str(tmp_path / "stage"), "--vocab", "300",
                   "--sample-mb", "1", "--out", str(out)])
    assert rc == 0
    return out


def test_special_tokens_registered_before_save(tiny_tokenizer_file):
    out = tiny_tokenizer_file
    data = json.load(open(out, encoding="utf-8"))
    sp = data["special_tokens"]
    assert set(DEFAULT_SPECIAL_TOKENS) <= set(sp), \
        f"specials missing: {set(DEFAULT_SPECIAL_TOKENS) - set(sp)}"
    ids = [sp[t] for t in DEFAULT_SPECIAL_TOKENS]
    assert len(set(ids)) == len(ids)                    # unique ids
    # specials merge-space ke THEEK baad contiguous append hote hain —
    # (chhota corpus merges jaldi khatam kar sakta hai, isliye requested
    # vocab se compare NAHI karte)
    first_special = 256 + len(data["merges"])
    assert min(ids) == first_special and max(ids) - min(ids) == len(ids) - 1


def test_w3_template_ids_match_saved_tokenizer(tiny_tokenizer_file):
    from dataset.fim import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
    from evals.chat_template import CHAT_TOKENS, register_chat_tokens

    tok = BPETokenizer.load(str(tiny_tokenizer_file))
    sp = tok.special_tokens

    # W3 ke saare markers saved tokenizer me hon hi chahiye
    assert set(CHAT_TOKENS) <= set(sp)
    assert {FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE} <= set(sp)

    # register_chat_tokens idempotent — same ids, koi shift nahi
    reg = register_chat_tokens(tok)
    assert all(reg[t] == sp[t] for t in CHAT_TOKENS)

    # encode specials ko ATOMIC pehchan-ta hai (single id, split nahi)
    enc = tok.encode("<|system|>x<|end|>")
    assert enc[0] == sp["<|system|>"]
    assert enc[-1] == sp["<|end|>"]
    assert "<|system|>" == tok.decode([sp["<|system|>"]])

    # meta file + _DONE marker provenance carry karte hain
    meta = json.load(open(str(tiny_tokenizer_file) + ".meta.json",
                          encoding="utf-8"))
    assert meta["special_tokens"] == sp
    done = json.load(open(os.path.join(os.path.dirname(tiny_tokenizer_file),
                                       "_DONE"), encoding="utf-8"))
    assert done["vocab_size"] == tok.vocab_size
    assert done["specials"] == len(sp)


def test_vocab_size_includes_specials(tiny_tokenizer_file):
    data = json.load(open(tiny_tokenizer_file, encoding="utf-8"))
    tok = BPETokenizer.load(str(tiny_tokenizer_file))
    expected = 256 + len(data["merges"]) + len(data["special_tokens"])
    assert tok.vocab_size == expected                   # model isi ko dekhega


# ------------------------------------------------------------------- #
# Efficiency harness
# ------------------------------------------------------------------- #

def test_build_val_split_per_source_deterministic(tmp_path):
    from w1_build_corpus import build_val_split

    stage = tmp_path / "stage"
    for src, files in (("hf_hi", 6), ("gh_repo", 3)):
        d = stage / src
        d.mkdir(parents=True)
        for i in range(files):
            (d / f"f{i}.txt").write_text(f"{src} {i}\n", encoding="utf-8")
    (stage / "_DONE").write_text("{}", encoding="utf-8")   # marker skip
    val = tmp_path / "val_src"
    n = build_val_split(str(stage), str(val), count=4)
    assert n == 7                       # hi capped at 4, repo poori 3
    assert sorted(os.listdir(val / "hf_hi")) == [
        "f0.txt", "f1.txt", "f2.txt", "f3.txt"]
    assert os.path.exists(val / "_VAL_DONE")
    # idempotent — dobara call par marker se wahi count
    assert build_val_split(str(stage), str(val), count=4) == 7


def test_efficiency_report_structure_and_totals(tmp_path):
    from w1_tokenizer_efficiency import efficiency_report, load_texts

    val = tmp_path / "val_src"
    hi = val / "hf_wikipedia-hi"
    en = val / "hf_ghcode-python"
    hi.mkdir(parents=True)
    en.mkdir(parents=True)
    (hi / "d.txt").write_text("यह एक हिंदी वाक्य है।\n" * 50, encoding="utf-8")
    (en / "c.py").write_text("def add(a, b):\n    return a + b\n" * 50,
                             encoding="utf-8")

    texts = load_texts(str(val))
    assert set(texts) == {"hf_wikipedia-hi", "hf_ghcode-python"}

    tok = BPETokenizer()
    tok.train(["hello def return ", "यह एक हिंदी वाक्य"], vocab_size=260)
    rep = efficiency_report(tok, texts)
    for src in ("hf_wikipedia-hi", "hf_ghcode-python"):
        r = rep[src]
        assert r["files"] >= 1 and r["chars"] > 0 and r["tokens"] > 0
        assert r["bytes"] >= r["chars"]                 # UTF-8 multibyte
        assert 0 < r["tokens_per_char"] and 0 < r["chars_per_token"]
        assert 0 < r["bytes_per_token"]
    agg = rep["_aggregate"]
    assert agg["chars"] == sum(rep[s]["chars"] for s in texts)
    assert agg["tokens"] == sum(rep[s]["tokens"] for s in texts)


def test_efficiency_cli_writes_report(tmp_path, capsys):
    from w1_tokenizer_efficiency import main as eff_main

    val = tmp_path / "val_src" / "s1"
    val.mkdir(parents=True)
    (val / "x.py").write_text("def x(): pass\n" * 30, encoding="utf-8")
    tokf = tmp_path / "tokenizer.json"
    t = BPETokenizer()
    t.train(["def x(): pass\n"], vocab_size=256)
    t.save(str(tokf))
    report = tmp_path / "eff.json"
    rc = eff_main(["--tok", str(tokf), "--val",
                   str(tmp_path / "val_src"), "--out", str(report)])
    assert rc == 0
    data = json.load(open(report, encoding="utf-8"))
    assert "s1" in data and "_aggregate" in data
