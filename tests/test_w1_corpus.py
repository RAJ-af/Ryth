"""Offline tests for W1 corpus tooling.

Real network kabhi nahi: HF downloader ko fake `datasets` module ke saath
inject karke test karte hain; baaki sab `local` sources + tmp dirs.
Run:  python3 -m pytest tests/test_w1_corpus.py -v
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


class _FakeDS:
    """load_dataset ka stand-in — do rows, phir ruk jaata hai."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets(monkeypatch_rows):
    """sys.modules me nakli `datasets` bhejo jiska load_dataset rows deta hai."""
    calls = {}

    def fake_load(name, split=None, streaming=None, data_dir=None):
        calls["name"] = name
        calls["split"] = split
        calls["streaming"] = streaming
        calls["data_dir"] = data_dir
        return _FakeDS(monkeypatch_rows)

    mod = types.ModuleType("datasets")
    mod.load_dataset = fake_load
    sys.modules["datasets"] = mod
    return calls


def test_hf_fetch_uses_subpath_as_data_dir(tmp_path):
    from corpus.download.huggingface import HuggingFaceDownloader
    from corpus.sources.registry import Source

    calls = _install_fake_datasets([
        {"content": "int main(){return 0;}", "lang": "c"},
        {"content": "", "lang": "c"},                      # khali row skip
    ])
    src = Source(id="hf:stack-c", kind="huggingface",
                 location="bigcode/the-stack-dedup",
                 languages=("c",), subpath="data/c")
    dl = HuggingFaceDownloader(max_examples=10)
    staged = dl.fetch(src, str(tmp_path))
    assert calls["data_dir"] == "data/c"
    assert calls["streaming"] is True
    files = sorted(os.listdir(staged.root))
    assert len(files) == 1 and files[0].endswith(".c")     # c ext ab mapped hai


def test_hf_max_bytes_stops_streaming(tmp_path):
    from corpus.download.huggingface import HuggingFaceDownloader
    from corpus.sources.registry import Source

    big = "x" * 1000
    _install_fake_datasets([{"content": big} for _ in range(50)])
    src = Source(id="hf:stack-py", kind="huggingface",
                 location="bigcode/the-stack-dedup",
                 languages=("python",), subpath="data/python")
    dl = HuggingFaceDownloader(max_bytes=3000)
    staged = dl.fetch(src, str(tmp_path))
    total = sum(os.path.getsize(os.path.join(staged.root, f))
                for f in os.listdir(staged.root))
    assert 3000 <= total < 3000 + 1100                     # ek file overshoot tak


def test_hf_unlimited_when_no_budget(tmp_path):
    from corpus.download.huggingface import HuggingFaceDownloader
    from corpus.sources.registry import Source

    _install_fake_datasets([{"content": "print(1)\n"}] * 5)
    src = Source(id="hf:t", kind="huggingface", location="x/y",
                 languages=("python",))
    dl = HuggingFaceDownloader(max_examples=5)
    staged = dl.fetch(src, str(tmp_path))
    assert len(os.listdir(staged.root)) == 5               # purana behaviour intact


def test_w1_probe_counts_license_histogram():
    from w1_probe_stack import w1_probe_stack

    _install_fake_datasets([
        {"content": "a", "lang": "c", "license": "mit"},
        {"content": "b", "lang": "c", "license": "mit"},
        {"content": "c", "lang": "c", "license": None},
        {"content": "d", "lang": "c"},                     # license column hi nahi
    ])
    out = w1_probe_stack("c", limit=10)
    assert out["rows"] == 4
    assert "license" in out["columns"]
    assert out["license_histogram"] == {"mit": 2, "unknown": 2}


def test_w1_sources_json_valid_against_registry():
    from corpus.sources.registry import Source

    cfg = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "w1_sources.json")
    entries = json.load(open(cfg, encoding="utf-8"))
    assert isinstance(entries, list) and len(entries) >= 2
    ids = set()
    for e in entries:
        s = Source(**e)                                    # schema validate ho gaya
        assert s.enabled and s.kind in ("huggingface", "github")
        ids.add(s.id)
    assert any(e["subpath"] == "data/python" for e in entries)
    assert any(e["subpath"] == "data/c" for e in entries)
    assert len(ids) == len(entries)                        # unique ids
