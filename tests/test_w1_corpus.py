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


def test_plan_budget_proportional_split():
    from w1_build_corpus import plan_budget

    entries = [{"id": "a", "weight": 1}, {"id": "b", "weight": 3}]
    got = plan_budget(entries, total_bytes=1000)
    assert got["a"] == 250 and got["b"] == 750              # 1:3


def test_build_local_sources_idempotent(tmp_path):
    from w1_build_corpus import build

    inp = tmp_path / "in"; inp.mkdir()
    (inp / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (inp / "util.c").write_text("int f(){return 1;}\n", encoding="utf-8")
    out = tmp_path / "out"
    args = types.SimpleNamespace(
        config=None, input=str(inp), out=str(out), total_gb=0.000001,
        per_source_bytes=400, seed=7)
    s1 = build(args)
    assert s1["total_bytes"] > 0
    assert (out / "_SUMMARY.json").exists()
    # dobara chalao — kuch badla nahi (idempotent)
    mtime = (out / "_SUMMARY.json").stat().st_mtime_ns
    s2 = build(args)
    assert s2 == s1
    assert (out / "_SUMMARY.json").stat().st_mtime_ns == mtime


def test_stratified_sample_roundrobins_languages(tmp_path):
    from w1_train_tokenizer import stratified_sample

    for i in range(4):
        (tmp_path / f"m{i}.py").write_text("def f(): pass\n" * 20, encoding="utf-8")
        (tmp_path / f"k{i}.c").write_text("int main(){}\n" * 20, encoding="utf-8")
    texts = stratified_sample(str(tmp_path), target_chars=100000, seed=1)
    py = sum(1 for t in texts if t.lstrip().startswith("def "))
    c = len(texts) - py
    assert py == 4 and c == 4                              # dono languages poori


def test_time_probe_returns_positive_rate(tmp_path):
    from w1_train_tokenizer import stratified_sample, time_probe
    from tokenizer.bpe import BPETokenizer

    (tmp_path / "a.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    texts = stratified_sample(str(tmp_path), target_chars=10 ** 9, seed=0)
    rate = time_probe(texts[:1], tok=BPETokenizer())
    assert rate > 0


def _mini_rds_part(tmp_path, tag):
    """Ek chhota sa part-dir banao jisme valid manifest + 1 shard ho (fixture).

    Real ShardManager use karte hain (dataset/sharding.py) — wahi manifest
    format likhta hai jo merge_manifests() padhega.
    """
    from dataset.config import RDEConfig
    from dataset.sharding import ShardManager
    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer(); tok.train(["hello world"], vocab_size=350)
    part = str(tmp_path / f"part_{tag}"); os.makedirs(part, exist_ok=True)
    sm = ShardManager(part, RDEConfig(seq_len=4), tok)
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    sm.add_chunk(ids[:4], {"repo": f"r{tag}"})
    sm.add_chunk(ids[4:], {"repo": f"r{tag}"})
    sm.finalize({"chunks": 2}, lock={"tag": tag})
    return part


def test_merge_manifests_concatenates_shards(tmp_path):
    from w1_pack_rds import merge_manifests

    p1 = _mini_rds_part(tmp_path, "a"); p2 = _mini_rds_part(tmp_path, "b")
    out = str(tmp_path / "merged")
    mm = merge_manifests([p1, p2], out, extra_meta={"note": "w1"})
    assert len(mm["shards"]) == 2
    ds_paths = [os.path.join(out, s["file"]) for s in mm["shards"]]
    assert all(os.path.exists(p_) for p_ in ds_paths)      # files copy hue
    back = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert back["note"] == "w1"


def test_merged_dir_loadable_by_rdsdataset(tmp_path):
    from dataset.dataset import RDSDataset
    from w1_pack_rds import merge_manifests

    p1 = _mini_rds_part(tmp_path, "a"); p2 = _mini_rds_part(tmp_path, "b")
    out = str(tmp_path / "merged")
    mm = merge_manifests([p1, p2], out, extra_meta={})
    ds = RDSDataset(out)
    assert len(ds) == sum(s.get("chunks", 0) for s in mm["shards"])


def test_kaggle_train_defaults_are_production():
    # argparse defaults inspect karo (model banana mehenga hai — sirf parser)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kaggle_train", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "kaggle_train.py"))
    kt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kt)                            # main defined only
    ns = kt.build_parser().parse_args([])
    kt.resolve_args(ns)
    assert ns.vocab == 24576 and ns.seq_len == 1024
    assert ns.lr == 6e-4 and ns.warmup == 2000
    assert ns.micro_batch == 16 and ns.grad_accum == 16 and ns.steps == 8000
    assert ns.dtype is None                                # auto; notebook fp16 deta hai
    assert ns.eff_tokens == 16 * 16 * 1024                 # 262144 tokens/step


def test_kaggle_train_resolve_overrides_eff_tokens():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kaggle_train2", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "kaggle_train.py"))
    kt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kt)
    ns = kt.build_parser().parse_args(
        ["--micro_batch", "4", "--grad_accum", "2"])
    kt.resolve_args(ns)
    assert ns.eff_tokens == 4 * 2 * ns.seq_len             # override ke saath sahi


def test_notebook_is_valid_json_with_w1_cells():
    nb_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notebooks", "ryth_kaggle_train.ipynb")
    nb = json.load(open(nb_path, encoding="utf-8"))
    assert nb["nbformat"] >= 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("w1_build_corpus.py", "w1_train_tokenizer.py",
                   "w1_pack_rds.py", "kaggle_train.py"):
        assert needle in src, f"notebook missing {needle}"
