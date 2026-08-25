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

    def fake_load(name, split=None, streaming=None, data_dir=None,
                  revision=None):
        calls["name"] = name
        calls["split"] = split
        calls["streaming"] = streaming
        calls["data_dir"] = data_dir
        calls["revision"] = revision
        return _FakeDS(monkeypatch_rows)

    mod = types.ModuleType("datasets")
    mod.load_dataset = fake_load
    sys.modules["datasets"] = mod
    return calls


def _install_gated_fake_datasets(gated_name):
    """Primary GATED behave kare (401/auth error), baaki sources rows dein."""
    calls = []

    def fake_load(name, split=None, streaming=None, data_dir=None,
                  revision=None):
        calls.append((name, data_dir, revision))
        if name == gated_name:
            raise RuntimeError(f"{gated_name} is gated: 401 authentication "
                               "required — accept gate at the hub")
        return _FakeDS([{"code": "int main(){return 0;}", "license": "isc"}])

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
    # ungated primary (stack-dedup/starcoderdata dono gated=auto hain):
    # codeparrot/github-code parquet-mirror, per-language dirs + revision
    assert all(e["location"] == "codeparrot/github-code" for e in entries)
    assert {e["subpath"] for e in entries} == {"Python-all", "C-all"}
    assert all(e.get("ref") == "refs/convert/parquet" for e in entries)
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


# ---- W1 review-fix regressions (2026-08-25 independent review) -----------


def test_merge_manifests_rejects_mismatched_config(tmp_path):
    # part manifests me config alag ho toh SILENT merge nahi — loud fail (review #3)
    from w1_pack_rds import merge_manifests

    p1 = _mini_rds_part(tmp_path, "a")
    p2 = _mini_rds_part(tmp_path, "b")
    mf_path = os.path.join(p2, "manifest.json")
    mf = json.load(open(mf_path, encoding="utf-8"))
    mf["seq_len"] = 999                                    # part-0 se alag
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(mf, f)
    try:
        merge_manifests([p1, p2], str(tmp_path / "merged"), extra_meta={})
        raise AssertionError("expected SystemExit on mismatched config")
    except SystemExit as e:
        assert "seq_len" in str(e)


def test_merged_manifest_carries_seq_len_override(tmp_path):
    from w1_pack_rds import merge_manifests

    p1 = _mini_rds_part(tmp_path, "a")
    p2 = _mini_rds_part(tmp_path, "b")
    out = str(tmp_path / "m2")
    merge_manifests([p1, p2], out, extra_meta={"seq_len": 1024})
    back = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert back["seq_len"] == 1024                         # RDSDataset isi se padhta hai


def test_stage_repos_repairs_dangling_symlink(tmp_path):
    # crash ke baad bacha dangling link → FileExistsError nahi, repair (review #4)
    import w1_pack_rds as pr

    repo = tmp_path / "repo_a"
    repo.mkdir()
    (repo / "m.py").write_text("x=1\n", encoding="utf-8")
    part_in = tmp_path / "in_00"
    part_in.mkdir()
    dst = part_in / "repo_a"
    os.symlink(str(tmp_path / "gone"), str(dst))           # dangling
    pr._stage_repos([str(repo)], str(part_in))
    assert os.path.realpath(dst) == os.path.realpath(str(repo))


def test_stage_repos_falls_back_to_copytree(tmp_path, monkeypatch):
    import w1_pack_rds as pr

    repo = tmp_path / "repo_b"
    repo.mkdir()
    (repo / "m.c").write_text("int x;\n", encoding="utf-8")
    part_in = tmp_path / "in_01"
    part_in.mkdir()

    def no_symlink(*a, **k):                               # FS symlink support nahi
        raise OSError("symlink unsupported")

    monkeypatch.setattr(os, "symlink", no_symlink)
    pr._stage_repos([str(repo)], str(part_in))
    assert (part_in / "repo_b" / "m.c").exists()           # real copy hui


def test_dirty_part_out_wiped_before_rerun(tmp_path):
    # crash mid-part ke baad stale shards resume ko corrupt na karein (review #5)
    import w1_pack_rds as pr

    part_out = tmp_path / "part_00"
    part_out.mkdir()
    (part_out / "stale_shard_099.rds").write_bytes(b"junk")
    assert pr._wipe_if_dirty(str(part_out)) is True
    assert not part_out.exists()                           # wipe ho gaya
    part_out.mkdir()
    (part_out / "_DONE").write_text("{}")
    assert pr._wipe_if_dirty(str(part_out)) is False       # DONE wala untouched


def test_stratified_sample_is_walk_order_independent(tmp_path, monkeypatch):
    # same seed + same files ⇒ same sample, chahe os.walk ka order kuch bhi ho (review #6)
    import w1_train_tokenizer as wt

    for i in range(6):
        # unique content — warna order-independent result trivially pass ho jata
        (tmp_path / f"f{i}.py").write_text(f"def f{i}(): pass\n" * 10,
                                           encoding="utf-8")
    real_walk = os.walk

    def reversed_walk(root, **kw):
        for dp, dn, fn in real_walk(root):
            yield dp, dn, list(reversed(fn))               # ulta order

    monkeypatch.setattr(wt.os, "walk", reversed_walk)
    t_rev = wt.stratified_sample(str(tmp_path), 10 ** 9, seed=7)
    monkeypatch.undo()
    t_fwd = wt.stratified_sample(str(tmp_path), 10 ** 9, seed=7)
    assert t_rev == t_fwd                                  # reproducible sample


def test_stratified_sample_skips_marker_files(tmp_path):
    from w1_train_tokenizer import stratified_sample

    (tmp_path / "a.py").write_text("def f(): pass\n" * 5, encoding="utf-8")
    (tmp_path / "_DONE").write_text('{"files": 999999}', encoding="utf-8")
    (tmp_path / "_SUMMARY.json").write_text('{"total": 1}', encoding="utf-8")
    texts = stratified_sample(str(tmp_path), 10 ** 6, seed=0)
    assert len(texts) == 1 and "999999" not in texts[0]    # markers sample me nahi


def test_discover_repo_dirs_ignores_marker_only_dirs(tmp_path):
    from w1_pack_rds import _discover_repo_dirs

    real = tmp_path / "real_repo"
    real.mkdir()
    (real / "m.py").write_text("x=1\n", encoding="utf-8")
    ghost = tmp_path / "ghost_repo"
    ghost.mkdir()
    (ghost / "_DONE").write_text("{}", encoding="utf-8")
    names = [os.path.basename(g) for g in _discover_repo_dirs(str(tmp_path))]
    assert "real_repo" in names and "ghost_repo" not in names


def test_plan_budget_sums_exactly_and_rejects_bad_weight():
    from w1_build_corpus import plan_budget

    entries = [{"id": "a", "weight": 1}, {"id": "b", "weight": 1},
               {"id": "c", "weight": 1}]
    budgets = plan_budget(entries, total_bytes=1000)
    assert sum(budgets.values()) == 1000                   # remainder last ko gaya
    assert budgets == {"a": 333, "b": 333, "c": 334}
    try:
        plan_budget([{"id": "z", "weight": 0}], 100)
        raise AssertionError("expected rejection for weight<1")
    except ValueError:
        pass


def test_summary_rewritten_when_config_changes(tmp_path):
    # --per-source-bytes badla par purani _SUMMARY.json stale nahi reh sakti (review #10)
    from w1_build_corpus import build

    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    out = tmp_path / "out"
    kw = dict(config=None, input=str(inp), out=str(out), total_gb=0.000001,
              seed=7)
    build(types.SimpleNamespace(per_source_bytes=400, **kw))
    build(types.SimpleNamespace(per_source_bytes=200, **kw))
    back = json.load(open(out / "_SUMMARY.json", encoding="utf-8"))
    assert back["config"]["per_source_bytes"] == 200       # fingerprint fresh hai


def test_kaggle_train_smoke_overrides_eff_tokens():
    # smoke override ke BAAD eff_tokens dobara compute hona chahiye (review #7)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kaggle_train3", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "kaggle_train.py"))
    kt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kt)
    ns = kt.build_parser().parse_args([])
    kt.resolve_args(ns)                                    # production 262144
    kt.apply_smoke(ns)
    assert ns.micro_batch == 8 and ns.grad_accum == 2 and ns.seq_len == 128
    assert ns.eff_tokens == 8 * 2 * 128                    # 2048, stale 262144 NAHI


# ------------------------------------------------------------------- #
# gated-source / fallback — Kaggle par bina HF_TOKEN ke chale (2026-08-25)
# bigcode/the-stack-dedup AUR starcoderdata dono gated=auto hain; ungated
# verified alternative: codeparrot/github-code @ refs/convert/parquet
# (columns: code/language/license/path/repo_name/size).
# ------------------------------------------------------------------- #

def test_hf_downloader_falls_back_when_primary_gated(tmp_path):
    from corpus.download.huggingface import HuggingFaceDownloader
    from corpus.sources.registry import Source

    calls = _install_gated_fake_datasets("bigcode/the-stack-dedup")
    src = Source(id="hf:stack-c", kind="huggingface",
                 location="bigcode/the-stack-dedup", languages=("c",),
                 subpath="data/c")
    dl = HuggingFaceDownloader()
    staged = dl.fetch(src, str(tmp_path),
                      fallbacks=[{"location": "codeparrot/github-code",
                                  "subpath": "C-all",
                                  "revision": "refs/convert/parquet"}])
    assert len(os.listdir(staged.root)) == 1               # fallback se aaya
    assert calls[0][0] == "bigcode/the-stack-dedup"        # pehle primary try
    assert calls[-1] == ("codeparrot/github-code", "C-all",
                         "refs/convert/parquet")           # phir fallback
    # revision sirf fallback me tha; primary ka None hi jaata hai
    assert calls[0][2] is None


def test_hf_gated_without_fallback_is_actionable_error(tmp_path):
    import pytest

    from corpus.download.huggingface import DownloadError, HuggingFaceDownloader
    from corpus.sources.registry import Source

    _install_gated_fake_datasets("bigcode/the-stack-dedup")
    src = Source(id="hf:stack-py", kind="huggingface",
                 location="bigcode/the-stack-dedup", languages=("python",))
    dl = HuggingFaceDownloader()
    with pytest.raises(DownloadError, match="gated|HF_TOKEN|fallback"):
        dl.fetch(src, str(tmp_path))


def test_probe_streams_configured_source_not_hardcoded(tmp_path):
    # probe ab configs/w1_sources.json se source uthata hai — hardcoded
    # stack-dedup kabhi nahi (gated tha); ref -> revision mapping yahan
    from w1_probe_stack import w1_probe_stack

    calls = _install_fake_datasets([
        {"code": "x=1", "license": "mit"},
        {"code": "y=2", "license": None},
    ])
    cfg = tmp_path / "src.json"
    cfg.write_text(json.dumps([
        {"id": "hf:x", "kind": "huggingface",
         "location": "codeparrot/github-code", "languages": ["c"],
         "category": "code", "subpath": "C-all",
         "ref": "refs/convert/parquet"}]), encoding="utf-8")
    out = w1_probe_stack("c", limit=5, config=str(cfg))
    assert out["rows"] == 2 and out["served_location"] == \
        "codeparrot/github-code"
    assert calls["data_dir"] == "C-all"
    assert calls["revision"] == "refs/convert/parquet"


def test_stage_download_tolerates_extra_config_keys(tmp_path):
    # config entry me naye keys ('fallbacks') ho to Source(**entry) TypeError
    # na de — from_dict filtering hi config-extensibility ka rasta hai
    _install_gated_fake_datasets("nobody/unused")
    from w1_build_corpus import _stage_download

    entry = {"id": "hf:t", "kind": "huggingface", "location": "x/y",
             "languages": ["c"], "category": "code", "license_hint": "MIT",
             "fallbacks": [{"location": "fb/alt", "revision": "r1"}]}
    out_root = tmp_path / "stage"
    res = _stage_download(entry, budget=10_000, stage_root=str(out_root),
                          local_input=None)
    assert res["files"] >= 1 and res["bytes"] > 0
    marker = os.path.join(out_root, "hf_t", "_DONE")
    assert os.path.exists(marker)                          # idempotent marker


def test_downloader_default_has_no_silent_example_cap(tmp_path):
    # max_examples=5000 default 1.2GB budget ko ~50MB par kaat deta tha —
    # >=600M token target ke liye default UNLIMITED hona chahiye (budget
    # hi cap hai)
    from corpus.download.huggingface import HuggingFaceDownloader
    from corpus.sources.registry import Source

    _install_fake_datasets([{"content": "print(1)\n"}] * 7000)
    src = Source(id="hf:many", kind="huggingface", location="x/y",
                 languages=("python",))
    dl = HuggingFaceDownloader(max_bytes=None)
    staged = dl.fetch(src, str(tmp_path))
    assert len(os.listdir(staged.root)) == 7000            # sab staged


# ------------------------------------------------------------------- #
# fresh-clone script invocation — python3 scripts/*.py ko repo-root na mile
# to `corpus`/`tokenizer`/`dataset` packages import nahi ho sakte the
# (Kaggle A1 probe fail: ModuleNotFoundError 'corpus'; kaggle_train.py me
# pehle se bootstrap tha, baaki chaar me nahi)
# ------------------------------------------------------------------- #

_SCRIPTS_NEEDING_BOOTSTRAP = ("w1_probe_stack", "w1_build_corpus",
                              "w1_pack_rds", "w1_train_tokenizer")


def test_entry_scripts_bootstrap_repo_root():
    # har W1 entry-script apne __file__ se repo-root sys.path me rakhe —
    # machine-specific path hack nahi, clone-relative discovery
    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts")
    for name in _SCRIPTS_NEEDING_BOOTSTRAP:
        src = open(os.path.join(base, name + ".py"), encoding="utf-8").read()
        assert "_REPO not in sys.path" in src, f"{name}: bootstrap missing"


def _fresh_clone_probe_harness(tmp_path):
    """Minimal fresh-clone + fake-datasets setup; (child_code, env) returns.

    EXACT Kaggle invocation simulate: `python3 <clone>/scripts/w1_probe_stack.py`
    -S = site/editable-hooks OFF (dev-machine par installed ryth bug chhupa
    deta tha), PYTHONPATH khali, cwd bahaar. Provenance sentinel prove karta
    hai ki `corpus` CLONE se aaya, installed package se nahi.
    """
    import shutil

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # minimal FRESH CLONE: sirf probe script + corpus tree + sentinel
    clone = tmp_path / "ryth"
    (clone / "scripts").mkdir(parents=True)
    shutil.copy(os.path.join(repo, "scripts", "w1_probe_stack.py"),
                clone / "scripts" / "w1_probe_stack.py")
    shutil.copytree(os.path.join(repo, "corpus"), clone / "corpus",
                    ignore=shutil.ignore_patterns("__pycache__"))
    sentinel = tmp_path / "sentinel.txt"
    init = clone / "corpus" / "__init__.py"
    orig_init = init.read_text(encoding="utf-8")
    init.write_text(orig_init +
                    f"\nopen({str(sentinel)!r}, 'a').write('clone-import\\n')\n",
                    encoding="utf-8")

    cfg = tmp_path / "src.json"
    cfg.write_text(json.dumps([
        {"id": "hf:x", "kind": "huggingface",
         "location": "codeparrot/github-code", "languages": ["c"],
         "category": "code", "subpath": "C-all"}]), encoding="utf-8")

    # fake datasets bhi CLONE ke andar hi rakho (-S ke baad sirf clone paths)
    ds_dir = clone / "datasets"
    ds_dir.mkdir()
    (ds_dir / "__init__.py").write_text(
        "class _DS:\n"
        "    def __iter__(self):\n"
        "        return iter([{'code': 'x=1', 'license': 'mit'},\n"
        "                     {'code': 'y=2'}])\n"
        "def load_dataset(*a, **k):\n"
        "    return _DS()\n", encoding="utf-8")

    probe = str(clone / "scripts" / "w1_probe_stack.py")
    child_code = (
        "import runpy, sys\n"
        f"sys.argv = [{probe!r}, '--subset', 'c', '--limit', '2', "
        f"'--config', {str(cfg)!r}]\n"
        f"runpy.run_path({probe!r}, run_name='__main__')\n")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return child_code, env


def test_probe_runs_as_fresh_clone_subprocess(tmp_path):
    import subprocess

    child_code, env = _fresh_clone_probe_harness(tmp_path)
    r = subprocess.run([sys.executable, "-S", "-c", child_code],
                       cwd=str(tmp_path), env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, \
        f"rc={r.returncode}\nSTDERR:\n{r.stderr[-800:]}"
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["rows"] == 2 and out["columns"] == ["code", "license"]
    sentinel = tmp_path / "sentinel.txt"
    assert sentinel.exists() and "clone-import" in sentinel.read_text(), \
        "`corpus` clone se resolve NAHI hua (installed/package-hook jeet gaya)"


# ------------------------------------------------------------------- #
# C-probe rc=134 (SIGABRT) — HF/pyarrow streaming teardown race.
# Lakshman-rekha: JSON print ho chuka tha uske BAAD native abort — yaani
# kaam poora tha, crash interpreter-shutdown ke background-thread cleanup
# me tha (rc=134 != 137 OOM; koi Python traceback nahi). datasets ka koi
# public close API nahi -> CLI mains controlled hard-exit karte hain.
# ------------------------------------------------------------------- #

def test_teardown_safe_exit_skips_racy_interpreter_teardown(tmp_path):
    # Miniature repro: atexit-handler wahi racy teardown simulate karta hai
    # (marker likhta hai + os.abort => SIGABRT rc=134). Normal exit hote to
    # yahi abort milta; teardown_safe_exit os._exit karta hai to handler
    # KABHI nahi chalta -> rc=0, output flushed, marker absent.
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_file = tmp_path / "out.txt"
    crash_marker = tmp_path / "teardown-ran.txt"

    def child(use_safe_exit: bool) -> str:
        exit_line = ("teardown_safe_exit(0)" if use_safe_exit
                     else "pass  # normal return -> atexit chalta hai")
        return (
            "import atexit, sys\n"
            f"def _racy_teardown():\n"
            f"    open({str(crash_marker)!r}, 'w').write('ran')\n"
            "    sys.stderr.write('terminate called without an active "
            "exception\\n')\n"
            "    import os\n"
            "    os.abort()\n"
            "atexit.register(_racy_teardown)\n"
            f"sys.path.insert(0, {repo!r})\n"
            "from corpus.download.huggingface import teardown_safe_exit\n"
            f"f = open({str(out_file)!r}, 'w'); "
            "f.write('probe-json-done\\n'); f.flush()\n"
            f"{exit_line}\n")

    # CONTROL (red dikhata hai): bina safe-exit ke racy teardown fire hota hai
    rc = subprocess.run([sys.executable, "-S", "-c", child(False)],
                        capture_output=True, text=True, timeout=60)
    assert rc.returncode != 0 and crash_marker.exists(), \
        "control case abort nahi hua — harness khud broken hai"

    # FIX (green): teardown_safe_exit abort-susceptible teardown ko skip karta hai
    crash_marker.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-S", "-c", child(True)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, \
        f"rc={r.returncode}\nSTDERR:\n{r.stderr[-800:]}"
    assert "terminate called" not in r.stderr
    assert out_file.read_text(encoding="utf-8") == "probe-json-done\n"
    assert not crash_marker.exists(), "atexit teardown skip hona chahiye tha"


def test_streaming_cli_mains_wire_teardown_safe_exit():
    # dono pyarrow-streaming CLI mains success path par hard-exit karte hain —
    # warna JSON/summary print ke BAAD wahi rc=134 abort A1 ko fail karta
    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts")
    for name in ("w1_probe_stack", "w1_build_corpus"):
        src = open(os.path.join(base, name + ".py"), encoding="utf-8").read()
        assert "teardown_safe_exit" in src, \
            f"{name}: main me teardown_safe_exit wiring missing"


def test_probe_subprocess_exits_cleanly_repeatedly(tmp_path):
    # stability: same invocation N baar — har baar rc=0 (race environment-
    # specific tha, isliye ek run ka pass kaafi nahi)
    import subprocess

    for i in range(3):
        tp = tmp_path / f"run{i}"
        tp.mkdir()
        child_code, env = _fresh_clone_probe_harness(tp)
        r = subprocess.run([sys.executable, "-S", "-c", child_code],
                           cwd=str(tp), env=env, capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0, \
            f"run{i}: rc={r.returncode}\nSTDERR:\n{r.stderr[-800:]}"
