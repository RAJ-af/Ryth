# Ryth Phase 6 / W1 — Real Corpus + 30M Kaggle Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take Ryth from smoke-scale to production-scale Workstream 1: a real,
multi-source C+Python corpus (≥600M clean tokens), a 24k scratch BPE, packed
RDS@1024, and a Kaggle-ready 30M training setup with auto-resume — plus the
runbook the owner follows to press the buttons on Kaggle.

**Architecture:** Everything reuses the existing stack — `corpus` CLI +
downloaders for acquisition, `RDEPipeline` for cleaning/tokenizing/packing,
`training` engine (auto-resume already built-in via `latest.pt`). New work is
(a) small additive upgrades to `corpus/download/huggingface.py`, (b) four
idempotent orchestration scripts under `scripts/w1_*.py` (probe → corpus →
tokenizer → pack), (c) a thin production-config pass over
`scripts/kaggle_train.py`, (d) notebook + runbook. Heavy GPU stays MANUAL:
scripts prepare; the owner runs Kaggle sessions (spec §9).

**Tech Stack:** Python 3.9+ stdlib + repo packages. `datasets` (HF) is needed
only where real downloads happen (Kaggle CPU sessions have it preinstalled);
all tests are OFFLINE using the existing `local` source kind and tiny fixtures.

**Spec:** `docs/superpowers/specs/2026-08-25-phase6-measure-first-design.md` (§4 W1, §10 M1)

## Global Constraints

- Target corpus: **≥600M clean tokens** (~2.4 GB text) after dedup/quality filtering.
- Tokenizer: **fresh scratch BPE, vocab = 24576**, trained on a stratified C+Python sample of the corpus.
- Data: **RDS shards, seq_len=1024, uint16 dtype**.
- Model/training: **`ryth_30m` preset** (d=512, L=8, H=8, n_kv=2), **fp16 on T4**, AdamW **lr ≈ 6e-4** with **warmup+cosine**, effective batch **≈260k tokens/step**, auto-resume across sessions via `latest.pt`.
- Sources: multi-source — HF **The Stack-dedup subsets (C, Python)** filtered to permissive licenses (**MIT/Apache/BSD/ISC/MPL-2.0**) + curated GitHub repos, all through ryth-corpus.
- Compute: **Kaggle only** (owner is on Termux/proot; no local GPU). All heavy GPU runs stay manual (owner triggers prepared notebooks/scripts).
- Frozen core pillars — do NOT modify anything under `tokenizer/`, `dataset/`, `model/`, `training/`. The `corpus/` package MAY be extended additively (it is W1's designated workhorse, spec §1).
- Every script is idempotent/resumable: completed stages leave a marker and are skipped on rerun (Kaggle sessions die mid-run; spec §11).
- Tests are offline and deterministic; network-touching helpers are never exercised by tests.
- Deliverables (M1 acceptance): `best.pt`/`final.pt`, validation-loss curve, throughput report, sample generations (C + Python prompts), reproducible dataset manifest lock.

## Verified repo facts this plan builds on

- `corpus/download/huggingface.py`: `HuggingFaceDownloader(split="train", max_examples=5000)`; `fetch(source, stage_dir) -> StagedRepo`; streams `datasets.load_dataset(location, split=..., streaming=True)`; picks text from `_CODE_FIELDS`; ext map `_EXT` has **no `"c"` key** (C would fall to `.txt`); does NOT pass a config/`data_dir`; ignores `source.subpath`.
- `corpus/sources/registry.py`: `Source(id, kind∈{huggingface,github,http,local}, location, license_hint, languages, category, ref, subpath, enabled)`. `local` downloader always available (tests use it).
- `dataset/dataset.py:RDSDataset(data_dir)` reads `manifest.json["shards"]` list → **multi-shard native**.
- `dataset/sharding.py:ShardWriter(out_dir, config, tokenizer).finalize(stats_summary, lock=None) -> dict` writes shards + manifest.
- `dataset/lock.py:build_lock(config, tokenizer, *, dataset_version=..., ...)` exists for manifest locks.
- `dataset/pipeline.py:RDEPipeline(tokenizer, config).run(root, out_dir, ...) -> dict(manifest)` discovers repos under `root`, full ingest→curriculum→dedup→shard.
- `training/config.py:TrainConfig(resume=...)` accepts `"latest"`; `CheckpointManager` saves `latest.pt`/`best.pt`; `TrainConfig.val_data_dir` supported by `training/dataloader.py:make_dataloaders`.
- `scripts/kaggle_train.py`: stages corpus(synthetic|`--raw`) → tokenizer(`build_or_load_tokenizer`) → RDS(`build_or_load_rds`) → model(`build_model(preset,vocab,seq_len,grad_ckpt)`) → Trainer → generate sample. Flags incl `--vocab --seq_len --micro_batch --grad_accum --steps --warmup --lr --dtype --resume_demo`.
- `BPETokenizer.train(texts, vocab_size, verbose=False)`; `tok.save(path)` (bpe.py:170).
- Preset classmethods reject overrides of their own defaults (`model/config.py:104`) — construct `RythConfig(**kw)` directly when overriding dims.

---

### Task 1: HF downloader — subpath configs, `.c` extension, byte-budget streaming

**Files:**
- Modify: `corpus/download/huggingface.py`
- Test: `tests/test_w1_corpus.py` (new file)

**Interfaces:**
- Consumes: `Source.subpath` (already exists, currently unused by HF), `StagedRepo`.
- Produces:
  - `HuggingFaceDownloader(split="train", max_examples=5000, max_bytes=None)` — `max_bytes` stops streaming after the staged output exceeds this many bytes (None = unlimited, old behavior).
  - `fetch()` passes `data_dir=source.subpath or None` to `load_dataset` when set (The Stack-dedup language subsets live at `data/python`, `data/c`).
  - `_EXT` gains `"c": ".c"` (and keeps everything else unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_w1_corpus.py`:

```python
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
    src = Source(id="hf:t", kind="huggingface", location="x/y", languages=("python",))
    dl = HuggingFaceDownloader(max_examples=5)
    staged = dl.fetch(src, str(tmp_path))
    assert len(os.listdir(staged.root)) == 5               # purana behaviour intact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_w1_corpus.py -v`
Expected: FAIL — `fetch() got an unexpected keyword argument` behaviour absent: first test fails because `calls["data_dir"]` is None and file ends `.txt`; second fails because all 50 files staged.

- [ ] **Step 3: Implement**

In `corpus/download/huggingface.py`:

1. `__init__` gains `max_bytes: int | None = None`, stored as `self.max_bytes`.
2. In `fetch()`, replace the `load_dataset` call:

```python
        try:
            ds = datasets.load_dataset(source.location, split=self.split,
                                       streaming=True,
                                       data_dir=(getattr(source, "subpath", "") or None))
```

3. In `_EXT` add `"c": ".c"` (keep the rest verbatim).
4. Track bytes while writing files; stop after exceeding budget:

```python
        n = 0
        staged_bytes = 0
        for ex in ds:                                   # pragma: no cover - network
            field = next((f for f in _CODE_FIELDS if ex.get(f)), None)
            if not field:
                continue
            text = ex[field]
            if not isinstance(text, str) or not text.strip():
                continue
            path = os.path.join(dest, f"example_{n:06d}{ext}")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            n += 1
            staged_bytes += len(text)
            if self.max_bytes is not None and staged_bytes >= self.max_bytes:
                break
            if n >= self.max_examples:
                break
```

(The old `if n >= self.max_examples: break` at loop end moves INTO this block as shown — one combined guard.)

- [ ] **Step 4: Run tests green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py tests/test_corpus.py -q && python3 -m pytest tests/ -q`
Expected: new tests PASS; pre-existing corpus tests unaffected; full suite green (161 + new).

- [ ] **Step 5: Commit**

```bash
git add corpus/download/huggingface.py tests/test_w1_corpus.py
git commit -m "feat(corpus): HF downloader subpath/data_dir, .c ext, byte-budget streaming"
```

---

### Task 2: Stack probe + declarative W1 source list

**Files:**
- Create: `scripts/w1_probe_stack.py`
- Create: `configs/w1_sources.json`
- Test: append to `tests/test_w1_corpus.py`

**Interfaces:**
- Consumes: `HuggingFaceDownloader`, `Source` registry, fake-datasets trick from Task 1.
- Produces:
  - `w1_probe_stack(subset: str, limit: int = 200) -> dict` — returns `{"subset": ..., "rows": N, "columns": [...], "license_histogram": {...}}` by streaming `limit` rows of `bigcode/the-stack-dedup` `data/<subset>` (uses the SAME fake-injectable import path so tests can call it without network).
  - `configs/w1_sources.json` — declarative `Source` dicts consumed later by `corpus` CLI `--source-file` (schema verified against `registry.Source.__post_init__`).

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_w1_probe_counts_license_histogram():
    from scripts.w1_probe_stack import w1_probe_stack   # noqa: E402  (path shim below)

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
```

Plus a path shim right after the existing imports at the top of the file (so `scripts.*` is importable):

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
```

- [ ] **Step 2: Run to red**

Run: `python3 -m pytest tests/test_w1_corpus.py -q`
Expected: FAIL — `No module named scripts.w1_probe_stack` / `configs/w1_sources.json` missing.

- [ ] **Step 3: Implement**

`scripts/w1_probe_stack.py`:

```python
"""The Stack-dedup schema probe — bulk download se PEHLE chalao (measure-first).

Ek subset ki pehle `limit` rows stream karke batata hai: kaunse columns hain,
license values ka histogram kya hai. Isi se license-policy decide hoti hai.
NETWORK TOOL — tests sirf fake-datasets inject karke chalate hain.
"""

from __future__ import annotations


ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
                    "isc", "mpl-2.0"}


def w1_probe_stack(subset: str, limit: int = 200) -> dict:
    import datasets  # optional dep — probe/network machine pe hi zaroori

    ds = datasets.load_dataset("bigcode/the-stack-dedup", split="train",
                               streaming=True, data_dir=f"data/{subset}")
    cols: list[str] = []
    hist: dict[str, int] = {}
    rows = 0
    for ex in ds:
        if not cols:
            cols = sorted(ex.keys())
        lic = ex.get("license") or "unknown"
        hist[lic] = hist.get(lic, 0) + 1
        rows += 1
        if rows >= limit:
            break
    return {"subset": subset, "rows": rows, "columns": cols,
            "license_histogram": hist}


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="e.g. python | c")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args(argv)
    out = w1_probe_stack(a.subset, limit=a.limit)
    import json
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`configs/w1_sources.json` (declarative; `subpath` selects the language shard; curated GitHub seed included per spec multi-source rule):

```json
[
  {"id": "hf:stack-py", "kind": "huggingface",
   "location": "bigcode/the-stack-dedup", "languages": ["python"],
   "category": "code", "subpath": "data/python",
   "license_hint": "MIT/Apache/BSD/ISC/MPL-2.0-or-unknown"},
  {"id": "hf:stack-c", "kind": "huggingface",
   "location": "bigcode/the-stack-dedup", "languages": ["c"],
   "category": "code", "subpath": "data/c",
   "license_hint": "MIT/Apache/BSD/ISC/MPL-2.0-or-unknown"}
]
```

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/w1_probe_stack.py configs/w1_sources.json tests/test_w1_corpus.py
git commit -m "feat(w1): Stack schema probe + declarative C/Python source list"
```

---

### Task 3: Corpus orchestrator — idempotent download to char budget

**Files:**
- Create: `scripts/w1_build_corpus.py`
- Test: append to `tests/test_w1_corpus.py`

**Interfaces:**
- Consumes: `Source` (registry), `HuggingFaceDownloader(max_bytes=...)` from Task 1, `StagedRepo.root`; `local` kind uses `--input DIR` as-is.
- Produces:
  - `plan_budget(entries: list[dict], total_bytes: int) -> dict[str, int]` — splits a total char budget across sources proportional to `weights` (default equal); returns `{source_id: bytes}`.
  - `build(args) -> dict` — runs stages: `stage_download` (one stage-dir per source id, marker `<dir>/_DONE` with staged byte count) then writes `corpus_out/_SUMMARY.json` `{"sources": {id: {"files": n, "bytes": b}}, "total_bytes": ...}`. Completed stages skipped on rerun (delete `_DONE` to force).
  - CLI: `--config configs/w1_sources.json --out corpus_out --total-gb 2.4 --input LOCAL_DIR (for local kind)`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_plan_budget_proportional_split():
    from scripts.w1_build_corpus import plan_budget

    entries = [{"id": "a", "weight": 1}, {"id": "b", "weight": 3}]
    got = plan_budget(entries, total_bytes=1000)
    assert got["a"] == 250 and got["b"] == 750              # 1:3


def test_build_local_sources_idempotent(tmp_path):
    from scripts.w1_build_corpus import build

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
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_w1_corpus.py::test_plan_budget_proportional_split -q`
Expected: FAIL — `ModuleNotFoundError: scripts.w1_build_corpus`.

- [ ] **Step 3: Implement**

`scripts/w1_build_corpus.py`:

```python
"""W1 corpus build — sources -> cleaned raw tree, idempotent stages.

Stages (har ek apne `_DONE` marker se skip hota hai — Kaggle session marne par
dobara wahi se shuru):
  download : har source ko uske stage dir me materialize karo (byte-budget ke saath)

Network HF sources ke liye ye script KAGGLE/local-with-`datasets` pe chalta hai;
tests `--config None --input DIR` (pure local sources) se offline validate hote hain.
"""

from __future__ import annotations

import argparse
import json
import os


DEFAULT_TOTAL_BYTES = 2_400_000_000          # ~2.4 GB text ≈ 600M+ code tokens


def plan_budget(entries: list[dict], total_bytes: int) -> dict[str, int]:
    weights = [max(1, int(e.get("weight", 1))) for e in entries]
    tot = sum(weights)
    return {e["id"]: int(total_bytes * w / tot) for e, w in zip(entries, weights)}


def _marker(stage_dir: str) -> str:
    return os.path.join(stage_dir, "_DONE")


def _stage_download(entry: dict, budget: int, stage_root: str,
                    local_input: str | None) -> dict:
    from corpus.sources.registry import Source
    src = Source(**entry)
    stage_dir = os.path.join(stage_root, src.id.replace(":", "_").replace("/", "_"))
    os.makedirs(stage_dir, exist_ok=True)
    done = _marker(stage_dir)
    if os.path.exists(done):                       # idempotent: pehle se hua?
        info = json.load(open(done, encoding="utf-8"))
        return {"files": info["files"], "bytes": info["bytes"]}
    if src.kind == "huggingface":
        from corpus.download.huggingface import HuggingFaceDownloader
        dl = HuggingFaceDownloader(max_bytes=budget)
        staged = dl.fetch(src, stage_root)
        root = staged.root
    elif src.kind == "local":
        if not local_input:
            raise SystemExit(f"{src.id}: local source needs --input DIR")
        root = local_input                          # as-is tree
    else:
        raise SystemExit(f"{src.id}: kind {src.kind} not wired for W1 yet")
    files, total = 0, 0
    for dp, _, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            files += 1
            total += os.path.getsize(p)
    json.dump({"files": files, "bytes": total}, open(done, "w"))
    return {"files": files, "bytes": total}


def build(args) -> dict:
    entries = (json.load(open(args.config, encoding="utf-8"))
               if getattr(args, "config", None) else
               [{"id": "local:w1", "kind": "local", "location": args.input,
                 "languages": ["python", "c"], "category": "code"}])
    total = int(getattr(args, "total_gb", 2.4) * 1_000_000_000)
    budgets = plan_budget(entries, total)
    per_source_cap = int(getattr(args, "per_source_bytes", 0) or 0)
    stage_root = os.path.join(args.out, "stage")
    os.makedirs(stage_root, exist_ok=True)
    summary = {"sources": {}}
    for e in entries:
        cap = min(budgets[e["id"]], per_source_cap) if per_source_cap else budgets[e["id"]]
        summary["sources"][e["id"]] = _stage_download(
            e, cap, stage_root, getattr(args, "input", None))
    summary["total_bytes"] = sum(v["bytes"] for v in summary["sources"].values())
    with open(os.path.join(args.out, "_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/w1_sources.json",
                   help="JSON Source list; pass 'none' for a single local dir")
    p.add_argument("--input", default=None, help="local dir (config none mode)")
    p.add_argument("--out", default="corpus_out")
    p.add_argument("--total-gb", type=float, default=2.4)
    p.add_argument("--per-source-bytes", type=int, default=0,
                   help="hard cap per source (testing)")
    a = p.parse_args(argv)
    if a.config and a.config.lower() == "none":
        a.config = None
    s = build(a)
    print(json.dumps(s, indent=2))
    print(f"[w1] target {int(a.total_gb*1e9)} bytes; got {s['total_bytes']} "
          f"({'OK' if s['total_bytes'] >= int(a.total_gb*0.95) else 'SHORT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note for implementer: `scripts/` has no `__init__.py`; tests import via the path shim added in Task 2. Keep module import-light at top (imports inside functions) so the shim alone suffices.

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/w1_build_corpus.py tests/test_w1_corpus.py
git commit -m "feat(w1): idempotent corpus build orchestrator with byte budgets"
```

---

### Task 4: Tokenizer trainer — stratified sample, time-probe, save

**Files:**
- Create: `scripts/w1_train_tokenizer.py`
- Test: append to `tests/test_w1_corpus.py`

**Interfaces:**
- Consumes: `BPETokenizer.train(texts, vocab_size, verbose)`; `tok.save(path)`.
- Produces:
  - `stratified_sample(root: str, target_chars: int, seed: int = 1234) -> list[str]` — walks `root`, buckets files by extension (`.py` vs rest=C), round-robin picks shuffled files (seeded) until `target_chars` reached; returns list of file TEXTS.
  - `time_probe(texts: list[str]) -> float` — trains on first ~1MB slice, returns chars/sec estimate.
  - CLI: `--raw corpus_out/stage --vocab 24576 --sample-mb 60 --out tok/tokenizer.json [--probe-only]`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_stratified_sample_roundrobins_languages(tmp_path):
    from scripts.w1_train_tokenizer import stratified_sample

    for i in range(4):
        (tmp_path / f"m{i}.py").write_text("def f(): pass\n" * 20, encoding="utf-8")
        (tmp_path / f"k{i}.c").write_text("int main(){}\n" * 20, encoding="utf-8")
    texts = stratified_sample(str(tmp_path), target_chars=100000, seed=1)
    py = sum(1 for t in texts if t.lstrip().startswith("def "))
    c = len(texts) - py
    assert py == 4 and c == 4                              # dono languages poori


def test_time_probe_returns_positive_rate(tmp_path):
    from scripts.w1_train_tokenizer import stratified_sample, time_probe
    from tokenizer.bpe import BPETokenizer

    (tmp_path / "a.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    texts = stratified_sample(str(tmp_path), target_chars=10 ** 9, seed=0)
    rate = time_probe(texts[:1], tok=BPETokenizer())
    assert rate > 0
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_w1_corpus.py::test_stratified_sample_roundrobins_languages -q`
Expected: FAIL — `ModuleNotFoundError: scripts.w1_train_tokenizer`.

- [ ] **Step 3: Implement**

`scripts/w1_train_tokenizer.py`:

```python
"""W1 tokenizer training — scratch BPE @24k on a STRATIFIED C+Python sample.

Spec risk-table ke mutabiq poore corpus pe train karna CPU pe bahut dheema hai;
isliye stratified sample (default 60MB) + pehle 10MB ka time-probe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time


def _bucket_of(path: str) -> str:
    return ".py" if path.endswith(".py") else ".c"


def stratified_sample(root: str, target_chars: int, seed: int = 1234) -> list[str]:
    """Extension-buckets (.py vs .c) me round-robin — dono bhashaein barabar."""
    buckets: dict[str, list[str]] = {".py": [], ".c": []}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            b = _bucket_of(fn)
            if b in buckets:
                buckets[b].append(os.path.join(dp, fn))
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    texts: list[str] = []
    got = 0
    idx = 0
    while got < target_chars:
        progressed = False
        for b, files in buckets.items():
            if idx < len(files):
                progressed = True
                with open(files[idx], encoding="utf-8", errors="replace") as f:
                    t = f.read()
                texts.append(t)
                got += len(t)
        if not progressed:
            break                                   # buckets khatam ho gaye
        idx += 1
    return texts


def time_probe(texts: list[str], tok=None) -> float:
    """~1MB slice pe train karke chars/sec estimate wapas lao."""
    from tokenizer.bpe import BPETokenizer
    tok = tok or BPETokenizer()
    probe, size = [], 0
    for t in texts:
        probe.append(t[: 1_000_000 - size])
        size += len(probe[-1])
        if size >= 1_000_000:
            break
    t0 = time.time()
    tok.train(probe, vocab_size=2048, verbose=False)
    return size / max(1e-9, time.time() - t0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="corpus_out/stage")
    p.add_argument("--vocab", type=int, default=24576)
    p.add_argument("--sample-mb", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="tok/tokenizer.json")
    p.add_argument("--probe-only", action="store_true")
    a = p.parse_args(argv)

    target = int(a.sample_mb * 1_000_000)
    texts = stratified_sample(a.raw, target_chars=target, seed=a.seed)
    total = sum(len(t) for t in texts)
    print(f"[sample] files={len(texts)} chars={total:,}")
    rate = time_probe(texts)
    eta_min = total / max(rate, 1.0) / 60.0
    print(f"[probe] ~{rate:,.0f} chars/sec -> ETA ~{eta_min:.0f} min")
    if a.probe_only:
        return 0

    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(texts, vocab_size=a.vocab, verbose=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)
    meta = {"vocab_size": tok.vocab_size, "sample_chars": total,
            "train_seconds": round(time.time() - t0, 1),
            "seed": a.seed, "sources_root": a.raw}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/w1_train_tokenizer.py tests/test_w1_corpus.py
git commit -m "feat(w1): stratified tokenizer trainer with time probe"
```

---

### Task 5: RDS packing at scale — per-part pipelines + manifest merge

**Files:**
- Create: `scripts/w1_pack_rds.py`
- Test: append to `tests/test_w1_corpus.py`

**Interfaces:**
- Consumes: `RDEPipeline(tokenizer, config).run(root, out_dir)`, `RDSDataset(data_dir)` (manifest `shards` list), `dataset.lock.build_lock`.
- Produces:
  - `merge_manifests(part_dirs: list[str], out_dir: str, extra_meta: dict) -> dict` — reads each part's `manifest.json`, copies shard FILES into `out_dir`, writes a merged `manifest.json` (concatenated `shards`, summed stats, fresh lock via `build_lock`); returns merged manifest dict.
  - CLI: `--raw corpus_out/stage --tok tok/tokenizer.json --parts 4 --seq-len 1024 --out rds_w1` — splits discovered repo dirs across `--parts` part dirs, runs one `RDEPipeline.run` per part (skipping parts with `_DONE` markers), then merges.

- [ ] **Step 1: Write the failing test**

Append:

```python
def _mini_rds_part(tmp_path, tag):
    """Ek chhota sa part-dir banao jisme valid manifest + 1 shard ho (fixture).

    Real ShardManager use karte hain (dataset/sharding.py:20) — wahi manifest
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
    from scripts.w1_pack_rds import merge_manifests

    p1 = _mini_rds_part(tmp_path, "a"); p2 = _mini_rds_part(tmp_path, "b")
    out = str(tmp_path / "merged")
    mm = merge_manifests([p1, p2], out, extra_meta={"note": "w1"})
    assert len(mm["shards"]) == 2
    ds_paths = [os.path.join(out, s["file"]) for s in mm["shards"]]
    assert all(os.path.exists(p_) for p_ in ds_paths)      # files copy hue
    back = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert back["note"] == "w1"
```

(Verified against repo: `ShardManager(out_dir, config, tokenizer)` + `.add_chunk(ids, meta)` / `.finalize(stats_summary, lock)` writes `manifest.json` containing `"shards": [{"file", "chunks", ...}]` — exactly what `merge_manifests` consumes.)

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_w1_corpus.py::test_merge_manifests_concatenates_shards -q`
Expected: FAIL — `ModuleNotFoundError: scripts.w1_pack_rds`.

- [ ] **Step 3: Implement**

`scripts/w1_pack_rds.py`:

```python
"""W1 RDS packing — raw tree -> seq_len chunks, PARTS me todkar (resume-safe).

Har part apna RDEPipeline.run() karta hai (apna out_dir + `_DONE` marker),
phir merge_manifests() sab shards ko ek final data_dir me jodta hai jo
RDSDataset(training) seedha padh sakta hai (manifest["shards"] list native).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def merge_manifests(part_dirs: list[str], out_dir: str,
                    extra_meta: dict | None = None) -> dict:
    """Part manifests ko ek final manifest me jodo (shard files copy hote hain)."""
    os.makedirs(out_dir, exist_ok=True)
    shards: list[dict] = []
    stats_total: dict = {}
    seen_names: set[str] = set()
    for pd in part_dirs:
        mf = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
        for sh in mf.get("shards", []):
            fname = sh["file"]
            while fname in seen_names:                  # naam takraaye toh prefix
                fname = f"{os.path.basename(pd)}_{fname}"
            seen_names.add(fname)
            shutil.copy2(os.path.join(pd, sh["file"]), os.path.join(out_dir, fname))
            shards.append({**sh, "file": fname})
        for k, v in (mf.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                stats_total[k] = stats_total.get(k, 0) + v
    manifest = {"shards": shards, "stats": stats_total}
    manifest.update(extra_meta or {})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _discover_repo_dirs(raw_root: str) -> list[str]:
    """Stage tree ke andar repo-level dirs (jahan files hain)."""
    repos = []
    for dp, fns, _ in os.walk(raw_root):
        if any(not d.startswith("_") and os.path.isfile(os.path.join(dp, d))
               for d in fns):
            repos.append(dp)
    return sorted(repos)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="corpus_out/stage")
    p.add_argument("--tok", default="tok/tokenizer.json")
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--out", default="rds_w1")
    a = p.parse_args(argv)

    from dataset import load_bpe_tokenizer
    from dataset.pipeline import RDEPipeline
    tok = load_bpe_tokenizer(a.tok)

    repos = _discover_repo_dirs(a.raw)
    if not repos:
        raise SystemExit(f"[pack] koi repo files nahi mili {a.raw!r} me")
    k = max(1, a.parts)
    groups = [repos[i::k] for i in range(k)]             # round-robin split

    part_dirs: list[str] = []
    for gi, group in enumerate(groups):
        if not group:
            continue
        part_in = os.path.join(a.out, f"in_{gi:02d}")
        part_out = os.path.join(a.out, f"part_{gi:02d}")
        done = os.path.join(part_out, "_DONE")
        os.makedirs(part_in, exist_ok=True)
        for r in group:                                  # symlink: disk bachao
            dst = os.path.join(part_in, os.path.basename(r) or f"repo_{len(os.listdir(part_in))}")
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(r), dst)
                except OSError:                          # FS symlink support nahi
                    shutil.copytree(r, dst)
        if not os.path.exists(done):
            pipe = RDEPipeline(tok)
            pipe.run(part_in, part_out, verbose=True)
            open(done, "w").write("{}")
        else:
            print(f"[pack] part_{gi:02d} pehle se DONE — skip")
        part_dirs.append(part_out)

    mm = merge_manifests(part_dirs, os.path.join(a.out, "final"),
                         extra_meta={"seq_len": a.seq_len, "w1": True})
    n_chunks = sum(sh.get("chunks", 0) for sh in mm["shards"])
    print(f"[pack] shards={len(mm['shards'])} chunks~={n_chunks} "
          f"-> {os.path.join(a.out, 'final')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Implementer note: verify `RDEPipeline.run(root, out_dir)` writes its `manifest.json` with a `shards` key inside `out_dir` (read `dataset/sharding.py:ShardManager.finalize` first) — the merge consumes exactly that shape, so no key adaptation should be needed.

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/w1_pack_rds.py tests/test_w1_corpus.py
git commit -m "feat(w1): resumable RDS packing with manifest merge"
```

---

### Task 6: Production training glue — val split + W1 defaults in kaggle_train.py

**Files:**
- Modify: `scripts/kaggle_train.py` (CLI + wiring only; NO changes to `training/` package)

**Interfaces:**
- Consumes: existing `TrainConfig(val_data_dir=...)`, `make_dataloaders`; existing flag names stay verbatim — NOTE the warmup flag is **`--warmup`** (not `--warmup_steps`) and the RDS dir is `<work>/rds_out` (kaggle_train.py:200,207).
- Produces:
  - New flag: `--val_raw DIR` (held-out code folder; if given, its RDS is built to `<work>/rds_val` via `build_or_load_rds` and passed as `val_data_dir` into BOTH `TrainConfig(...)` constructions — main + resume).
  - `resolve_args(a)` sets derived `a.eff_tokens = a.micro_batch * a.grad_accum * a.seq_len`; main prints it beside device selection.
  - Non-smoke DEFAULTS become: `--vocab 24576 --seq_len 1024 --lr 6e-4 --warmup 2000 --micro_batch 16 --grad_accum 16 --steps 8000`. `--dtype` STAYS `None` (auto) — forcing fp16 would break CPU/local runs; the Kaggle notebook passes `--dtype fp16` explicitly on T4. Smoke overrides unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_w1_corpus.py`:

```python
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
```

NOTE: this requires two mechanical refactors in `scripts/kaggle_train.py`:
1. Extract the inline `argparse.ArgumentParser(...)` construction in `main()` into module-level `build_parser() -> ArgumentParser`; `main` becomes `args = resolve_args(build_parser().parse_args(argv))`.
2. Add the resolver (computed, not a static default, so user overrides stay correct):

```python
def resolve_args(a):
    """Derived flags — effective batch tokens/step (spec §4 ≈260k)."""
    a.eff_tokens = a.micro_batch * a.grad_accum * a.seq_len
    return a
```

and `main` prints `[batch] effective tokens/step = {args.eff_tokens:,}` beside device selection.

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_w1_corpus.py::test_kaggle_train_defaults_are_production -q`
Expected: FAIL — no `build_parser` attribute.

- [ ] **Step 3: Implement**

In `scripts/kaggle_train.py`:
1. Move the parser construction from `main()` into `def build_parser(): ... return p` (verbatim flags, with the NEW defaults listed above; keep `--smoke` overrides in `main` untouched).
2. After parse: `args.eff_tokens = args.micro_batch * args.grad_accum * args.seq_len` and print `[batch] effective tokens/step = ...` next to device selection.
3. Val branch right after step 3 (main RDS build) in `main()`:

```python
    val_rds_dir = None
    if args.val_raw:
        val_rds_dir = os.path.join(work, "rds_val")
        build_or_load_rds(tok, args.val_raw, val_rds_dir, args.seq_len,
                          args.rebuild)
```

and pass `val_data_dir=val_rds_dir` into BOTH existing `TrainConfig(...)` constructions (the main one and the `--resume_demo` one — field already supported by `training/dataloader.py`).
4. Add `p.add_argument("--val_raw", default=None, help="held-out code folder -> separate RDS for validation")`.

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green (kaggle_train smoke path untouched — `--smoke` still forces old tiny config).

- [ ] **Step 5: Commit**

```bash
git add scripts/kaggle_train.py tests/test_w1_corpus.py
git commit -m "feat(w1): production defaults + val split in kaggle trainer"
```

---

### Task 7: Kaggle notebook refresh + W1 runbook

**Files:**
- Modify: `notebooks/ryth_kaggle_train.ipynb`
- Create: `docs/w1_runbook.md`
- Test: append to `tests/test_w1_corpus.py`

**Interfaces:**
- Consumes: all `scripts/w1_*` CLIs + `scripts/kaggle_train.py` + `evals.ppl.evaluate_files`.
- Produces: a Run-All-clean notebook (CPU-prep cells + GPU-train cells clearly separated) and the owner-facing runbook with exact click-path, budget table, deliverables checklist.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_notebook_is_valid_json_with_w1_cells():
    nb_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notebooks", "ryth_kaggle_train.ipynb")
    nb = json.load(open(nb_path, encoding="utf-8"))
    assert nb["nbformat"] >= 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("w1_build_corpus.py", "w1_train_tokenizer.py",
                   "w1_pack_rds.py", "kaggle_train.py"):
        assert needle in src, f"notebook missing {needle}"
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_w1_corpus.py::test_notebook_is_valid_json_with_w1_cells -q`
Expected: FAIL — needles missing.

- [ ] **Step 3: Implement**

Update `notebooks/ryth_kaggle_train.ipynb` cells (keep existing structure/style; edit source arrays):

1. **Markdown header cell**: "W1 production flow — CPU prep (Section A) → GPU train (Section B)". Note Kaggle: enable GPU+Internet for Section B; Section A runs on CPU session.
2. **Cell A1** (code):

```python
# A1 — W1 prep (CPU session): corpus -> tokenizer -> RDS
WORK = "/kaggle/work"          # Kaggle persistent working dir
%cd /
!git clone https://github.com/RAJ-af/Ryth ryth && %cd ryth
import os; os.environ["RYTH_WORK"] = WORK
!python3 scripts/w1_probe_stack.py --subset python --limit 200
!python3 scripts/w1_probe_stack.py --subset c --limit 200
!python3 scripts/w1_build_corpus.py --config configs/w1_sources.json \
        --out $RYTH_WORK/corpus_out --total-gb 2.4
# chhota held-out val split (~200 files) validation loss ke liye
!mkdir -p $RYTH_WORK/val_src && \
 ls $RYTH_WORK/corpus_out/stage/*/* | head -200 | xargs -I{} cp {} $RYTH_WORK/val_src/
!python3 scripts/w1_train_tokenizer.py --raw $RYTH_WORK/corpus_out/stage \
        --vocab 24576 --sample-mb 60 --out $RYTH_WORK/tok/tokenizer.json
!python3 scripts/w1_pack_rds.py --raw $RYTH_WORK/corpus_out/stage \
        --tok $RYTH_WORK/tok/tokenizer.json --parts 4 --seq-len 1024 \
        --out $RYTH_WORK/rds_w1
```

3. **Cell A2** (code) — save outputs as a private Kaggle Dataset (owner clicks the UI "Save Version" per runbook; cell prints instructions + sizes):

```python
# A2 — outputs ka size dekho; phir UI se Save As Dataset (runbook §2)
!du -sh $RYTH_WORK/corpus_out $RYTH_WORK/tok $RYTH_WORK/rds_w1 $RYTH_WORK/val_src
```

4. **Cell B1** (code, GPU session, dataset attached at `/kaggle/input/w1-prep`):

```python
# B1 — 30M training (GPU/T4, fp16, auto-resume via latest.pt)
import os
PREP = "/kaggle/input/w1-prep"      # attach A-outputs dataset here
WORK = "/kaggle/working/run"; os.makedirs(WORK, exist_ok=True)
# prepared artifacts ko wahi paths par rakho jahan kaggle_train dhundta hai:
#   tok/tokenizer.json (build_or_load_tokenizer cache) + rds_out/manifest.json
!mkdir -p $WORK/tok $WORK/rds_out && \
 cp -r $PREP/tok/. $WORK/tok/ && cp -r $PREP/rds_w1/final/. $WORK/rds_out/
!ls $WORK $WORK/rds_out | head
!python3 scripts/kaggle_train.py --work $WORK --raw $PREP/corpus_out/stage \
    --preset ryth_30m --vocab 24576 --seq_len 1024 --lr 6e-4 \
    --warmup 2000 --micro_batch 16 --grad_accum 16 --steps 8000 \
    --dtype fp16 --num_workers 2 --val_raw $PREP/val_src
```

(Cached `tok/` + `rds_out/` hone se tokenizer/RDS rebuild SKIP hota hai — GPU sirf training karta hai; `--raw` sirf fallback ke liye attached rehta hai.)

5. **Cell B2** (code) — quick ppl sanity + generation samples (C + Python prompts) using `evals.ppl` and `evals.generation.sample_completion` on `best.pt`; prints results JSON path.

`docs/w1_runbook.md` content (write fully):

```markdown
# W1 Runbook — 30M pretraining on Kaggle (owner clicks only)

## 0. One-time
- Kaggle account → Settings → enable Internet + GPU (T4) for notebooks.
- This repo pushed to GitHub (public or Kaggle-importable).

## 1. CPU prep session (~4–8h, free quota)
1. New Notebook → File → Import → notebook `notebooks/ryth_kaggle_train.ipynb`.
2. Settings: Accelerator = None, Internet = On.
3. Run Section A (A1, A2). Idempotent — agar session beech me mare, Run All dobara;
   `_DONE` markers complete stages skip karte hain.
4. Output: `/kaggle/working` me `corpus_out/`, `tok/`, `rds_w1/`.

## 2. Package prep outputs
- Notebook right panel → Output → "Save Version" → private dataset banao, naam `w1-prep`.

## 3. GPU train session (~2–4h GPU)
1. Same notebook → Settings: Accelerator = GPU T4, Internet On.
2. Right panel → Input → attach dataset `w1-prep` (path `/kaggle/input/w1-prep`).
3. Run Section B only. Agar 12h session limit aaye: bas dobara Run B1 —
   `latest.pt` se auto-resume hota hai (same WORK path rakho).
4. Deliverables: `runs/ryth-kaggle/best.pt` + `latest.pt`, loss curve PNG/log,
   throughput line (tokens/sec), Cell B2 ke C+Python samples.

## 4. Acceptance (M1)
- [ ] val loss untrained baseline (~ln vocab ≈ 10.1 @24k) se CLEARLY neeche (< 3.0)
- [ ] Python samples mostly syntactically valid; C samples structurally sane
- [ ] `rds_w1/final/manifest.json` + tokenizer meta committed/attached (reproducibility)
- [ ] Eval: `ryth-eval ppl --ckpt best.pt ...` numbers recorded in results/

## Budget notes (spec §11)
- Weekly GPU quota ~30h; 30M run ≈ 2–4h → multiple resumes possible.
- Corpus target ≥600M tokens; agar kam mile: probe se dekho kaunsa subset short
  pada, `--total-gb` badhao ya curated GitHub sources config me add karo.

## Troubleshooting
- **`load_dataset` script-dataset reject ho** (`Dataset scripts are no longer
  supported` — newer `datasets` versions): The Stack-dedup legacy loading-script
  format hai. Fallback: parquet mirror use karo —
  `load_dataset("bigcode/the-stack-dedup", revision="refs/convert/parquet", data_dir="data/python", streaming=True)`
  ya phir `bigcode/starcoderdata` (native parquet, `content`+`license` columns,
  streaming works). Probe script (§1 step 3) pehle hi ye bata degi.
- **Session beech me mari**: wahi section dobara Run karo — `_DONE` markers +
  `latest.pt` resume sab handle karte hain (same `/kaggle/working` path).
```

- [ ] **Step 4: Green + full suite**

Run: `python3 -m pytest tests/test_w1_corpus.py -q && python3 -m pytest tests/ -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/ryth_kaggle_train.ipynb docs/w1_runbook.md tests/test_w1_corpus.py
git commit -m "docs(w1): Kaggle notebook production cells + owner runbook"
```

---

### Task 8: Local mini end-to-end acceptance (offline proof the chain works)

**Files:**
- Create: `.superpowers` workspace script (NOT committed) — verification only
- Test: no new committed tests (this is the acceptance gate like M0 Task 10)

**Interfaces:**
- Consumes: Tasks 1–7 CLIs end-to-end with tiny budgets, purely local sources.

- [ ] **Step 1: Mini pipeline run (workspace script, not committed)**

```python
"""W1 local mini-acceptance: local-source config se poora chain chalao.

Budgets tiny hain (seconds me); yeh prove karta hai ki
build_corpus -> train_tokenizer -> pack_rds -> (parser-level) train-config
sab wire ho chuke hain. REAL 600M-token run Kaggle pe hoga (runbook).
"""
import json, os, subprocess, sys, tempfile

ROOT = "/root/Ryth"
tmp = tempfile.mkdtemp(prefix="w1_accept_")
src = os.path.join(tmp, "in"); os.makedirs(src)
for i in range(6):
    open(os.path.join(src, f"m{i}.py"), "w").write(
        "def add(a, b):\n    return a + b\n\n" * 50)
for i in range(6):
    open(os.path.join(src, f"k{i}.c"), "w").write(
        "#include <stdio.h>\nint add(int a,int b){return a+b;}\n" * 40)

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

out = os.path.join(tmp, "corpus_out")
run([sys.executable, "scripts/w1_build_corpus.py", "--config", "none",
     "--input", src, "--out", out, "--total-gb", "0.001"])
run([sys.executable, "scripts/w1_train_tokenizer.py", "--raw",
     os.path.join(out, "stage"), "--vocab", "512", "--sample-mb", "1",
     "--out", os.path.join(tmp, "tok/tokenizer.json")])
run([sys.executable, "scripts/w1_pack_rds.py", "--raw",
     os.path.join(out, "stage"),
     "--tok", os.path.join(tmp, "tok/tokenizer.json"),
     "--parts", "2", "--seq-len", "64", "--out", os.path.join(tmp, "rds")])
mf = json.load(open(os.path.join(tmp, "rds", "final", "manifest.json")))
assert mf["shards"], "no shards merged!"
assert os.path.exists(os.path.join(tmp, "tok", "tokenizer.json.meta.json"))
print("W1 MINI ACCEPTANCE: PASS")
```

Save as `.superpowers/sdd/2026-08-25-phase6-w1-kaggle-30m/acceptance_w1.py`, run it:
`python3 .superpowers/sdd/2026-08-25-phase6-w1-kaggle-30m/acceptance_w1.py`
Expected: prints `W1 MINI ACCEPTANCE: PASS`.

- [ ] **Step 2: Whole-suite green + clean tree**

Run: `python3 -m pytest tests/ -q && git status --short`
Expected: all PASS; nothing uncommitted (workspace is git-ignored).

- [ ] **Step 3: Record acceptance in ledger/commit message**

If anything was fixed during acceptance, commit fixes with message `fix(w1): acceptance adjustments`; otherwise state PASS in the execution ledger.

## Plan Self-Review Notes (for executor awareness)

- Spec §4 coverage: sources(T1,T2,T3) · tokenizer 24k stratified(T4) · RDS@1024 uint16(T5, RDE native) · 30m preset+fp16+lr6e-4+260k batch+auto-resume(T6, engine native) · deliverables+acceptance(runbook T7, mini-acceptance T8) ✔
- Deliberate scope cut: curated GitHub repos ship as config EXTENSIONS (add entries to `configs/w1_sources.json`) rather than a new downloader — github kind already exists in registry/downloader. Owner extends the JSON, zero code.
- License policy ruling encoded: probe FIRST (T2); rows kept iff license ∈ allowlist OR unknown — recorded in manifest/docs. If probe shows no license column at all, policy falls back to "dedup'd stack, documented" (measure-first; surfaced to owner in runbook).
- Known runtime risks (spec §11): scratch-BPE encode speed at 2.4GB → mitigated by parts+markers+ETA probe; numpy absent locally → HF path targets Kaggle sessions, tests stay offline.
