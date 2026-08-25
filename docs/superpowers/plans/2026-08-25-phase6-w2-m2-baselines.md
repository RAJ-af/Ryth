# Ryth Phase 6 / W2 — M2 Eval Baselines Recorded Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record M2 acceptance proof — real-benchmark eval baselines
(HumanEval/MBPP pass@k + held-out ppl) written as committed `results/*.json`,
with repeatable tooling so post-training runs drop into the same format.

**Architecture:** The `evals/` package itself already exists (M0 done — all 5
spec §5 units + random-weight acceptance). What remains is M2 (§10): *baseline
scores recorded, results JSONs in repo*. This plan adds: a `--limit` flag for
cheap CPU baseline sweeps, an aggregating `ryth-eval report` subcommand,
committed real benchmark files under `bench/` (MIT / CC-BY-4.0, small), a
deterministic baseline runner script, and the recorded-numbers doc.

**Tech Stack:** stdlib + existing evals package; network used ONCE to fetch
benchmark files (then committed — future runs offline).

**Spec:** `docs/superpowers/specs/2026-08-25-phase6-measure-first-design.md`
(§5 W2 units, §10 M2 row)

## Global Constraints

- No core-pillar changes (`tokenizer/`, `dataset/`, `model/`, `training/` untouched); `evals/`, `scripts/`, `docs/`, new `bench/` are fair game.
- Every task writes a JSON result file into `results/` (spec §5 design rule).
- Tests stay OFFLINE — network downloads never run under pytest.
- Baseline settings are deliberately small (random weights ⇒ score is 0
  regardless); full-settings rerun commands are documented for post-training.
- Determinism: seeded sampling, sorted iteration, stable file order.
- Security note stays explicit: pass@k executes generated code locally (docs/evals.md §Security).
- Deliverable: `results/w2_*.json` committed + `docs/w2_baselines.md` with
  numbers, provenance, licenses, and post-training rerun commands.

## Verified repo facts this plan builds on

- `evals/cli.py:main(argv)` — inline argparse, subcommands humaneval|mbpp|ppl,
  shared `common(sp)` flags; result shapes:
  pass@k → `{"meta": {...}, "pass_at_k": {"pass@1": f}, ...}`;
  ppl → `{"meta": {"task": "ppl"}, "perplexity": {label: f}}`.
- `evals/datasets.py` — `download_humaneval(dest_dir)` (44 KB gz),
  `download_mbpp(dest_dir)`; `load_problems(path)` jsonl reader;
  `mbpp.load_mbpp(path)` tolerant adapter.
- `evals.humaneval.evaluate(problems, *, sampler=None, model=None, tok=None,
  n_samples=20, ..., ks=(1,5,10), progress=print)`; `save_results(res, path)`.
- `evals.ppl.evaluate_files(model, tok, {label: path}, **kw) -> {label: ppl}`.
- M0 acceptance recipe: near-byte tokenizer = `BPETokenizer().train(["hello
  world"], vocab_size=260)`; `RythConfig(...)` constructed DIRECTLY when
  overriding dims (preset classmethods reject dim overrides).
- Network reachable (HumanEval URL probed: HTTP 200).

---

### Task 1: `--limit` flag for humaneval/mbpp (cheap baseline sweeps)

**Files:**
- Modify: `evals/cli.py`
- Test: append to `tests/test_evals.py`

**Interfaces:**
- Produces: `apply_limit(problems: list, limit: int | None) -> list` in
  `evals/cli.py`; CLI flag `--limit INT` (default None = no slice) on both
  humaneval and mbpp subparsers.

- [ ] **Step 1: Failing tests**

Append to `tests/test_evals.py`:

```python
def test_apply_limit_slices_and_defaults():
    from evals.cli import apply_limit

    probs = list("abcde")
    assert apply_limit(probs, None) == probs          # default: no slice
    assert apply_limit(probs, 0) == probs             # 0/negative => no slice
    assert apply_limit(probs, 3) == list("abc")
    assert apply_limit([], 5) == []


def test_cli_has_limit_flag():
    import io
    import contextlib
    from evals.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), pytest.raises(SystemExit):
        main(["humaneval", "--help"])
    assert "--limit" in buf.getvalue()
```

(If `pytest` isn't imported at top of test_evals.py yet, add it.)

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_evals.py::test_apply_limit_slices_and_defaults tests/test_evals.py::test_cli_has_limit_flag -q`
Expected: FAIL — ImportError (no apply_limit) / missing --limit.

- [ ] **Step 3: Implement**

In `evals/cli.py` add module-level function + flag + wiring:

```python
def apply_limit(problems: list, limit: int | None) -> list:
    """Baseline sweeps ke liye pehli N problems; None/<=0 => poori list."""
    if not limit or limit <= 0:
        return problems
    return problems[:limit]
```

Inside the `for name in ("humaneval", "mbpp"):` loop add:
`sp.add_argument("--limit", type=int, default=None,
help="pehli N problems (baseline/CPU sweeps)")`

In the else-branch right after `problems = (...)`:
`problems = apply_limit(problems, getattr(args, "limit", None))`

- [ ] **Step 4: Green + suite**

Run: `python3 -m pytest tests/test_evals.py -q && python3 -m pytest tests/ -q`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add evals/cli.py tests/test_evals.py
git commit -m "feat(evals): --limit flag for cheap baseline problem sweeps"
```

---

### Task 2: `ryth-eval report` — aggregate results dir into one table

**Files:**
- Modify: `evals/cli.py`
- Modify: `docs/evals.md` (one line in Quickstart)
- Test: append to `tests/test_evals.py`

**Interfaces:**
- Produces: `key_metrics(data: dict) -> dict` — extracts comparable metrics
  from any results JSON: flattens `pass_at_k` items verbatim, `perplexity`
  items verbatim, plus any other top-level numeric fields; returns `{}` when
  nothing comparable. Subcommand: `ryth-eval report DIR [--out FILE]` prints a
  markdown table (`file | metrics…`); with `--out`, writes it instead.

- [ ] **Step 1: Failing tests**

Append:

```python
def test_key_metrics_extracts_shapes(tmp_path):
    from evals.cli import key_metrics

    he = {"meta": {"task": "humaneval"}, "pass_at_k": {"pass@1": 0.0},
          "n_problems": 3}
    assert key_metrics(he) == {"pass@1": 0.0, "n_problems": 3}
    ppl = {"meta": {"task": "ppl"}, "perplexity": {"python": 10.5}}
    assert key_metrics(ppl) == {"python": 10.5}
    assert key_metrics({"meta": {}}) == {}            # kuch comparable nahi


def test_report_subcommand_writes_markdown(tmp_path, capsys):
    import json as _json
    from evals.cli import main

    d = tmp_path / "results"; d.mkdir()
    _json.dump({"pass_at_k": {"pass@1": 0.0}},
               open(d / "w2_humaneval_baseline.json", "w"))
    rc = main(["report", str(d)])
    assert rc == 0
    tbl = capsys.readouterr().out
    assert "w2_humaneval_baseline.json" in tbl and "pass@1" in tbl
    out_file = d / "table.md"
    rc2 = main(["report", str(d), "--out", str(out_file)])
    assert rc2 == 0
    assert "pass@1" in out_file.read_text(encoding="utf-8")
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_evals.py::test_key_metrics_extracts_shapes tests/test_evals.py::test_report_subcommand_writes_markdown -q`
Expected: FAIL — ImportError / invalid choice 'report'.

- [ ] **Step 3: Implement**

In `evals/cli.py`:

```python
def key_metrics(data: dict) -> dict:
    """Results-JSON se comparable numbers nikalo (kisi bhi task ka shape)."""
    out: dict = {}
    pak = data.get("pass_at_k")
    if isinstance(pak, dict):
        out.update({k: v for k, v in pak.items() if isinstance(v, (int, float))})
    ppl = data.get("perplexity")
    if isinstance(ppl, dict):
        out.update({k: v for k, v in ppl.items() if isinstance(v, (int, float))})
    for k, v in data.items():
        if k in ("meta", "pass_at_k", "perplexity"):
            continue
        if isinstance(v, (int, float)):
            out[k] = v
    return out


def cmd_report(results_dir: str, out_path: str | None = None) -> int:
    """results/*.json ko ek markdown comparison table me jodo."""
    import glob
    import json
    import os

    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                m = key_metrics(json.load(f))
        except (OSError, ValueError):
            m = {}
        name = os.path.basename(p)
        cells = " | ".join(f"{k}: {v:.4f}" for k, v in m.items()) or "—"
        rows.append(f"| {name} | {cells} |")
    lines = ["| file | metrics |", "|---|---|"] + rows
    table = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table)
    else:
        print(table, end="")
    return 0
```

In `main()` before parsing: `sub.add_parser("report")` with its own flags
(`results_dir` positional, `--out`), early-return branch:

```python
    if args.task == "report":
        return cmd_report(args.results_dir, args.out)
```

NOTE: `report` must NOT hit `common()` — no ckpt/tokenizer needed; place the
branch immediately after `args = ap.parse_args(argv)`.

- [ ] **Step 4: Green + suite**

Run: `python3 -m pytest tests/test_evals.py -q && python3 -m pytest tests/ -q`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add evals/cli.py tests/test_evals.py docs/evals.md
git commit -m "feat(evals): ryth-eval report aggregates results dir into table"
```

---

### Task 3: Real benchmarks committed under `bench/` (+ counts test)

**Files:**
- Create: `bench/humaneval.jsonl.gz`, `bench/mbpp.jsonl` (fetched once, committed)
- Create: `bench/README.md` (provenance + licenses)
- Test: append to `tests/test_evals.py`

**Interfaces:**
- Consumes: `evals.datasets.download_humaneval/download_mbpp` (network, manual step).
- Produces: stable offline benchmark paths every later run uses:
  `bench/humaneval.jsonl.gz` (164 problems), `bench/mbpp.jsonl` (~974 rows).

- [ ] **Step 1: Fetch (manual, NOT in tests)**

```bash
python3 -c "from evals.datasets import download_humaneval, download_mbpp; download_humaneval('bench'); download_mbpp('bench')"
python3 -c "from evals.datasets import load_problems; from evals import mbpp; print(len(load_problems('bench/humaneval.jsonl.gz')), len(mbpp.load_mbpp('bench/mbpp.jsonl')))"
```

Expected: `164 974` (ya close — MBPP adapter kuch rows chhod sakta hai).

- [ ] **Step 2: Write `bench/README.md`**

```markdown
# bench/ — real benchmark files (committed for reproducible offline runs)

| File | Source | License |
|---|---|---|
| `humaneval.jsonl.gz` | openai/human-eval (data/HumanEval.jsonl.gz) | MIT |
| `mbpp.jsonl` | google-research/google-research-datasets mbpp | CC-BY-4.0 |

Fetch ke liye: `evals.datasets.download_humaneval/download_mbpp` (network opt-in).
Tests sirf counts verify karte hain — kabhi download nahi karte.

NOTE (M2): `bench/val_python.txt` provisional held-out set hai (HumanEval
prompts+canonical solutions se bana). Post-W1 real corpus val split aayega
(w1-prep `val_src/`) — phir wahi use hoga, ye rakhna sirf history ke liye.
```

- [ ] **Step 3: Counts test (offline — files ab committed hain)**

Append to `tests/test_evals.py`:

```python
def test_bench_files_present_and_parse():
    import os
    from evals import mbpp
    from evals.datasets import load_problems

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    he = load_problems(os.path.join(root, "bench", "humaneval.jsonl.gz"))
    mp = mbpp.load_mbpp(os.path.join(root, "bench", "mbpp.jsonl"))
    assert len(he) == 164
    assert len(mp) >= 800
    assert all(p.prompt and p.test for p in he[:5])
```

- [ ] **Step 4: Green + suite**

Run: `python3 -m pytest tests/test_evals.py::test_bench_files_present_and_parse -q && python3 -m pytest tests/ -q`
Expected: PASS (164 parsed), suite green.

- [ ] **Step 5: Commit**

```bash
git add bench/
git commit -m "feat(w2): real HumanEval+MBPP files committed under bench/"
```

---

### Task 4: Baseline runner — deterministic one-command sweep

**Files:**
- Create: `scripts/w2_baselines.py`
- Test: append to `tests/test_evals.py`

**Interfaces:**
- Consumes: Tasks 1–3 outputs (`--limit`, `bench/*`, near-byte tokenizer recipe).
- Produces:
  - `build_val_python(bench_dir: str, out_path: str) -> int` — deterministic
    concat of HumanEval `prompt + canonical_solution` texts (sorted by task_id)
    separated by blank lines; returns chars written; idempotent bytes.
  - `run_all(results_dir: str, bench_dir: str, limit: int = 40) -> dict` —
    trains near-byte BPE(260), builds random-weight ryth_30m (seq_len 256),
    runs humaneval+mbpp (n_samples=1, ks=(1,), max_new_tokens=32, seed fixed)
    + ppl on val_python, saves `results/w2_{humaneval,mbpp,ppl}_baseline.json`;
    returns summary dict.
  - CLI: `python3 scripts/w2_baselines.py [--results results] [--bench bench]
    [--limit 40]`.

- [ ] **Step 1: Failing tests**

Append:

```python
def test_build_val_python_deterministic(tmp_path):
    import subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out1 = tmp_path / "v1.txt"; out2 = tmp_path / "v2.txt"
    for o in (out1, out2):
        subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, %r); "
                        "from scripts.w2_baselines import build_val_python; "
                        "print(build_val_python(%r, %r))"
                        % (os.path.join(root, "scripts"),
                           os.path.join(root, "bench"), str(o))],
                       check=True, capture_output=True)
    b1 = out1.read_bytes(); b2 = out2.read_bytes()
    assert b1 == b2 and len(b1) > 1000


def test_run_all_smoke_tiny(tmp_path):
    # bahut chhota limit — sirf wiring prove (random weights, score 0 hi hoga)
    from w2_baselines import run_all

    s = run_all(str(tmp_path / "results"), limit=2, max_new_tokens=4,
                bench_dir=os.path.join(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))), "bench"))
    assert s["humaneval"]["pass_at_k"]["pass@1"] == 0.0
    assert s["mbpp"]["pass_at_k"]["pass@1"] == 0.0
    assert s["ppl"]["python"] > 1.0
    assert (tmp_path / "results" / "w2_ppl_baseline.json").exists()
```

(`sys.path` shim already inserts `<repo>/scripts` — see top of test_w1_corpus.py;
for test_evals.py add the same two-line shim above these tests OR put imports
inside functions with a local insert. Use the shim approach matching
test_w1_corpus.py.)

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_evals.py::test_run_all_smoke_tiny -q`
Expected: FAIL — ModuleNotFoundError w2_baselines.

- [ ] **Step 3: Implement**

`scripts/w2_baselines.py`:

```python
"""W2/M2 baseline sweep — random-weight 30M par real benchmarks, ek command.

Random weights => pass@k 0 aur ppl ~ln(vocab) hota hai HI — point ye hai ki
poora harness REAL files par end-to-end chale aur committed results JSON
banaye jiske against trained checkpoints compare honge (spec §10 M2).

Budgets jaan-boojh ke chhote hain (CPU minutes, ghanton nahi). Post-training
full settings docs/w2_baselines.md me documented hain.
"""

from __future__ import annotations

import argparse
import json
import os


def build_val_python(bench_dir: str, out_path: str) -> int:
    """Provisional held-out Python text: HumanEval prompt+solution, sorted."""
    from evals.datasets import load_problems

    probs = sorted(load_problems(os.path.join(bench_dir, "humaneval.jsonl.gz")),
                   key=lambda p: p.task_id)
    parts = [f"{p.prompt}\n{p.canonical_solution}\n"
             for p in probs if p.canonical_solution]
    text = "\n\n".join(parts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text)


def _near_byte_tokenizer():
    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=260, verbose=False)   # M0 recipe
    return tok


def _random_model(vocab: int, seq_len: int):
    from model import RythConfig, RythForCausalLM

    cfg = RythConfig(vocab_size=vocab)
    cfg.max_seq_len = seq_len
    return RythForCausalLM(cfg)


def run_all(results_dir: str, limit: int = 40, max_new_tokens: int = 32,
            bench_dir: str = "bench") -> dict:
    from evals import mbpp
    from evals.datasets import load_problems
    from evals.humaneval import evaluate as he_eval
    from evals.mbpp import evaluate as mp_eval
    from evals.ppl import evaluate_files

    os.makedirs(results_dir, exist_ok=True)
    tok = _near_byte_tokenizer()
    model = _random_model(tok.vocab_size, seq_len=256)
    quiet = lambda *a, **k: None                      # noqa: E731

    val_py = os.path.join(results_dir, "val_python.txt")
    n_chars = build_val_python(bench_dir, val_py)
    print(f"[val] python held-out chars={n_chars:,}")

    he = he_eval(load_problems(os.path.join(bench_dir, "humaneval.jsonl.gz"))[:limit],
                 model=model, tok=tok, n_samples=1, ks=(1,),
                 max_new_tokens=max_new_tokens, progress=quiet)
    with open(os.path.join(results_dir, "w2_humaneval_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump(he, f, indent=2)

    mp = mp_eval(mbpp.load_mbpp(os.path.join(bench_dir, "mbpp.jsonl"))[:limit],
                 model=model, tok=tok, n_samples=1, ks=(1,),
                 max_new_tokens=max_new_tokens, progress=quiet)
    with open(os.path.join(results_dir, "w2_mbpp_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2)

    ppl = evaluate_files(model, tok, {"python": val_py}, seq_len=256)
    with open(os.path.join(results_dir, "w2_ppl_baseline.json"),
              "w", encoding="utf-8") as f:
        json.dump({"meta": {"task": "ppl"}, "perplexity": ppl}, f, indent=2)

    print("[baselines]", json.dumps(
        {"pass@1_he": he["pass_at_k"]["pass@1"],
         "pass@1_mp": mp["pass_at_k"]["pass@1"],
         "ppl_python": round(ppl["python"], 2)}))
    return {"humaneval": he, "mbpp": mp, "ppl": ppl}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--bench", default="bench")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=32)
    a = p.parse_args(argv)
    run_all(a.results, limit=a.limit, max_new_tokens=a.max_new_tokens,
            bench_dir=a.bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Implementer note: agar `evaluate()` signatures me koi kwarg naam alag ho
(e.g. `timeout_s`) to REAL signature follow karo (evals/humaneval.py,
evals/mbpp.py padh lo) — assertions wale contract me koi badlav nahi.

- [ ] **Step 4: Green + suite**

Run: `python3 -m pytest tests/test_evals.py -q && python3 -m pytest tests/ -q`
Expected: PASS, suite green (smoke test seconds me, poora suite ~1 min).

- [ ] **Step 5: Commit**

```bash
git add scripts/w2_baselines.py tests/test_evals.py
git commit -m "feat(w2): deterministic baseline sweep runner"
```

---

### Task 5: Run the real baseline sweep + record numbers doc

**Files:**
- Create: `results/w2_humaneval_baseline.json`, `results/w2_mbpp_baseline.json`,
  `results/w2_ppl_baseline.json` (committed — spec M2: "results JSONs in repo";
  val_python.txt NOT committed — regenerable)
- Create: `docs/w2_baselines.md`
- Test: none new (acceptance gate like M0 Task 10)

- [ ] **Step 1: Run full sweep (limit 40, CPU)**

Run: `time python3 scripts/w2_baselines.py --results results --bench bench --limit 40`
Expected: completes in minutes; prints pass@1 0.0 / 0.0 and ppl ~10.5 (ln 260
≈ 5.56 nats/token… NOTE: actual value untrained model ke initialization par
depend karta hai; ~e^5.5≈250 se e^12 range — jo bhi aaye RECORD karo, ye
hi baseline hai). Writes 3 JSONs.

- [ ] **Step 2: Sanity-check outputs**

Run: `python3 -m pytest tests/ -q && ryth-eval report results`
Expected: suite green; report table shows all three w2_* rows.

- [ ] **Step 3: Write `docs/w2_baselines.md`**

```markdown
# W2 / M2 — Eval baselines (recorded <DATE>)

Setup: random-weight `ryth_30m` (24.3M params), near-byte BPE vocab 260,
seq_len 256, CPU. Limit 40 problems/task, n_samples=1, max_new_tokens=32.
Random weights ⇒ scores 0 expected — ye RUN proof hai, quality proof nahi.

| Benchmark | Metric | Value | File |
|---|---|---|---|
| HumanEval (40) | pass@1 | 0.0 | results/w2_humaneval_baseline.json |
| MBPP (40) | pass@1 | 0.0 | results/w2_mbpp_baseline.json |
| Held-out Python ppl | perplexity | <RECORD> | results/w2_ppl_baseline.json |

Compare karne ke liye: `ryth-eval report results`.

## Post-training rerun (30M checkpoint, FULL settings)

```bash
# w1-prep artifacts ke saath (Kaggle ya local):
ryth-eval humaneval --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json --problems_file bench/humaneval.jsonl.gz \
  --n_samples 20 --max_new_tokens 256 --ks 1,5,10
ryth-eval mbpp --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json --problems_file bench/mbpp.jsonl \
  --n_samples 20 --max_new_tokens 256 --ks 1,5,10
ryth-eval ppl --ckpt runs/ryth-kaggle/best.pt \
  --tokenizer tok/tokenizer.json \
  --files python=val_src_py.txt --files c=val_src_c.txt
ryth-eval report results --out results/table.md
```

Expectations (capability ladder memory): 30M base model pass@1 ~0–2%,
ppl clearly ln(24576)≈10.1 se neeche. Honest framing — ye prototype hai,
production-grade coding assistant nahi.

## Provenance

- HumanEval MIT (openai/human-eval), MBPP CC-BY-4.0 — bench/README.md.
- val_python.txt HumanEval content se bana tha (provisional); W1 corpus ke
  aane par val_src se replace hoga. ⚠ pass@k generated code LOCALLY execute
  karta hai — trusted machine par hi chalao (docs/evals.md).
```

Fill `<DATE>`/`<RECORD>` from actual run output — placeholders commit me
nahi jane chahiye.

- [ ] **Step 4: Commit**

```bash
git add results/ docs/w2_baselines.md
git commit -m "feat(w2): M2 baseline scores recorded (HumanEval/MBPP/ppl)"
```

## Plan Self-Review Notes

- Spec coverage: §10 M2 "results JSONs in repo" = Task 5; §5 design rule
  (every task → JSON) respected; §12 testing strategy followed (offline tests,
  deterministic seeds, regression alongside fixes). ✔
- Deliberate scope cut: chat-template/chat-mode baseline NOT run (mode="base"
  only) — chat template needs SFT-era checkpoints to be meaningful; harness
  support already shipped in M0.
- Known risk: ppl absolute value untrained-init par depend karta hai — doc me
  "jo bhi aaye RECORD karo" ruling likha hai; comparison baseline isi config
  se bana rahega (same seed/script).
- Runtime guard: smoke test limit=2/max_new_tokens=4; full sweep limit=40
  (~minutes CPU). Agar sweep >15 min lage: --limit 20 karke doc update.
