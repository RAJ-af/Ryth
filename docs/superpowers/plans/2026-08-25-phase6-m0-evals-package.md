# Ryth Phase 6 / M0 — `evals/` Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `evals/` package — chat template, pass@k metrics, sandboxed
code execution, generation helpers, HumanEval/MBPP runners, perplexity, and a
`ryth-eval` CLI — so every future Ryth checkpoint has a measurable quality
score (spec §5, milestone M0).

**Architecture:** New top-level package `evals/` following repo conventions
(sibling of `tokenizer/`, `model/`, `training/`). It *consumes* the existing
public API (`model.RythForCausalLM`, `model.generate`,
`dataset.load_bpe_tokenizer`) and modifies **no core package**. Runners accept
dependency-injected samplers so tests run offline without a real model.

**Tech Stack:** Python 3.9+ stdlib + PyTorch (already a repo dep). No new
third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-phase6-measure-first-design.md` (§5 Workstream 2, §10 M0)

## Global Constraints

- Python 3.9+ floor: every new file starts with `from __future__ import annotations`.
- `evals/` may import only from: stdlib, `torch`, and repo packages
  (`model`, `dataset`). No network calls inside library code; downloads live in
  `evals/datasets.py` as opt-in helpers and are **never** exercised by tests.
- Do not modify anything under `tokenizer/`, `dataset/`, `model/`, `training/`,
  `corpus/`. If something seems to need it, stop and flag.
- Comments/docstrings follow repo style (Hinglish + English mix).
- Every eval runner writes a JSON results file; results include config echo.
- All tests are offline and deterministic (fixed seeds); no pytest-network.
- Console script name: `ryth-eval` (entry point pattern identical to existing
  scripts in `pyproject.toml`).
- Generated code is EXECUTED by the harness — `execution.py` documents this
  risk explicitly (runs arbitrary model output locally under a timeout).

## File Structure

```
evals/
├── __init__.py        # public exports of the package
├── chat_template.py   # Ryth chat sentinels + render/parse (pure stdlib)
├── metrics.py         # pass@k estimator + aggregation (pure stdlib)
├── execution.py       # subprocess program runner w/ timeout (stdlib)
├── datasets.py        # Problem loader (.jsonl/.jsonl.gz) + download helpers
├── generation.py      # ckpt loading + sampling + stop-string/code extraction (torch)
├── humaneval.py       # HumanEval-style runner (torch optional via sampler)
├── mbpp.py            # MBPP-style runner (same harness shape)
├── ppl.py             # held-out perplexity per text file (torch)
└── cli.py             # ryth-eval CLI
tests/
├── fixtures/humaneval_tiny.jsonl   # our OWN toy problems (HumanEval schema)
├── fixtures/mbpp_tiny.jsonl        # our OWN toy problems (MBPP schema)
└── test_evals.py                   # full suite for the package
docs/evals.md                        # usage + dataset download notes
pyproject.toml                       # + "evals" package + ryth-eval script
```

---

### Task 1: Package skeleton + `chat_template.py`

**Files:**
- Create: `evals/__init__.py`
- Create: `evals/chat_template.py`
- Test: `tests/test_evals.py`

**Interfaces:**
- Consumes: tokenizer object exposing `add_special_tokens(list[str]) -> None`,
  `special_tokens: dict[str,int]`, `encode(str) -> list[int]`,
  `decode(list[int]) -> str` (the scratch BPE in `tokenizer/bpe.py` already has all four).
- Produces:
  - `CHAT_TOKENS: tuple[str, ...]` = `("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")`
  - `register_chat_tokens(tok) -> dict[str, int]` — idempotent registration; returns role→id map
  - `render(messages: list[dict], *, add_generation_prompt: bool = False) -> str`
    — each message is `{"role": "system"|"user"|"assistant", "content": str}`
  - `extract_assistant(text: str) -> str` — content after last `<|assistant|>`
    up to next `<|end|>` (or end-of-string), stripped

- [ ] **Step 1: Write the failing test**

Create `tests/test_evals.py`:

```python
"""Unit tests for the evals package.

Chat-template + metrics + execution pure stdlib; generation/ppl need torch.
Run:  python -m pytest tests/test_evals.py -v
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe import BPETokenizer
from evals.chat_template import (CHAT_TOKENS, extract_assistant,
                                 register_chat_tokens, render)


def _tok() -> BPETokenizer:
    return BPETokenizer()


def test_register_and_render_roundtrip():
    tok = _tok()
    ids = register_chat_tokens(tok)
    assert set(ids) == {"<|system|>", "<|user|>", "<|assistant|>", "<|end|>"}
    # idempotent: dobara register karne par ids na badlein
    assert register_chat_tokens(tok) == ids
    msg = [{"role": "user", "content": "hello"}]
    s = render(msg, add_generation_prompt=True)
    assert s == "<|user|>hello<|end|><|assistant|>"
    # special tokens encode me intact rehne chahiye
    enc = tok.encode(s)
    assert "<|end|>" in tok.decode(enc)


def test_render_system_and_assistant_roles():
    msgs = [
        {"role": "system", "content": "You are Ryth."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert render(msgs) == ("<|system|>You are Ryth.<|end|>"
                            "<|user|>hi<|end|>"
                            "<|assistant|>hello<|end|>")


def test_extract_assistant_cuts_at_end():
    full = "<|user|>q<|end|><|assistant|>def f():\n    return 1<|end|>"
    assert extract_assistant(full) == "def f():\n    return 1"
    # bina <|end|> ke poora tail milta hai
    assert extract_assistant("x<|assistant|>abc") == "abc"
    assert extract_assistant("no markers here") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals'`

- [ ] **Step 3: Write minimal implementation**

Create `evals/__init__.py`:

```python
"""Ryth evals — measurement harness for checkpoints (chat, pass@k, ppl).

Core pillars ko modify nahi karta; sirf unki public API use karta hai.
"""

from __future__ import annotations
```

Create `evals/chat_template.py`:

```python
"""Ryth chat format — tokenizer ke special-token system par bana.

Format (SFT data aur inference dono isi se banenge):

    <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{reply}<|end|>

Pure standard library. Tokenizer object sirf 4 methods expect karta hai jo
scratch BPE pehle se deta hai (see Interfaces in the plan).
"""

from __future__ import annotations

CHAT_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
_ROLES = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def register_chat_tokens(tok) -> dict[str, int]:
    """Chat sentinels tokenizer me add karo (agar pehle se nahi hain)."""
    have = getattr(tok, "special_tokens", {}) or {}
    missing = [t for t in CHAT_TOKENS if t not in have]
    if missing:
        tok.add_special_tokens(missing)
    return {t: tok.special_tokens[t] for t in CHAT_TOKENS}


def render(messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    """Messages -> chat-formatted string. Har turn apne sentinel se bandhta hai."""
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        if role not in _ROLES:
            raise ValueError(f"unknown role {role!r} (expected one of {sorted(_ROLES)})")
        parts.append(f"{_ROLES[role]}{m['content']}<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def extract_assistant(text: str) -> str:
    """Last assistant-turn ka content nikalo (<|end|> tak)."""
    idx = text.rfind("<|assistant|>")
    if idx == -1:
        return ""
    body = text[idx + len("<|assistant|>"):]
    end = body.find("<|end|>")
    return (body[:end] if end != -1 else body).strip()
```

Note: `BPETokenizer.add_special_tokens` silently skips tokens that already
exist (checked in `tokenizer/bpe.py:105-112`), which makes re-registration safe.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add evals/__init__.py evals/chat_template.py tests/test_evals.py
git commit -m "feat(evals): package skeleton + Ryth chat template"
```

---

### Task 2: `metrics.py` — pass@k estimator

**Files:**
- Create: `evals/metrics.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Produces:
  - `pass_at_k(n: int, c: int, k: int) -> float` — unbiased estimator
    (Chen et al., 2021): probability that at least one of k draws from n
    samples is correct, given c correct. Stable product form, no factorials.
  - `aggregate(results: list[dict], ks=(1, 5, 10)) -> dict[str, float]` —
    input items `{"task_id": str, "n": int, "n_passed": int}`; returns
    `{"pass@1": mean, "pass@5": mean, ...}` (k > max n → value for that task
    computed with k=n semantics capped: estimator formula still valid since
    comb(n-c,k)=0 when k>n-c).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
from evals.metrics import aggregate, pass_at_k


def test_pass_at_k_known_values():
    assert pass_at_k(100, 100, 1) == 1.0          # sab correct
    assert pass_at_k(100, 0, 1) == 0.0            # kuch correct nahi
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9   # c/n
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9   # 1 - C(50,1)/C(100,1)... simple: 1-(n-c)/n
    # monotone: zyada k => zyada ya barabar chance
    assert pass_at_k(20, 5, 5) >= pass_at_k(20, 5, 1)


def test_aggregate_mean_over_tasks():
    res = [{"task_id": "a", "n": 10, "n_passed": 10},
           {"task_id": "b", "n": 10, "n_passed": 5}]
    out = aggregate(res, ks=(1, 2))
    assert abs(out["pass@1"] - 0.75) < 1e-9       # (1.0 + 0.5)/2
    assert out["pass@2"] >= out["pass@1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_pass_at_k_known_values -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `evals/metrics.py`:

```python
"""pass@k — Codex paper (Chen et al. 2021) ka unbiased estimator.

Factorials se overflow hota hai, isliye stable product form:
    pass@k = 1 - prod_{i=n-c+1..n} (1 - k/i)
Pure standard library.
"""

from __future__ import annotations


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    k = min(k, n)
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def aggregate(results: list[dict], ks=(1, 5, 10)) -> dict[str, float]:
    """Per-task pass@k ka macro-average. Item: {task_id, n, n_passed}."""
    out: dict[str, float] = {}
    for k in ks:
        vals = [pass_at_k(r["n"], r["n_passed"], k) for r in results]
        out[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add evals/metrics.py tests/test_evals.py
git commit -m "feat(evals): unbiased pass@k estimator + aggregation"
```

---

### Task 3: `execution.py` — sandboxed program runner

**Files:**
- Create: `evals/execution.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Produces:
  - `ExecResult` dataclass: `ok: bool, exit_code: int | None, stdout: str,
    stderr: str, timed_out: bool`
  - `run_program(code: str, timeout_s: float = 10.0) -> ExecResult`
    — writes code to a temp `.py`, runs `[sys.executable, path]` in a fresh
    subprocess, captures output. `ok=True` iff exit code 0 and not timed out.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
from evals.execution import run_program


def test_run_program_pass_fail_timeout_syntax():
    r = run_program("print('PASS')\n")
    assert r.ok and r.stdout.strip() == "PASS" and not r.timed_out
    r = run_program("raise AssertionError('boom')\n")
    assert not r.ok and "AssertionError" in r.stderr
    r = run_program("while True:\n    pass\n", timeout_s=1.0)
    assert not r.ok and r.timed_out
    r = run_program("def broken(:\n")
    assert not r.ok and not r.timed_out          # SyntaxError = non-zero exit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_run_program_pass_fail_timeout_syntax -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `evals/execution.py`:

```python
"""Generated code ko alag subprocess me chalao (timeout ke saath).

⚠️ SECURITY NOTE: ye model-output ko LOCAL machine pe execute karta hai —
timeout ke alawa koi heavy sandboxing nahi hai (no network jail). Sirf apne
trusted box pe use karo; CI/cloud me container ke andar chalana behtar hai.
Pure standard library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_program(code: str, timeout_s: float = 10.0) -> ExecResult:
    """Python program likho, fresh interpreter me chalao, result wapas lao."""
    with tempfile.TemporaryDirectory(prefix="ryth_exec_") as td:
        path = os.path.join(td, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout_s)
            return ExecResult(ok=(p.returncode == 0), exit_code=p.returncode,
                              stdout=p.stdout[-4000:], stderr=p.stderr[-4000:],
                              timed_out=False)
        except subprocess.TimeoutExpired as e:
            return ExecResult(ok=False, exit_code=None,
                              stdout=(e.stdout or "")[-4000:] if isinstance(
                                  e.stdout, str) else "",
                              stderr=f"TIMEOUT after {timeout_s}s",
                              timed_out=True)
```

(`os` import needed at top: `import os`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py::test_run_program_pass_fail_timeout_syntax -v`
Expected: PASS (timeout case ~1s)

- [ ] **Step 5: Commit**

```bash
git add evals/execution.py tests/test_evals.py
git commit -m "feat(evals): subprocess program runner with timeout"
```

---

### Task 4: `datasets.py` + toy fixtures

**Files:**
- Create: `evals/datasets.py`
- Create: `tests/fixtures/humaneval_tiny.jsonl`
- Create: `tests/fixtures/mbpp_tiny.jsonl`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Produces:
  - `Problem` dataclass: `task_id: str, prompt: str, test: str,
    entry_point: str = "", canonical_solution: str = ""`
  - `load_problems(path: str) -> list[Problem]` — accepts `.jsonl` and
    `.jsonl.gz`; required keys `task_id`, `prompt`, `test`; optional
    `entry_point`, `canonical_solution`. Unknown keys ignored.
  - `download_humaneval(dest_dir: str) -> str` / `download_mbpp(dest_dir: str)
    -> str` — urllib GET of official sources into dest_dir, returns local
    path. NOT called anywhere in tests.

- [ ] **Step 1: Create toy fixtures (our own problems, official schema)**

`tests/fixtures/humaneval_tiny.jsonl`:

```json
{"task_id": "tiny/1", "prompt": "def add_two(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "entry_point": "add_two", "canonical_solution": "    return a + b\n", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-5, 5) == 0\n"}
{"task_id": "tiny/2", "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the largest of a, b, c.\"\"\"\n", "entry_point": "max_of_three", "canonical_solution": "    return max(a, b, c)\n", "test": "def check(candidate):\n    assert candidate(1, 2, 3) == 3\n    assert candidate(9, 2, 3) == 9\n    assert candidate(-1, -2, -3) == -1\n"}
```

`tests/fixtures/mbpp_tiny.jsonl`:

```json
{"task_id": "m1", "text": "Write a function add2(a, b) that returns the sum of a and b.", "test_list": ["assert add2(2, 3) == 5", "assert add2(0, 0) == 0"], "canonical_solution": "def add2(a, b):\n    return a + b\n"}
{"task_id": "m2", "text": "Write a function square(x) that returns x multiplied by itself.", "test_list": ["assert square(3) == 9", "assert square(-2) == 4"], "canonical_solution": "def square(x):\n    return x * x\n"}
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_evals.py`:

```python
import gzip
from evals.datasets import Problem, load_problems

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_load_problems_jsonl():
    ps = load_problems(os.path.join(_FIXTURES, "humaneval_tiny.jsonl"))
    assert [p.task_id for p in ps] == ["tiny/1", "tiny/2"]
    p = ps[0]
    assert isinstance(p, Problem)
    assert p.entry_point == "add_two"
    assert "candidate" in p.test and p.prompt.startswith("def ")


def test_load_problems_gz(tmpdir):
    src = os.path.join(_FIXTURES, "humaneval_tiny.jsonl")
    gz = os.path.join(str(tmpdir), "p.jsonl.gz")
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        fout.write(fin.read())
    assert len(load_problems(gz)) == 2


def test_load_problems_missing_key_raises(tmpdir):
    bad = os.path.join(str(tmpdir), "bad.jsonl")
    with open(bad, "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_id": "x"}) + "\n")
    try:
        load_problems(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "prompt" in str(e)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_load_problems_jsonl -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

Create `evals/datasets.py`:

```python
"""Problem-file loading (offline-first) + opt-in downloaders.

Real benchmarks (HumanEval MIT / MBPP CC-BY) manually ya download_* se lao;
tests kabhi network nahi maangte — fixtures humare khud ke toy problems hain.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass

_HUMANEVAL_URL = ("https://raw.githubusercontent.com/openai/human-eval/master/"
                  "data/HumanEval.jsonl.gz")
_MBPP_URL = ("https://raw.githubusercontent.com/google-research/google-research-datasets/"
             "master/mbpp/mbpp.jsonl")


@dataclass
class Problem:
    task_id: str
    prompt: str
    test: str
    entry_point: str = ""
    canonical_solution: str = ""


def _to_problem(d: dict) -> Problem:
    for key in ("task_id", "prompt", "test"):
        if key not in d:
            raise ValueError(f"problem record missing required key {key!r}: "
                             f"got keys {sorted(d)}")
    return Problem(task_id=str(d["task_id"]), prompt=d["prompt"], test=d["test"],
                   entry_point=d.get("entry_point", ""),
                   canonical_solution=d.get("canonical_solution", ""))


def load_problems(path: str) -> list[Problem]:
    """`.jsonl` ya `.jsonl.gz` -> list[Problem]."""
    opener = gzip.open if path.endswith(".gz") else open
    problems: list[Problem] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(_to_problem(json.loads(line)))
    return problems


def download_humaneval(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "humaneval.jsonl.gz")
    urllib.request.urlretrieve(_HUMANEVAL_URL, out)
    return out


def download_mbpp(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "mbpp.jsonl")
    urllib.request.urlretrieve(_MBPP_URL, out)
    return out
```

Note for the implementer: real MBPP records carry `text`/`test_list`, not
`prompt`/`test` — Task 7's MBPP runner converts them BEFORE calling
`load_problems` (it reads raw jsonl itself via a small adapter there; keep
`load_problems` strict for canonical schema).

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: 8 PASS

- [ ] **Step 6: Commit**

```bash
git add evals/datasets.py tests/fixtures/ tests/test_evals.py
git commit -m "feat(evals): problem loader (jsonl/gz) + offline toy fixtures"
```

---

### Task 5: `generation.py` — checkpoint loading + sampling

**Files:**
- Create: `evals/generation.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Consumes: `model.RythConfig`, `model.RythForCausalLM`, `model.generate`
  (signature: `generate(model, input_ids, *, max_new_tokens=50, temperature=1.0,
  top_k=None, eos_id=None)` → `[B, T+n] long`); checkpoint format
  `{"model": state_dict}` (training/checkpoint.py:90);
  tokenizer with `encode/decode/vocab_size/special_tokens`.
- Produces:
  - `find_eos(tok) -> int | None` — id of `<|end|>`/`<|eos|>` if registered, else None
  - `truncate_at_stops(text: str, stops: tuple[str, ...]) -> str` — cut at earliest occurrence
  - `extract_code(text: str) -> str` — strip markdown fences (` ```python ... ``` `)
    if present, else return text unchanged
  - `load_model(ckpt_path: str, vocab_size: int, *, preset: str = "ryth_30m",
    seq_len: int = 1024, device: str = "cpu") -> model` — rebuild via preset
    (same logic as `scripts/kaggle_train.py:build_model`), then
    `load_state_dict(state["model"])`, `.to(device).eval()`
  - `sample_completion(model, tok, prompt: str, *, mode="base", messages=None,
    max_new_tokens=256, temperature=0.8, top_k=40, stop_strings=("```",),
    eos_token="<|end|>") -> str` — batch-size-1 loop; renders via
    `chat_template.render` when `mode="chat"`; decodes ONLY newly generated
    tokens; truncates at stop strings; strips trailing whitespace

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
import torch
from model import RythConfig, RythForCausalLM
from model.checkpoint import save_checkpoint
from evals.generation import (extract_code, find_eos, load_model,
                              sample_completion, truncate_at_stops)


def test_truncate_at_stops():
    assert truncate_at_stops("abc```more", ("```",)) == "abc"
    assert truncate_at_stops("no stops", ("```",)) == "no stops"
    assert truncate_at_stops("a\nb\nclass X", ("\nclass", "\ndef")) == "a\nb"


def test_extract_code_strips_fences():
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("plain code") == "plain code"


def test_sample_completion_with_real_tiny_model():
    tok = _tok()
    cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=64,
                              d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    out = sample_completion(model, tok, "def f():\n",
                            max_new_tokens=8, temperature=1.0)
    assert isinstance(out, str)


def test_find_eos_none_by_default():
    assert find_eos(_tok()) is None
    tok = _tok()
    tok.add_special_tokens(["<|end|>"])
    assert find_eos(tok) == tok.special_tokens["<|end|>"]
```

And an integration test for `load_model` + roundtrip through a saved
checkpoint (uses `save_checkpoint`):

```python
def test_load_model_from_checkpoint(tmpdir):
    tok = _tok()
    cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=32,
                              d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    ck = os.path.join(str(tmpdir), "best.pt")
    save_checkpoint(ck, model, cfg)
    loaded = load_model(ck, tok.vocab_size, preset=None, seq_len=32)
    assert loaded.config.vocab_size == tok.vocab_size   # attr = .config (decoder.py:28)
    x = torch.randint(0, tok.vocab_size, (1, 4))
    logits, _ = loaded(x)
    assert logits.shape[-1] == tok.vocab_size
```

(`preset=None` means "trust the checkpoint's stored config": `load_model`
reads `state.get("config")` — saved by `model/checkpoint.py:save_checkpoint`
via `asdict(config)` — and rebuilds `RythConfig(**config_dict)`, falling back
to `preset` when absent.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_truncate_at_stops -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `evals/generation.py`:

```python
"""Checkpoint -> model, aur prompt -> completion sampling.

Core packages ko chhua bina: preset/config rebuild wahi pattern jo
scripts/kaggle_train.py use karta hai. Sampling batch-size-1 (per-sequence
EOS stop ke liye — see model.generate docstring).
"""

from __future__ import annotations

import torch

from .chat_template import register_chat_tokens, render


def find_eos(tok) -> int | None:
    for name in ("<|end|>", "<|eos|>", "<|endoftext|>"):
        tid = (getattr(tok, "special_tokens", {}) or {}).get(name)
        if tid is not None:
            return tid
    return None


def truncate_at_stops(text: str, stops: tuple[str, ...]) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_code(text: str) -> str:
    """Markdown fences ho toh andar ka code nikaalo, warna jaisa hai waisa."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]                          # ``` / ```python
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return t


def load_model(ckpt_path: str, vocab_size: int | None = None, *,
               preset: str | None = "ryth_30m", seq_len: int = 1024,
               device: str = "cpu"):
    from model import RythConfig, RythForCausalLM

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("config")
    if isinstance(cfg_dict, dict) and "d_model" in cfg_dict:
        mcfg = RythConfig(**cfg_dict)              # checkpoint apna config laaya
    else:
        assert preset is not None, "checkpoint me config nahi -- preset do"
        mcfg = getattr(RythConfig, preset)(vocab_size=vocab_size or 32000)
    if seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = seq_len
    net = RythForCausalLM(mcfg)
    net.load_state_dict(state["model"])
    return net.to(device).eval()


@torch.no_grad()
def sample_completion(model, tok, prompt: str, *, mode: str = "base",
                      messages: list[dict] | None = None, max_new_tokens: int = 256,
                      temperature: float = 0.8, top_k: int | None = 40,
                      stop_strings: tuple[str, ...] = (),
                      eos_token: str = "<|end|>") -> str:
    from model import generate

    if mode == "chat":
        register_chat_tokens(tok)
        msgs = messages if messages is not None else [{"role": "user",
                                                       "content": prompt}]
        text_prompt = render(msgs, add_generation_prompt=True)
        stops = tuple(set(stop_strings) | {eos_token})
    else:
        text_prompt = prompt
        stops = stop_strings

    ids = tok.encode(text_prompt) or [0]             # empty-prompt guard
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    eos_id = find_eos(tok) if mode == "chat" else None
    out = generate(model, x, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_k=top_k, eos_id=eos_id)
    new_ids = out[0, x.size(1):].tolist()
    text = tok.decode(new_ids)
    return truncate_at_stops(text, stops).rstrip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: all PASS (if `RythConfig(**cfg_dict)` rejects a key like
`use_gradient_checkpointing` present in older saves, pop unknown keys against
`dataclasses.fields(RythConfig)` before construction — handle explicitly).

- [ ] **Step 5: Commit**

```bash
git add evals/generation.py tests/test_evals.py
git commit -m "feat(evals): checkpoint loading + stop-aware sampling"
```

---

### Task 6: `humaneval.py` — HumanEval-style runner

**Files:**
- Create: `evals/humaneval.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Consumes: `Problem`, `run_program`, `pass_at_k`/`aggregate`,
  `sample_completion` (all earlier tasks).
- Produces:
  - `HARNESS_SUFFIX(check) -> str` internal; assembled program =
    `{prompt}{completion}\n{test}\ncheck({entry_point})\n`
  - `evaluate(problems: list[Problem], *, sampler=None, model=None, tok=None,
    n_samples: int = 20, mode: str = "base", temperature: float = 0.8,
    top_k: int | None = 40, max_new_tokens: int = 256, timeout_s: float = 10.0,
    seed: int = 1234, ks=(1, 5, 10), progress=print) -> dict`
    — `sampler(prompt) -> str` overrides the model entirely (DI for tests /
    cheap baselines); exactly ONE of sampler / (model+tok) required.
    Returns `{"meta": {...}, "tasks": [{"task_id","n","n_passed","samples_ok"}],
    "pass_at_k": {...}}`
  - `save_results(res: dict, out_path: str) -> None` — pretty JSON

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
from evals.datasets import load_problems
from evals.humaneval import evaluate, save_results

_HE = os.path.join(_FIXTURES, "humaneval_tiny.jsonl")


def test_humaneval_perfect_sampler_scores_100(tmpdir):
    problems = load_problems(_HE)
    canon = {p.task_id: p.canonical_solution for p in problems}
    res = evaluate(problems, sampler=lambda prompt: canon.get(
        _id_from_prompt(prompt, problems), ""), n_samples=3, ks=(1,))
    assert abs(res["pass_at_k"]["pass@1"] - 1.0) < 1e-9


def test_humaneval_empty_sampler_scores_0(tmpdir):
    problems = load_problems(_HE)
    res = evaluate(problems, sampler=lambda prompt: "", n_samples=2, ks=(1,))
    assert res["pass_at_k"]["pass@1"] == 0.0


def test_humaneval_results_file_written(tmpdir):
    problems = load_problems(_HE)[:1]
    canon = problems[0].canonical_solution
    res = evaluate(problems, sampler=lambda prompt: canon, n_samples=2, ks=(1,))
    out = os.path.join(str(tmpdir), "res.json")
    save_results(res, out)
    with open(out, encoding="utf-8") as f:
        back = json.load(f)
    assert back["tasks"][0]["n_passed"] == 2


def _id_from_prompt(prompt, problems):           # test helper (sampler DI demo)
    for p in problems:
        if prompt.startswith(p.prompt.split("\n")[0]):
            return p.task_id
    return ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_humaneval_empty_sampler_scores_0 -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `evals/humaneval.py`:

```python
"""HumanEval-style pass@k runner.

Har problem ke liye n samples generate karo, phir HAR sample ko alag
subprocess me `prompt + completion + test + check(entry_point)` ke saath
chalao. Sampler dependency-injected hai — tests bina model ke chalte hain.
"""

from __future__ import annotations

import json

from .datasets import Problem
from .execution import run_program
from .metrics import aggregate, pass_at_k
from .generation import extract_code


def _build_program(p: Problem, completion: str) -> str:
    code = extract_code(completion)
    return f"{p.prompt}{code}\n{p.test}\ncheck({p.entry_point})\n"


def evaluate(problems: list[Problem], *, sampler=None, model=None, tok=None,
             n_samples: int = 20, mode: str = "base", temperature: float = 0.8,
             top_k: int | None = 40, max_new_tokens: int = 256,
             timeout_s: float = 10.0, seed: int = 1234, ks=(1, 5, 10),
             progress=print) -> dict:
    if sampler is None:
        if model is None or tok is None:
            raise ValueError("sampler YA (model+tok) dono me se ek do")
        import torch
        torch.manual_seed(seed)                      # reproducible sampling
        from .generation import sample_completion

        def sampler(prompt: str) -> str:
            return sample_completion(model, tok, prompt, mode=mode,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k)

    tasks = []
    for p in problems:
        n_passed = 0
        samples_ok = []
        for _ in range(n_samples):
            completion = sampler(p.prompt)
            program = _build_program(p, completion)
            r = run_program(program, timeout_s=timeout_s)
            samples_ok.append(bool(r.ok))
            n_passed += int(r.ok)
        tasks.append({"task_id": p.task_id, "n": n_samples,
                      "n_passed": n_passed, "samples_ok": samples_ok})
        progress(f"[humaneval] {p.task_id}: {n_passed}/{n_samples}")
    res = {"meta": {"mode": mode, "n_samples": n_samples,
                    "temperature": temperature, "top_k": top_k,
                    "max_new_tokens": max_new_tokens, "seed": seed,
                    "task": "humaneval"},
           "tasks": tasks,
           "pass_at_k": aggregate(tasks, ks=ks)}
    return res


def save_results(res: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: all PASS (perfect-sampler test runs 2×3 tiny subprocess execs)

- [ ] **Step 5: Commit**

```bash
git add evals/humaneval.py tests/test_evals.py
git commit -m "feat(evals): HumanEval-style pass@k runner with DI sampler"
```

---

### Task 7: `mbpp.py` — MBPP-style runner

**Files:**
- Create: `evals/mbpp.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Consumes: same building blocks as Task 6.
- Produces:
  - `load_mbpp(path: str) -> list[Problem]` — adapter: real MBPP rows carry
    `text` + `test_list`; converts to `Problem(prompt=text, test="\n".join(test_list))`
    (also tolerates already-canonical rows with `prompt`/`test`).
  - `evaluate(...)` — same signature contract as `evals.humaneval.evaluate`
    except assembled program = `{completion}\n{test}\n` (no `check()` wrapper;
    MBPP asserts call the function directly).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
from evals.mbpp import evaluate as mbpp_evaluate, load_mbpp

_MBPP = os.path.join(_FIXTURES, "mbpp_tiny.jsonl")


def test_load_mbpp_adapter_maps_fields():
    ps = load_mbpp(_MBPP)
    assert ps[0].task_id == "m1"
    assert ps[0].prompt.startswith("Write a function add2")
    assert ps[0].test == "assert add2(2, 3) == 5\nassert add2(0, 0) == 0"


def test_mbpp_perfect_sampler_scores_100():
    problems = load_mbpp(_MBPP)
    canon = {p.task_id: p.canonical_solution for p in problems}
    res = mbpp_evaluate(problems, sampler=lambda prompt: canon.get(
        _mbpp_id(prompt, problems), ""), n_samples=2, ks=(1,))
    assert abs(res["pass_at_k"]["pass@1"] - 1.0) < 1e-9


def _mbpp_id(prompt, problems):
    for p in problems:
        if prompt.startswith(p.prompt.split(".")[0][:20]):
            return p.task_id
    return ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_load_mbpp_adapter_maps_fields -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `evals/mbpp.py`:

```python
"""MBPP-style runner — assert-based tests, koi check() wrapper nahi."""

from __future__ import annotations

import gzip
import json

from .datasets import Problem
from .generation import extract_code
from .execution import run_program
from .metrics import aggregate


def load_mbpp(path: str) -> list[Problem]:
    """MBPP rows (text/test_list) -> canonical Problem."""
    opener = gzip.open if path.endswith(".gz") else open
    problems = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" in d and "test" in d:            # already canonical
                problems.append(Problem(
                    task_id=d["task_id"], prompt=d["prompt"], test=d["test"],
                    entry_point=d.get("entry_point", ""),
                    canonical_solution=d.get("canonical_solution", "")))
                continue
            problems.append(Problem(
                task_id=str(d.get("task_id") or d.get("code", "")),
                prompt=d["text"],
                test="\n".join(d["test_list"]),
                canonical_solution=d.get("canonical_solution",
                                         d.get("code", ""))))
    return problems


def evaluate(problems, *, sampler=None, model=None, tok=None,
             n_samples: int = 20, mode: str = "base", temperature: float = 0.8,
             top_k: int | None = 40, max_new_tokens: int = 256,
             timeout_s: float = 10.0, seed: int = 1234, ks=(1, 5, 10),
             progress=print) -> dict:
    if sampler is None:
        if model is None or tok is None:
            raise ValueError("sampler YA (model+tok) dono me se ek do")
        from .generation import sample_completion

        def sampler(prompt: str) -> str:
            return sample_completion(model, tok, prompt, mode=mode,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k)

    tasks = []
    for p in problems:
        n_passed = 0
        for _ in range(n_samples):
            completion = sampler(p.prompt)
            program = f"{extract_code(completion)}\n{p.test}\n"
            n_passed += int(run_program(program, timeout_s=timeout_s).ok)
        tasks.append({"task_id": p.task_id, "n": n_samples,
                      "n_passed": n_passed})
        progress(f"[mbpp] {p.task_id}: {n_passed}/{n_samples}")
    return {"meta": {"task": "mbpp", "mode": mode, "n_samples": n_samples},
            "tasks": tasks, "pass_at_k": aggregate(tasks, ks=ks)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add evals/mbpp.py tests/test_evals.py
git commit -m "feat(evals): MBPP runner with field adapter"
```

---

### Task 8: `ppl.py` — held-out perplexity

**Files:**
- Create: `evals/ppl.py`
- Modify: `tests/test_evals.py` (append)

**Interfaces:**
- Consumes: `model(input_ids) -> (logits, cache)` (repo convention,
  `scripts/kaggle_train.py:263`); tokenizer `encode/decode`.
- Produces:
  - `perplexity(model, tok, text: str, *, seq_len: int = 512,
    device: str = "cpu") -> float` — non-overlapping windows; mean NLL over
    all predicted positions; `exp()` of it. Empty/1-token text → `inf`.
  - `evaluate_files(model, tok, files: dict[str, str], **kw) -> dict[str, float]`
    — label → filepath; returns label → ppl.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
from evals.ppl import evaluate_files, perplexity


def test_perplexity_finite_deterministic_and_reasonable():
    tok = _tok()
    cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=128,
                              d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    torch.manual_seed(0)
    model = RythForCausalLM(cfg)
    text = ("def add(a, b):\n    return a + b\n" * 20)
    p1 = perplexity(model, tok, text, seq_len=64)
    p2 = perplexity(model, tok, text, seq_len=64)
    assert p1 == p2 and p1 > 1.0                    # random init => high ppl


def test_perplexity_short_text_is_inf():
    tok = _tok()
    cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=16,
                              d_model=32, n_layers=1, n_heads=2, n_kv_heads=1)
    model = RythForCausalLM(cfg)
    assert perplexity(model, tok, "", seq_len=8) == float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_perplexity_finite_deterministic_and_reasonable -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `evals/ppl.py`:

```python
"""Held-out perplexity — C vs Python files alag-alag report karne ke liye."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, tok, text: str, *, seq_len: int = 512,
               device: str = "cpu") -> float:
    ids = tok.encode(text)
    if len(ids) < 2:
        return float("inf")
    dev = next(model.parameters()).device
    total_nll, n_pred = 0.0, 0
    for start in range(0, len(ids) - 1, seq_len):
        window = ids[start:start + seq_len + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], dtype=torch.long, device=dev)
        y = torch.tensor([window[1:]], dtype=torch.long, device=dev)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               reduction="sum")
        total_nll += loss.item()
        n_pred += y.numel()
    if n_pred == 0:
        return float("inf")
    return math.exp(total_nll / n_pred)


@torch.no_grad()
def evaluate_files(model, tok, files: dict[str, str], **kw) -> dict[str, float]:
    out = {}
    for label, path in files.items():
        with open(path, encoding="utf-8", errors="replace") as f:
            out[label] = perplexity(model, tok, f.read(), **kw)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add evals/ppl.py tests/test_evals.py
git commit -m "feat(evals): held-out perplexity evaluator"
```

---

### Task 9: `cli.py` + pyproject wiring + docs

**Files:**
- Create: `evals/cli.py`
- Create: `docs/evals.md`
- Modify: `pyproject.toml` (packages list + `[project.scripts]`)
- Modify: `evals/__init__.py` (final exports)
- Modify: `tests/test_evals.py` (append CLI smoke test)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `main(argv: list[str] | None = None) -> int` — argparse with subcommands
    `humaneval | mbpp | ppl`. Shared flags: `--ckpt PATH --tokenizer PATH
    --out PATH(default: auto `results/<task>_<ckpt-stem>.json`) --device cpu|cuda
    --preset ryth_30m --seq_len 1024`. humaneval/mbpp add:
    `--problems_file P (required) --n_samples 20 --temperature 0.8 --top_k 40
    --max_new_tokens 256 --timeout 10 --mode base|chat --ks 1,5,10`;
    ppl adds: `--files LABEL=PATH [--files ...] (required, repeatable)`.
    Prints summary + writes JSON; returns 0 on success.
  - `evals/__init__.py` exports: `Problem`, `load_problems`, `render`,
    `register_chat_tokens`, `extract_assistant`, `pass_at_k`, `aggregate`,
    `perplexity`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
def test_cli_ppl_smoke(tmpdir):
    from evals.cli import main

    tok = _tok()
    cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=64,
                              d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = RythForCausalLM(cfg)
    ck = os.path.join(str(tmpdir), "b.pt")
    save_checkpoint(ck, model, cfg)
    txt = os.path.join(str(tmpdir), "t.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("def f():\n    return 1\n" * 10)
    outj = os.path.join(str(tmpdir), "r.json")
    rc = main(["ppl", "--ckpt", ck, "--tokenizer", tk_path(tok),
               "--files", f"py={txt}", "--out", outj])
    assert rc == 0
    with open(outj, encoding="utf-8") as f:
        assert "py" in json.load(f)["perplexity"]


def tk_path(tok):                                 # helper: tokenizer ko file me save karo
    import tempfile
    d = tempfile.mkdtemp(prefix="ryth_tok_")
    p = os.path.join(d, "tokenizer.json")
    tok.save(p)                                   # API verified: bpe.py:170
    return p
```

(The smoke test's contract: CLI exits 0 and JSON contains the label key.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evals.py::test_cli_ppl_smoke -v`
Expected: FAIL — `No module named 'evals.cli'`

- [ ] **Step 3: Implement CLI + wiring**

Create `evals/cli.py`:

```python
"""ryth-eval CLI — checkpoint ki quality napo, JSON results ke saath."""

from __future__ import annotations

import argparse
import os

from .datasets import load_problems
from .generation import load_model


def _auto_out(task: str, ckpt: str) -> str:
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    os.makedirs("results", exist_ok=True)
    return os.path.join("results", f"{task}_{stem}.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ryth-eval",
                                 description="Ryth evaluation harness")
    sub = ap.add_subparsers(dest="task", required=True)

    def common(sp):
        sp.add_argument("--ckpt", required=True)
        sp.add_argument("--tokenizer", required=True)
        sp.add_argument("--out", default=None)
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--preset", default="ryth_30m")
        sp.add_argument("--seq_len", type=int, default=1024)

    for name in ("humaneval", "mbpp"):
        sp = sub.add_parser(name)
        common(sp)
        sp.add_argument("--problems_file", required=True)
        sp.add_argument("--n_samples", type=int, default=20)
        sp.add_argument("--temperature", type=float, default=0.8)
        sp.add_argument("--top_k", type=int, default=40)
        sp.add_argument("--max_new_tokens", type=int, default=256)
        sp.add_argument("--timeout", type=float, default=10.0)
        sp.add_argument("--mode", choices=("base", "chat"), default="base")
        sp.add_argument("--ks", default="1,5,10")

    sp = sub.add_parser("ppl")
    common(sp)
    sp.add_argument("--files", action="append", required=True,
                    help="LABEL=PATH (repeatable)")

    args = ap.parse_args(argv)

    from dataset import load_bpe_tokenizer
    tok = load_bpe_tokenizer(args.tokenizer)
    model = load_model(args.ckpt, tok.vocab_size, preset=args.preset,
                       seq_len=args.seq_len, device=args.device)
    ks = tuple(int(k) for k in args.ks.split(",")) if hasattr(args, "ks") else ()
    out = args.out or _auto_out(args.task, args.ckpt)

    if args.task == "ppl":
        from .ppl import evaluate_files
        files = {}
        for spec in args.files:
            label, _, path = spec.partition("=")
            files[label] = path
        scores = evaluate_files(model, tok, files, seq_len=args.seq_len,
                                device=args.device)
        result = {"meta": {"task": "ppl"}, "perplexity": scores}
    else:
        from .humaneval import evaluate as he_eval
        from .humaneval import save_results
        from . import mbpp as M
        problems = (M.load_mbpp(args.problems_file) if args.task == "mbpp"
                    else load_problems(args.problems_file))
        fn = M.evaluate if args.task == "mbpp" else he_eval
        result = fn(problems, model=model, tok=tok, n_samples=args.n_samples,
                    mode=args.mode, temperature=args.temperature,
                    top_k=args.top_k, max_new_tokens=args.max_new_tokens,
                    timeout_s=args.timeout, ks=ks)

    save_results(result, out)
    print(f"\n== RESULTS ({args.task}) ==")
    print(f"  written: {out}")
    if "pass_at_k" in result:
        for k, v in result["pass_at_k"].items():
            print(f"  {k}: {v:.4f}")
    else:
        for label, v in result["perplexity"].items():
            print(f"  ppl[{label}]: {v:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Modify `pyproject.toml`: add `"evals"` to `[tool.setuptools] packages`, and
under `[project.scripts]` add:

```toml
ryth-eval = "evals.cli:main"
```

Finalize `evals/__init__.py` exports:

```python
"""Ryth evals — measurement harness for checkpoints (chat, pass@k, ppl)."""

from __future__ import annotations

from .chat_template import CHAT_TOKENS, extract_assistant, register_chat_tokens, render
from .datasets import Problem, load_problems
from .metrics import aggregate, pass_at_k
from .ppl import perplexity

__all__ = ["CHAT_TOKENS", "Problem", "aggregate", "extract_assistant",
           "load_problems", "pass_at_k", "perplexity", "register_chat_tokens",
           "render"]
```

Create `docs/evals.md`:

```markdown
# Evals — measuring Ryth checkpoints

Offline-first quality harness: pass@k (HumanEval/MBPP-style) + held-out
perplexity. Runs on CPU (Termux-friendly).

## Quickstart

```bash
pip install -e ".[dev]"
ryth-eval ppl  --ckpt runs/x/best.pt --tokenizer tok/tokenizer.json \
              --files python=val_py.txt --files c=val_c.txt
ryth-eval humaneval --ckpt best.pt --tokenizer tok/tokenizer.json \
              --problems_file humaneval.jsonl --n_samples 20
ryth-eval mbpp     --ckpt best.pt --tokenizer tok/tokenizer.json \
              --problems_file mbpp.jsonl --n_samples 20
```

Results land in `results/*.json` — track them across runs (spec §5).

## Getting real benchmark files

```bash
python -c "from evals.datasets import download_humaneval; download_humaneval('bench')"
python -c "from evals.datasets import download_mbpp; download_mbpp('bench')"
```

HumanEval is MIT, MBPP is CC-BY-4.0. A random-weight model scores ~0% —
that is the expected baseline (spec acceptance criterion).

## Security note

The pass@k harness EXECUTES generated code locally in a subprocess with a
timeout (no network jail). Only run it on machines you trust; prefer a
container in CI.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evals.py -v`
Expected: all PASS

- [ ] **Step 5: Full-suite regression + install check**

```bash
python3 -m pytest tests/ -q                 # poora suite green (135 + naye)
pip install -e ".[dev]" -q && ryth-eval --help
```
Expected: all PASS; help prints subcommands.

- [ ] **Step 6: Commit**

```bash
git add evals/cli.py evals/__init__.py docs/evals.md pyproject.toml tests/test_evals.py
git commit -m "feat(evals): ryth-eval CLI + docs + packaging (M0 core)"
```

---

### Task 10: M0 acceptance verification

**Files:**
- No new files (verification only)
- Optional: append run-notes to spec? No — record results in commit message.

**Interfaces:**
- Consumes: complete package.

- [ ] **Step 1: Random-weight end-to-end acceptance (spec §5)**

Uses the REAL `ryth_30m` preset at byte-vocab (~24.5M params — same shape the
Kaggle run will produce), so this literally is "a random-weight 30M model
end-to-end on CPU". Expect a few minutes on a phone; fine on any laptop.

```bash
python3 - <<'EOF'
import os, tempfile, torch
from tokenizer.bpe import BPETokenizer
from model import RythConfig, RythForCausalLM
from model.checkpoint import save_checkpoint
from evals.humaneval import evaluate, save_results
from evals.datasets import load_problems

tok = BPETokenizer()                       # byte-level: vocab ≈ 259
cfg = RythConfig.ryth_30m(vocab_size=tok.vocab_size, max_seq_len=256)
model = RythForCausalLM(cfg)
print(f"params: {model.num_params()/1e6:.1f}M")   # expect ~24.5M
d = tempfile.mkdtemp(); ck = os.path.join(d, "rand.pt")
save_checkpoint(ck, model, cfg)
probs = load_problems("tests/fixtures/humaneval_tiny.jsonl")
res = evaluate(probs, model=model, tok=tok, n_samples=1, ks=(1,),
               mode="base", max_new_tokens=64)
save_results(res, os.path.join(d, "baseline.json"))
print("random-weight baseline:", res["pass_at_k"], "(≈0 expected ✅)")
EOF
```
Expected: runs clean, pass@1 = 0.0. This IS the spec's acceptance proof.

- [ ] **Step 2: Whole-suite green + commit state clean**

```bash
python3 -m pytest tests/ -q && git status --short
```
Expected: all PASS, nothing uncommitted.

## Plan Self-Review Notes (for executor awareness)

- Spec §5 unit table ↔ tasks: chat_template(T1) · humaneval(T6) · mbpp(T7) ·
  ppl(T8) · cli(T9) ✔; JSON-result rule covered by save_results everywhere ✔;
  "reuse model.generate, no core modifications" enforced by imports ✔;
  CPU/Termux-friendliness: only torch + stdlib, subprocess timeouts small ✔.
- Known deliberate simplification: single-GPU/batch-1 sampling (speed fine for
  30M-scale models; parallel sampling is post-M2 optimization — YAGNI now).
- Out of scope (separate plans): W1 Kaggle corpus+training ops, W3 `sft/`
  teacher pipeline, kNN-Memory experiment (post-M1).
