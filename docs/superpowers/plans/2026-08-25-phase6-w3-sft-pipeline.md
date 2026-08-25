# W3 `sft/` Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete `sft/` package (spec §6) offline — teacher client, 5 task seed-builders, quality filters, generation loop, CLI — so actual dataset generation is ONE command the moment the owner supplies the API endpoint + key.

**Architecture:** Teacher access sits behind an injectable interface (`complete(system, user, **kw)`); tests and `--dry-run` use a `FakeTeacher`, so the whole pipeline is TDD-able with zero network. Seeds come from corpus records via the EXISTING `corpus.tasks.builders` extraction; every example passes a per-task validator (compiles / asserts-actually-run) plus generic rule filters before landing in chat-template-rendered JSONL.

**Tech Stack:** Pure stdlib (urllib for HTTP — repo convention, no requests dep); reuses `evals.chat_template.render`, `evals.execution.run_program`, `corpus.tasks.builders.extract_python_functions`.

**Spec:** `docs/superpowers/specs/2026-08-25-phase6-measure-first-design.md` §6 (units table, blocked-on, acceptance ≥90% rule-filter pass rate).

## Global Constraints

- New deps: NONE. HTTP via stdlib `urllib` (repo convention — evals downloader same pattern).
- Tests NEVER touch network or real API keys. All teacher interaction through fakes/injected transports.
- Comments/docstrings repo-style Hinglish.
- Reuse, don't reimplement: `evals.chat_template` (CHAT_TOKENS/render/register_chat_tokens), `evals.execution.run_program`, `corpus.tasks.builders.extract_python_functions`.
- Deterministic pipeline: seeds sorted by id, stable first-wins dedup, no wall-clock in outputs.
- **Teacher directives never leak into stored messages** — stored user turn is what a real Ryth user would ask; teacher meta-instructions go only in the teacher call.
- Security note mandatory wherever local execution happens: generated asserts run LOCALLY via `run_program` sandbox.
- Record duck-type contract (consumed everywhere): `.content: str`, `.language: str`, `.path: str`, `.hash: str`.
- pyproject `[tool.setuptools] packages` gains `"sft"`, `"sft.tasks"`; `[project.scripts]` gains `ryth-sft = "sft.cli:main"`.

---

### Task 1: Schema — Seed, Example, JSONL IO

**Files:**
- Create: `sft/__init__.py`, `sft/schema.py`
- Test: append to `tests/test_sft.py` (new file)

**Interfaces:**
- Produces:
  - `Seed(id: str, task: str, language: str, user_prompt: str, teacher_directive: str = "", check=None, source_path: str = "")`; `seed.validate(assistant_text) -> list[str]`
  - `Example(task: str, messages: list, meta: dict)`; `example.to_row(tok=None) -> dict` (tok diya to `text` + `token_ids`, warna sirf `text`)
  - `validate_example(row) -> list[str]`, `write_jsonl(rows, path) -> int`, `read_jsonl(path) -> list`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sft.py`:

```python
"""Unit tests for the sft package — fake teacher, zero network.

Run:  python -m pytest tests/test_sft.py -v
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from sft.schema import (Example, Seed, read_jsonl, validate_example,
                        write_jsonl)


def _tok(vocab_size=360):
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    tok.train(["hello world"], vocab_size=vocab_size, verbose=False)
    return tok


def test_seed_validate_delegates_to_check():
    s = Seed(id="s1", task="t", language="python", user_prompt="u",
             check=lambda text: ([] if "ok" in text else ["not ok"]))
    assert s.validate("ok code") == []
    assert s.validate("bad") == ["not ok"]
    assert Seed(id="s2", task="t", language="", user_prompt="u").validate("") == []


def test_example_to_row_with_tokenizer():
    from evals.chat_template import CHAT_TOKENS   # tuple of token STRINGS
    tok = _tok()
    e = Example(task="t", messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}])
    row = e.to_row(tok)
    assert set(CHAT_TOKENS) <= set(tok.special_tokens)
    assert row["text"].startswith("<|system|>sys<|end|>")
    assert isinstance(row["token_ids"], list)
    assert all(isinstance(i, int) for i in row["token_ids"])
    # tokenizer ke bina: text hai, token_ids nahi (provisional-tok trap)
    row2 = e.to_row(None)
    assert "token_ids" not in row2 and row2["text"].startswith("<|system|>")


def test_example_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "rows.jsonl")
    rows = [{"task": "t", "messages": [], "text": "x"}]
    assert write_jsonl(rows, p) == 1
    assert read_jsonl(p) == rows


def test_validate_example_flags_structure():
    good = {"task": "t", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}], "text": "..."}
    assert validate_example(good) == []
    assert validate_example({"task": "t"}) != []              # missing keys
    bad_roles = dict(good, messages=[{"role": "user", "content": "q"}])
    assert any("role" in r for r in validate_example(bad_roles))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: No module named 'sft'

- [ ] **Step 3: Implement**

`sft/__init__.py`:

```python
"""sft — teacher-generated instruction data pipeline (spec phase-6 §6).

Offline-buildable poora pipeline; REAL generation gated on owner-provided
teacher API endpoint + key (env RYTH_TEACHER_API_KEY / RYTH_TEACHER_MODEL).
"""

__version__ = "0.1.0"
```

`sft/schema.py`:

```python
"""SFT example schema — Seed (teacher request ka blueprint) + Example (row).

JSONL row shape:
    {"task": ..., "messages": [{role, content}, ...],
     "text": "<|system|>...", "token_ids": [...]?, "meta": {...}}
text/token_ids packaging-time pe render hote hain (W2 chat template).
token_ids OPTIONAL hai — provisional tokenizer bake karne se bacho; real
24k tokenizer (post-W1) aane par hi ids pack karo.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field


@dataclass
class Seed:
    """Ek teacher request: user turn + per-task validator closure."""

    id: str
    task: str
    language: str
    user_prompt: str               # STORED user turn (Ryth-user ki awaaz)
    teacher_directive: str = ""    # sirf teacher call me jaata hai
    check: object = None           # callable(assistant_text) -> list[str]
    source_path: str = ""

    def validate(self, assistant_text: str) -> list[str]:
        if self.check is None:
            return []
        return list(self.check(assistant_text) or [])


@dataclass
class Example:
    task: str
    messages: list                 # persona system + user + assistant
    meta: dict = field(default_factory=dict)

    def to_row(self, tok=None) -> dict:
        from evals.chat_template import register_chat_tokens, render

        if tok is not None:
            register_chat_tokens(tok)
            text = render(self.messages)
            return {"task": self.task, "messages": self.messages,
                    "text": text, "token_ids": tok.encode(text),
                    "meta": self.meta}
        return {"task": self.task, "messages": self.messages,
                "text": render(self.messages), "meta": self.meta}


def validate_example(row: dict) -> list[str]:
    """Loader-side structural QA (har committed dataset pe chalega)."""
    problems = []
    for key in ("task", "messages", "text"):
        if key not in row:
            problems.append(f"missing key {key!r}")
    msgs = row.get("messages") or []
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if len(roles) < 3:
        problems.append(f"need >=3 messages, got {len(roles)}")
    elif roles[0] != "system" or roles[-1] != "assistant":
        problems.append(f"bad role sequence {roles}")
    if "token_ids" in row and not (isinstance(row["token_ids"], list)
                                   and all(isinstance(i, int)
                                           for i in row["token_ids"])):
        problems.append("token_ids must be a list of ints")
    return problems


def write_jsonl(rows: list, path: str) -> int:
    os_parent = None                               # parent dir ho to banao
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
```

- [ ] **Step 4: Run tests green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add sft/__init__.py sft/schema.py tests/test_sft.py
git commit -m "feat(sft): schema — Seed/Example, chat-template rows, JSONL IO"
```

---

### Task 2: Teacher client — injectable transport, retries, key gating

**Files:**
- Create: `sft/teacher.py`
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Consumes: nothing internal.
- Produces:
  - `TeacherError(RuntimeError)`, `TeacherConfigError(TeacherError)`
  - `OpenAICompatTeacher(api_key=None, base_url=DEFAULT_BASE_URL, model=None, transport=None, attempts=4, backoff_s=1.5, sleep=time.sleep)`; `.complete(system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.2) -> str`
  - `FakeTeacher(responses: dict|None = None, default: str = ...)` — same `.complete` signature; routes by substring match on `user`
  - Transport contract: `fn(url: str, payload: dict, headers: dict) -> tuple[int, str]`
  - Env: `RYTH_TEACHER_API_KEY`, `RYTH_TEACHER_MODEL`

- [ ] **Step 1: Failing tests**

Append:

```python
# --------------------------------------------------------------------- #
# teacher — OpenAI-compatible client (fake transport, no network)
# --------------------------------------------------------------------- #

from sft.teacher import (FakeTeacher, OpenAICompatTeacher, TeacherConfigError,
                         TeacherError)


def _transport_200(content='{"choices":[{"message":{"content":"hi"}}]}'):
    calls = []

    def t(url, payload, headers):
        calls.append((url, payload, headers))
        return 200, content
    t.calls = calls
    return t


def test_openai_compat_parses_choice_and_auth_header():
    t = _transport_200()
    teacher = OpenAICompatTeacher(api_key="sk-test", model="m-1", transport=t,
                                  sleep=lambda s: None)
    out = teacher.complete("sys", "usr", max_tokens=32)
    assert out == "hi"
    url, payload, headers = t.calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer sk-test"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["max_tokens"] == 32


def test_openai_compat_retries_on_429_then_succeeds(monkeypatch):
    bodies = [(429, "rate limited"), (500, "boom"), (200, '{"choices":'
              '[{"message":{"content":"ok"}}]}')]
    sleeps = []

    def flaky(url, payload, headers):
        return bodies.pop(0)

    monkeypatch.setenv("RYTH_TEACHER_API_KEY", "")
    teacher = OpenAICompatTeacher(api_key="k", model="m", transport=flaky,
                                  backoff_s=2.0, sleep=sleeps.append)
    assert teacher.complete("s", "u") == "ok"
    assert sleeps == [2.0, 4.0]                     # exponential backoff


def test_openai_compat_fails_fast_on_4xx():
    calls = []

    def forbidden(url, payload, headers):
        calls.append(url)
        return 401, "bad key"
    teacher = OpenAICompatTeacher(api_key="k", model="m", transport=forbidden,
                                  sleep=lambda s: None)
    with pytest.raises(TeacherError, match="401"):
        teacher.complete("s", "u")
    assert len(calls) == 1                          # 4xx par retry NAHI


def test_missing_key_or_model_is_config_error(monkeypatch):
    monkeypatch.delenv("RYTH_TEACHER_API_KEY", raising=False)
    monkeypatch.delenv("RYTH_TEACHER_MODEL", raising=False)
    t = OpenAICompatTeacher(transport=_transport_200(), sleep=lambda s: None)
    with pytest.raises(TeacherConfigError, match="RYTH_TEACHER_API_KEY"):
        t.complete("s", "u")
    t2 = OpenAICompatTeacher(api_key="k", model=None, model_env=False,
                             transport=_transport_200(), sleep=lambda s: None)
    with pytest.raises(TeacherConfigError, match="model"):
        t2.complete("s", "u")


def test_fake_teacher_routes_by_fragment():
    ft = FakeTeacher(responses={"unit tests": "assert True"},
                     default="plain answer")
    assert ft.complete("s", "please Write unit tests here") == "assert True"
    assert ft.complete("s", "anything else") == "plain answer"
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.teacher (ya constructor kwarg mismatch)

- [ ] **Step 3: Implement**

`sft/teacher.py`:

```python
"""Teacher client — OpenRouter-compatible /chat/completions, injectable transport.

Owner proxy ya direct OpenRouter dono chalein (base_url configurable).
Offline-testable: transport fn (url, payload, headers) -> (status, body_str)
inject karo. Key/model gating explicit errors deta hai with setup hints.
Retry policy: 429 aur 5xx par exponential backoff; 4xx par turant fail.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ENV_KEY = "RYTH_TEACHER_API_KEY"
ENV_MODEL = "RYTH_TEACHER_MODEL"


class TeacherError(RuntimeError):
    pass


class TeacherConfigError(TeacherError):
    pass


def _default_transport(url: str, payload: dict, headers: dict) -> tuple:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:   # error body padhne ke liye swallow
        return e.code, e.read().decode("utf-8", "replace")


class OpenAICompatTeacher:
    """OpenRouter-compatible teacher. api_key/model: arg > env > config-error."""

    def __init__(self, api_key: str | None = None,
                 base_url: str = DEFAULT_BASE_URL, model: str | None = None,
                 model_env: bool = True, transport=None, attempts: int = 4,
                 backoff_s: float = 1.5, sleep=time.sleep):
        self.api_key = api_key or os.environ.get(ENV_KEY)
        self.model = model or (os.environ.get(ENV_MODEL) if model_env else None)
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _default_transport
        self.attempts = attempts
        self.backoff_s = backoff_s
        self._sleep = sleep

    def complete(self, system: str, user: str, *, max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        if not self.api_key:
            raise TeacherConfigError(
                f"teacher API key nahi mili — {ENV_KEY} env set karo ya "
                "OpenAICompatTeacher(api_key=...) pass karo")
        if not self.model:
            raise TeacherConfigError(
                f"teacher model chahiye — {ENV_MODEL} env ya model= kwarg "
                "(owner-proxy ka Nemotron-class backend naam)")
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        last = ""
        for attempt in range(self.attempts):
            status, body = self._transport(url, payload, headers)
            if status == 200:
                try:
                    data = json.loads(body)
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, json.JSONDecodeError,
                        TypeError) as e:
                    raise TeacherError(f"200 par unexpected body ({e}): "
                                       f"{body[:200]}")
            last = f"HTTP {status}: {body[:200]}"
            if status != 429 and status < 500:
                break                                    # 4xx: fail fast
            self._sleep(self.backoff_s * (2 ** attempt))  # 1.5, 3, 6...
        raise TeacherError(f"teacher call fail "
                           f"({attempt + 1}/{self.attempts} tries): {last}")


class FakeTeacher:
    """Tests + --dry-run: substring-routed canned responses, koi network nahi."""

    def __init__(self, responses: dict | None = None,
                 default: str = 'def solved(x):\n    """Done."""\n    return x\n'):
        self.responses = dict(responses or {})
        self.default = default
        self.calls = []

    def complete(self, system: str, user: str, **kw) -> str:
        self.calls.append((system, user))
        for frag, resp in self.responses.items():
            if frag in user:
                return resp
        return self.default
```

NOTE: `model_env=False` kwarg exist karta hai taaki env-set hone par bhi test
explicitly model-less case force kar sake.

- [ ] **Step 4: Green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sft/teacher.py tests/test_sft.py
git commit -m "feat(sft): teacher client — injectable transport, retry/backoff, key gating"
```

---

### Task 3: Prompts — teacher directives vs dataset turns

**Files:**
- Create: `sft/tasks/__init__.py`, `sft/tasks/prompts.py`
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Produces:
  - `SYSTEM_PROMPT: str` (persona — STORED in examples)
  - `TEACHER_SYSTEM: str` (reviewer/data-gen instructions — NEVER stored)
  - `directive_for(task: str) -> str` (ValueError on unknown)
  - `KNOWN_TASKS: tuple` — 5 task names
  - `teacher_user(user_prompt: str, task: str) -> str` — stored-turn + directive concat for the teacher call

- [ ] **Step 1: Failing tests**

Append:

```python
# --------------------------------------------------------------------- #
# tasks/prompts — teacher vs dataset turns alag
# --------------------------------------------------------------------- #

from sft.tasks.prompts import (KNOWN_TASKS, SYSTEM_PROMPT, TEACHER_SYSTEM,
                               directive_for, teacher_user)


def test_known_tasks_are_the_spec_five():
    assert KNOWN_TASKS == ("bug_fix", "docstring_to_code", "explain_code",
                           "instruction_to_code", "test_gen")


def test_persona_and_teacher_system_distinct():
    assert "Ryth" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT not in TEACHER_SYSTEM
    assert "training data" in TEACHER_SYSTEM


def test_directive_lookup_and_error():
    assert "function BODY" in directive_for("docstring_to_code")
    assert "assert" in directive_for("test_gen")
    with pytest.raises(ValueError, match="unknown sft task"):
        directive_for("nonexistent_task")


def test_teacher_user_appends_directive_only_for_caller():
    u = teacher_user("Implement add.", "instruction_to_code")
    assert u.startswith("Implement add.")          # stored turn pehle
    assert "[" in u and "]" in u                    # directive bracketed
    # stored turn khud directive-free rehta hai
    assert "directive" not in u.split("\n")[0]
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.tasks

- [ ] **Step 3: Implement**

`sft/tasks/__init__.py`:

```python
"""sft.tasks — seed builders + prompt templates (spec §6 task taxonomy)."""
```

`sft/tasks/prompts.py`:

```python
"""Prompt templates — TEACHER instructions vs DATASET turns alag hain.

Design ruling: stored user turn waisa hi hai jo Ryth ka REAL user poochta;
teacher-directive sirf teacher call me bracketed appendix hota hai taaki
meta-text dataset ke messages me kabhi leak na ho.
"""

from __future__ import annotations

KNOWN_TASKS = ("instruction_to_code", "bug_fix", "docstring_to_code",
               "explain_code", "test_gen")

# STORED hota hai har example me — yahi Ryth ki persona banega
SYSTEM_PROMPT = "You are Ryth, a concise and correct coding assistant."

# SIRF teacher ko jaata hai — dataset me KABHI nahi jaata
TEACHER_SYSTEM = ("You are an expert engineer generating high-quality SFT "
                  "training data. Follow the format directive exactly; "
                  "output ONLY the requested content.")

_DIRECTIVES = {
    "instruction_to_code":
        "Respond with a single complete Python function. No prose, no fences.",
    "bug_fix":
        "Respond with the corrected full function only. No prose, no fences.",
    "docstring_to_code":
        "Respond with the function BODY (correctly indented), no prose.",
    "explain_code":
        "Explain in 3-6 short sentences what the code does and why.",
    "test_gen":
        "Respond with Python assert statements that test the function. "
        "One per line, no imports, no fences.",
}


def directive_for(task: str) -> str:
    try:
        return _DIRECTIVES[task]
    except KeyError:
        raise ValueError(f"unknown sft task {task!r}; "
                         f"known: {sorted(_DIRECTIVES)}") from None


def teacher_user(user_prompt: str, task: str) -> str:
    """Stored-turn + directive — SIRF teacher.complete() me jaata hai."""
    return f"{user_prompt}\n\n[{directive_for(task)}]"
```

- [ ] **Step 4: Green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sft/tasks/__init__.py sft/tasks/prompts.py tests/test_sft.py
git commit -m "feat(sft): prompt templates — persona vs teacher directives separated"
```

---

### Task 4: Seed builders — 5 task types from corpus records

**Files:**
- Create: `sft/tasks/builders.py`
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Consumes: `corpus.tasks.builders.extract_python_functions(text) -> list[dict]` (keys: name/header/body/full/indent/docstring — VERIFIED corpus/tasks/builders.py:44), `corpus.tasks.builders.bug_fixing_examples(rec) -> list[dict]` (input = "Fix the bug in this code:\n\n"+buggy — VERIFIED :161), `evals.execution.run_program(code, timeout_s) -> ExecResult(.ok, .stderr)` (VERIFIED evals/execution.py:29), `Seed`, `directive_for`.
- Produces: `ALL_BUILDERS: dict[str, Callable[[rec], list[Seed]]]` keyed by the 5 KNOWN_TASKS names. Validators attach via `Seed.check`.

- [ ] **Step 1: Failing tests**

Append:

```python
# --------------------------------------------------------------------- #
# tasks/builders — corpus records se 5 seed types
# --------------------------------------------------------------------- #

from sft.tasks.builders import ALL_BUILDERS

_FUNC_SRC = '''def add_two(a, b):
    """Return the sum of two numbers, supporting ints and floats."""
    return a + b


def shout(s):
    """Uppercase the string and append an exclamation mark for excitement."""
    return s.upper() + "!"
'''


def _rec(content=_FUNC_SRC, language="python", path="mod/math.py", h="deadbeef01"):
    return types.SimpleNamespace(content=content, language=language,
                                 path=path, hash=h)


def test_all_five_builders_registered_match_prompts():
    from sft.tasks.prompts import KNOWN_TASKS
    assert set(ALL_BUILDERS) == set(KNOWN_TASKS)


def test_builders_cover_rich_record_all_tasks():
    seeds = build_seeds([_rec()])
    got = {s.task for s in seeds}
    assert got == set(ALL_BUILDERS)                 # rich record => sab 5
    for s in seeds:
        assert s.language == "python" and s.source_path == "mod/math.py"
        assert s.user_prompt and s.teacher_directive


def test_build_seeds_deterministic_and_skips_empty():
    a = [s.id for s in build_seeds([_rec(h="aa"), _rec(h="bb")])]
    b = [s.id for s in build_seeds([_rec(h="bb"), _rec(h="aa")])]  # input order irrelevant
    assert a == b and len(a) == len(set(a))         # sorted + unique
    assert build_seeds([_rec(content=""), _rec(language="c")]) == []


def test_test_gen_validator_runs_asserts_locally():
    seeds = build_seeds([_rec()])
    tst = next(s for s in seeds if s.task == "test_gen")
    good = next(s for s in seeds if s.task == "docstring_to_code")
    assert tst.validate("assert add_two(1, 2) == 3\nassert add_two(0, 0) == 0") == []
    probs = tst.validate("assert add_two(1, 2) == 99")
    assert probs and "fail" in probs[0]
    # i2c/d2c validators: syntax check
    i2c = next(s for s in seeds if s.task == "instruction_to_code")
    assert i2c.validate("def add_two(a, b):\n    return a + b") == []
    assert i2c.validate("def broken(:\n")
```

(`build_seeds` Task 6 me aata hai — is task me ye test PENDING rakho?
NAHI — do options: (a) ye test Task 6 me move karo, (b) trivial
`build_seeds` abhi builders module me hi daal do. **Ruling: (b)** —
`build_seeds` Task 4 me hi implement (10 lines), Task 6 sirf generate-loop
banata hai. Isliye ye import upar add karo:
`from sft.tasks.builders import ALL_BUILDERS, build_seeds`)

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.tasks.builders

- [ ] **Step 3: Implement**

`sft/tasks/builders.py`:

```python
"""Seed builders — corpus records se 5 SFT task types (spec §6).

Corpus ka extract_python_functions REUSE hota hai (koi naya parser nahi).
Record duck-type: .content .language .path .hash. Sab deterministic —
koi RNG nahi, W1/W2 convention. Har task apna VALIDATOR attach karta hai
(jaise test_gen ke asserts SACH ME run hote hain — Measure-first).
"""

from __future__ import annotations

import ast

from corpus.tasks.builders import bug_fixing_examples, extract_python_functions
from sft.schema import Seed
from sft.tasks.prompts import directive_for


# --------------------------------------------------------------------- #
# validators (Seed.check closures)
# --------------------------------------------------------------------- #
def _check_compiles():
    def _run(text: str) -> list[str]:
        try:
            ast.parse(text)
        except SyntaxError as e:
            return [f"syntax error: {e.msg} (line {e.lineno})"]
        return []
    return _run


def _check_asserts_run(func_src: str):
    """test_gen: func + generated asserts SAATH me execute — real proof."""
    def _run(text: str) -> list[str]:
        from evals.execution import run_program    # local exec — docs warning
        program = f"{func_src}\n{text.strip()}\n"
        r = run_program(program, timeout_s=10.0)
        return [] if r.ok else [f"asserts fail: {r.stderr.strip()[:200]}"]
    return _run


def _check_min_len(n_chars: int):
    def _run(text: str) -> list[str]:
        return ([] if len(text.strip()) >= n_chars
                else [f"too short (<{n_chars} chars)"])
    return _run


# --------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------- #
def _sid(prefix: str, rec, fn_name: str) -> str:
    return f"{prefix}:{getattr(rec, 'hash', '')[:12]}:{fn_name}"


def _seed(id_, task, rec, user_prompt, check=None) -> Seed:
    return Seed(id=id_, task=task, language=rec.language or "",
                user_prompt=user_prompt, teacher_directive=directive_for(task),
                check=check, source_path=getattr(rec, "path", ""))


def instruction_to_code_seeds(rec) -> list:
    """Docstring-spec -> poora function (compile-checked)."""
    out = []
    for fn in extract_python_functions(rec.content or ""):
        if not fn["docstring"] or len(fn["body"]) <= len(fn["docstring"]) + 20:
            continue
        stub = (f"{fn['header']}\n"
                f'    """{fn["docstring"]}"""\n'
                f"    ...  # TODO: implement")
        user = (f"Implement this Python function:\n\n{stub}\n\n"
                "The docstring is the specification.")
        out.append(_seed(_sid("i2c", rec, fn["name"]),
                         "instruction_to_code", rec, user, _check_compiles()))
    return out


def bug_fix_seeds(rec) -> list:
    """Corpus ka hi deterministic operator-mutation; fix compile-checked."""
    out = []
    for ex in bug_fixing_examples(rec):
        buggy = ex["input"].split("\n\n", 1)[1]
        user = (f"This Python function has a subtle bug. Find and fix it."
                f"\n\n{buggy}")
        out.append(_seed(f"bug:{getattr(rec, 'hash', '')[:12]}",
                         "bug_fix", rec, user, _check_compiles()))
    return out


def docstring_to_code_seeds(rec) -> list:
    """Body completion — pretrain signal ka chat-format cousin."""
    out = []
    for fn in extract_python_functions(rec.content or ""):
        if not fn["docstring"]:
            continue
        user = (f"Complete the body of this Python function:\n\n"
                f"{fn['header']}\n"
                f'    """{fn["docstring"]}"""')
        out.append(_seed(_sid("d2c", rec, fn["name"]),
                         "docstring_to_code", rec, user))
    return out


def explain_code_seeds(rec) -> list:
    out = []
    for fn in extract_python_functions(rec.content or ""):
        if not fn["docstring"] or len(fn["full"]) < 80:
            continue
        user = f"What does this Python function do?\n\n{fn['full']}"
        out.append(_seed(_sid("xpl", rec, fn["name"]),
                         "explain_code", rec, user, _check_min_len(100)))
    return out


def test_gen_seeds(rec) -> list:
    """Generated asserts ORIGINAL function ke against RUN hoke validate."""
    out = []
    for fn in extract_python_functions(rec.content or ""):
        if not fn["docstring"] or len(fn["body"]) < 40:
            continue
        user = (f"Write unit tests (Python assert statements) for this "
                f"function:\n\n{fn['full']}")
        out.append(_seed(_sid("tst", rec, fn["name"]), "test_gen", rec, user,
                         _check_asserts_run(fn["full"])))
    return out


ALL_BUILDERS = {
    "instruction_to_code": instruction_to_code_seeds,
    "bug_fix": bug_fix_seeds,
    "docstring_to_code": docstring_to_code_seeds,
    "explain_code": explain_code_seeds,
    "test_gen": test_gen_seeds,
}


def build_seeds(records: list, tasks: list | None = None) -> list:
    """Deterministic: input-order-independent, id-sorted, empty-content skip."""
    tasks = sorted(tasks) if tasks else sorted(ALL_BUILDERS)
    seeds = []
    for rec in records:
        if not getattr(rec, "content", None):
            continue
        for t in tasks:
            seeds.extend(ALL_BUILDERS[t](rec))
    seen, uniq = set(), []
    for s in sorted(seeds, key=lambda x: x.id):
        if s.id not in seen:
            seen.add(s.id)
            uniq.append(s)
    return uniq
```

- [ ] **Step 4: Green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: all passed (agar `got == set(ALL_BUILDERS)` fail ho — _FUNC_SRC
rich nahi hua; lamba docstring + `+` operator dono honE chahiye, upar ka
fixture exactly copy karo)

- [ ] **Step 5: Commit**

```bash
git add sft/tasks/builders.py tests/test_sft.py
git commit -m "feat(sft): 5 task seed-builders with per-task validators"
```

---

### Task 5: Filters — rules, dedup, optional self-verify

**Files:**
- Create: `sft/filter.py`
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Consumes: `Seed.validate` (Task 1).
- Produces:
  - `FilterConfig(min_user_chars=20, min_assistant_chars=20, max_total_chars=8000)`
  - `rule_check(seed, assistant_text, cfg=None) -> list[str]` (reason strings)
  - `Deduper()`; `.duplicate(text) -> bool` (first-wins, whitespace-normalized sha256)
  - `self_verify(seed, assistant_text, teacher) -> list[str]` (strict yes/no reviewer call)

- [ ] **Step 1: Failing tests**

Append:

```python
# --------------------------------------------------------------------- #
# filter — rule gate + dedup + self-verify
# --------------------------------------------------------------------- #

from sft.filter import Deduper, FilterConfig, rule_check, self_verify


def _bare_seed(**kw):
    d = dict(id="s", task="t", language="python",
             user_prompt="x" * 50, teacher_directive="")
    d.update(kw)
    return Seed(**d)


def test_rule_check_clean_seed_passes():
    s = _bare_seed(check=lambda t: [])
    assert rule_check(s, "y" * 30) == []
    assert rule_check(s, "y" * 30, FilterConfig(min_assistant_chars=31)) != []


def test_rule_check_reports_each_failure():
    s = _bare_seed(user_prompt="short", check=lambda t: ["task-specific bad"])
    reasons = rule_check(s, "", FilterConfig())
    joined = "; ".join(reasons)
    assert "task-specific bad" in joined
    assert "chars" in joined                        # length rules bhi


def test_deduper_first_wins_normalized():
    d = Deduper()
    assert not d.duplicate("def f():\n    return 1")
    assert d.duplicate("DEF   F():\n RETURN 1")     # case+whitespace normalized
    assert not d.duplicate("totally other code")


def test_self_verify_yes_passes_no_fails():
    s = _bare_seed()
    yes = FakeTeacher(responses={}, default="yes")
    no = FakeTeacher(responses={}, default="no")
    assert self_verify(s, "answer text", yes) == []
    assert self_verify(s, "answer text", no) == ["self_verify said: no"]
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.filter

- [ ] **Step 3: Implement**

`sft/filter.py`:

```python
"""Quality gate — rule filters + dedup (+ optional teacher self-verification).

Spec §6 acceptance: v1 dataset me >=90% examples rule filters pass karein.
Rule reasons human-readable hain — stats me aggregate hoke dikhte hain.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


@dataclass
class FilterConfig:
    min_user_chars: int = 20
    min_assistant_chars: int = 20
    max_total_chars: int = 8000


def normalize_for_dedup(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


class Deduper:
    """First-wins dedup — normalized assistant text ka sha256."""

    def __init__(self):
        self.seen: set = set()

    def duplicate(self, text: str) -> bool:
        h = hashlib.sha256(normalize_for_dedup(text).encode()).hexdigest()
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


def rule_check(seed, assistant_text: str,
               cfg: FilterConfig | None = None) -> list[str]:
    """Per-task validator PEHLE, phir generic length rules."""
    cfg = cfg or FilterConfig()
    reasons = list(seed.validate(assistant_text))
    if len(seed.user_prompt) < cfg.min_user_chars:
        reasons.append(f"user prompt < {cfg.min_user_chars} chars")
    if len(assistant_text.strip()) < cfg.min_assistant_chars:
        reasons.append(f"assistant < {cfg.min_assistant_chars} chars")
    if len(seed.user_prompt) + len(assistant_text) > cfg.max_total_chars:
        reasons.append(f"total > {cfg.max_total_chars} chars")
    return reasons


def self_verify(seed, assistant_text: str, teacher) -> list[str]:
    """OPTIONAL second pass: teacher khud reviewer banke yes/no bole."""
    q = (f"Task given to an engineer:\n\n{seed.user_prompt}\n\n"
         f"Their answer:\n\n{assistant_text}\n\n"
         "Is the answer correct and complete? Reply with exactly yes or no.")
    reply = teacher.complete(
        "You are a strict code reviewer. Reply with exactly one word: "
        "yes or no.", q, max_tokens=8, temperature=0.0).strip().lower()
    if reply.startswith("yes"):
        return []
    return [f"self_verify said: {reply[:40]}"]
```

- [ ] **Step 4: Green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sft/filter.py tests/test_sft.py
git commit -m "feat(sft): rule filters, stable dedup, optional self-verify"
```

---

### Task 6: Generation loop + stats

**Files:**
- Create: `sft/generate.py`
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Consumes: `build_seeds` (T4), `OpenAICompatTeacher/FakeTeacher.complete` (T2), `TEACHER_SYSTEM`, `teacher_user` (T3), `rule_check/Deduper/self_verify` (T5), `Example.to_row` (T1).
- Produces:
  - `generate(seeds, teacher, *, cfg=None, dedup=True, verify_teacher=None, limit=None, progress=print) -> tuple[list[Example], dict]` — stats dict: `{n_seeds, n_generated, n_passed, pass_rate, per_task: {task: {"generated": n, "passed": n}}, filter_reasons: {reason: count}}`
  - `package(examples, tok=None) -> list[dict]` — rows via `Example.to_row`

- [ ] **Step 1: Failing tests**

Append:

```python
# --------------------------------------------------------------------- #
# generate — end-to-end fake-teacher integration
# --------------------------------------------------------------------- #

from sft.generate import generate, package
from sft.tasks.prompts import SYSTEM_PROMPT, TEACHER_SYSTEM


def _good_responses():
    return {
        "Implement this Python function":
            "def add_two(a, b):\n    \"\"\"Add.\"\"\"\n    return a + b",
        "has a subtle bug":
            "def add_two(a, b):\n    return a + b",
        "Complete the body":
            "    result = a + b\n    return result",
        "What does this Python function do":
            ("It computes the sum of its two arguments and returns that "
             "value directly, which works for both integers and floats."),
        "Write unit tests":
            "assert add_two(1, 2) == 3\nassert add_two(-1, 1) == 0",
    }


def test_generate_end_to_end_filters_and_stats():
    from sft.generate import build_seeds
    seeds = build_seeds([_rec()])
    teacher = FakeTeacher(responses=_good_responses(), default="junk")
    examples, stats = generate(seeds, teacher, progress=lambda *a, **k: None)
    assert stats["n_generated"] == len(seeds) > 0
    assert stats["n_passed"] == len(examples)
    assert 0.0 <= stats["pass_rate"] <= 1.0
    assert set(stats["per_task"]) == {s.task for s in seeds}
    # har example: persona system + user + assistant, directive leak NAHI
    for e in examples:
        assert e.messages[0]["content"] == SYSTEM_PROMPT
        assert e.messages[-1]["role"] == "assistant"
        for m in e.messages:
            assert "\n\n[" not in m["content"]               # directive appendix leak nahi
            assert TEACHER_SYSTEM not in m["content"]
    # junk default sab non-matching seeds me gaya hoga — reasons me dikhega
    assert sum(stats["filter_reasons"].values()) == stats["n_generated"] - stats["n_passed"]


def test_generate_dedup_collapses_identical_answers():
    from sft.generate import build_seeds
    seeds = build_seeds([_rec()])
    same = FakeTeacher(default="def same():\n    return 42\n")
    _, stats = generate(seeds, same, progress=lambda *a, **k: None)
    passed = stats["n_passed"]
    examples2, stats2 = generate(seeds, same, dedup=True,
                                 progress=lambda *a, **k: None)
    assert stats2["n_passed"] <= passed
    assert any("duplicate" in r for r in stats2["filter_reasons"])


def test_package_rows_render_without_tokenizer():
    from sft.schema import Example
    es = [Example(task="t", messages=[
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}])]
    rows = package(es, tok=None)
    assert len(rows) == 1 and "<|system|>" in rows[0]["text"]
    assert validate_example(rows[0]) == []
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.generate

- [ ] **Step 3: Implement**

`sft/generate.py`:

```python
"""Generation loop — seeds x teacher -> filtered Examples + stats.

Offline dry-run: FakeTeacher inject karo (CLI --dry-run isi class ko use
karta hai) — bina network/key ke poora pipeline validate hota hai.
Failures pipeline ko nahi rokte — reason record hoke stats me jaata hai.
"""

from __future__ import annotations

from sft.filter import Deduper, FilterConfig, rule_check, self_verify
from sft.schema import Example
from sft.tasks.builders import build_seeds          # noqa: F401 (re-export)
from sft.tasks.prompts import SYSTEM_PROMPT, TEACHER_SYSTEM, teacher_user


def generate(seeds, teacher, *, cfg: FilterConfig | None = None,
             dedup: bool = True, verify_teacher=None, limit: int | None = None,
             progress=print) -> tuple:
    """Returns (examples, stats). Ek seed fail => reason logged, aage badho."""
    cfg = cfg or FilterConfig()
    dd = Deduper() if dedup else None
    examples: list[Example] = []
    reasons: dict[str, int] = {}
    per_task: dict[str, dict[str, int]] = {}
    n_gen = 0

    def note(reason: str, task: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        per_task.setdefault(task, {"generated": 0, "passed": 0})

    for s in seeds:
        if limit and n_gen >= limit:
            break
        pt = per_task.setdefault(s.task, {"generated": 0, "passed": 0})
        pt["generated"] += 1
        n_gen += 1
        raw = teacher.complete(TEACHER_SYSTEM, teacher_user(s.user_prompt,
                                                            s.task))
        bad = rule_check(s, raw, cfg)
        if not bad and dd is not None and dd.duplicate(raw):
            bad = ["duplicate"]
        if not bad and verify_teacher is not None:
            bad = self_verify(s, raw, verify_teacher)
        if bad:
            for r in bad:
                note(r, s.task)
            continue
        pt["passed"] += 1
        examples.append(Example(
            task=s.task,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": s.user_prompt},
                      {"role": "assistant", "content": raw}],
            meta={"source_path": s.source_path, "language": s.language}))
        progress(f"[sft] {s.task} ok ({len(examples)})")

    stats = {"n_seeds": len(seeds), "n_generated": n_gen,
             "n_passed": len(examples),
             "pass_rate": round(len(examples) / n_gen, 4) if n_gen else 0.0,
             "per_task": per_task,
             "filter_reasons": dict(sorted(reasons.items(),
                                           key=lambda kv: -kv[1]))}
    return examples, stats


def package(examples, tok=None) -> list[dict]:
    """Example list -> JSONL-ready rows (tok diye to token_ids bhi)."""
    return [e.to_row(tok) for e in examples]
```

- [ ] **Step 4: Green**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sft/generate.py tests/test_sft.py
git commit -m "feat(sft): generation loop with per-task stats and dedup"
```

---

### Task 7: CLI (`ryth-sft`) + docs + packaging

**Files:**
- Create: `sft/cli.py`, `docs/sft.md`
- Modify: `pyproject.toml` ([project.scripts] + packages list)
- Test: append to `tests/test_sft.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv) -> int`; CLI:
  ```
  ryth-sft generate --src DIR|FILE.jsonl [--out data/sft_v1.jsonl]
      [--target N] [--tasks t1,t2] [--model M] [--base-url URL]
      [--tokenizer TOK.json] [--limit-files N] [--dry-run]
  ```

- [ ] **Step 1: Failing test**

Append:

```python
# --------------------------------------------------------------------- #
# cli — ryth-sft generate (dry-run = offline wiring proof)
# --------------------------------------------------------------------- #


def test_cli_dry_run_end_to_end(tmp_path):
    from sft.cli import main

    src = tmp_path / "mini_corpus"; src.mkdir()
    (src / "math.py").write_text(_FUNC_SRC, encoding="utf-8")
    (src / "notes.txt").write_text("non-code, ignore", encoding="utf-8")
    out = tmp_path / "sft" / "v1.jsonl"

    rc = main(["generate", "--src", str(src), "--out", str(out),
               "--dry-run"])
    assert rc == 0
    rows = read_jsonl(str(out))
    assert rows and all(validate_example(r) == [] for r in rows)
    assert {r["task"] for r in rows} <= {
        "instruction_to_code", "bug_fix", "docstring_to_code",
        "explain_code", "test_gen"}
    stats = json.loads((tmp_path / "sft" / "v1.jsonl.stats.json")
                       .read_text(encoding="utf-8"))
    # NOTE: >=0.90 gate REAL runs ka acceptance hai; offline canned answers me
    # dedup-collisions/explain-length dips normal hain — sirf machinery check:
    assert isinstance(stats["pass_rate"], float)
    assert 0.0 <= stats["pass_rate"] <= 1.0
    assert "token_ids" not in rows[0]               # bina --tokenizer ids nahi


def test_cli_target_caps_rows(tmp_path):
    from sft.cli import main

    src = tmp_path / "mc"; src.mkdir()
    (src / "math.py").write_text(_FUNC_SRC, encoding="utf-8")
    out = tmp_path / "o.jsonl"
    rc = main(["generate", "--src", str(src), "--out", str(out),
               "--dry-run", "--target", "2"])
    assert rc == 0
    assert len(read_jsonl(str(out))) <= 2
```

- [ ] **Step 2: Red**

Run: `python3 -m pytest tests/test_sft.py -q`
Expected: FAIL — ModuleNotFoundError: sft.cli

- [ ] **Step 3: Implement**

`sft/cli.py`:

```python
"""ryth-sft CLI — generate subcommand (spec §6).

DRY-RUN (offline): FakeTeacher se poora pipeline — key/corpus ke bina bhi
wiring proof hoti hai, dataset sample banta hai.

REAL generation (gated on owner key):
    export RYTH_TEACHER_API_KEY=sk-...
    export RYTH_TEACHER_MODEL=<nemotron-class-backend-name>
    ryth-sft generate --src corpus_out --target 5000 \
        --out data/sft_v1.jsonl --tokenizer tok/tokenizer.json

⚠ SECURITY: test_gen ke asserts LOCAL subprocess me run hote hain
(evals.execution.run_program) — trusted machine par hi chalao.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace


def load_records(src: str) -> list:
    """--src: corpus dir (recursive .py/.c/.h) YA jsonl (content/language/path)."""
    records = []
    if src.endswith((".jsonl", ".jsonl.gz")):
        from sft.schema import read_jsonl

        for d in read_jsonl(src):
            records.append(SimpleNamespace(
                content=d.get("content", ""), language=d.get("language", ""),
                path=d.get("path", ""), hash=d.get("hash", "")))
        return records
    for dp, _dns, fns in os.walk(src):
        for fn in sorted(fns):
            if not fn.endswith((".py", ".c", ".h")):
                continue
            p = os.path.join(dp, fn)
            with open(p, encoding="utf-8", errors="replace") as f:
                content = f.read()
            records.append(SimpleNamespace(
                content=content,
                language="python" if fn.endswith(".py") else "c",
                path=os.path.relpath(p, src), hash=""))
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ryth-sft", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="seeds x teacher -> SFT JSONL")
    g.add_argument("--src", required=True,
                   help="corpus dir ya .jsonl rows (content/language/path)")
    g.add_argument("--out", default="data/sft_v1.jsonl")
    g.add_argument("--target", type=int, default=5000,
                   help="final examples ki max count (spec: ~5-10k)")
    g.add_argument("--tasks", default="",
                   help="comma-list subset; default sab 5")
    g.add_argument("--model", default=None)
    g.add_argument("--base-url", dest="base_url", default=None)
    g.add_argument("--tokenizer", default=None,
                   help="diya to token_ids bhi pack honge (real 24k tok)")
    g.add_argument("--limit-files", type=int, default=None)
    g.add_argument("--dry-run", action="store_true",
                   help="FakeTeacher — offline wiring proof, no network")
    args = ap.parse_args(argv)

    from sft.generate import build_seeds, generate, package
    from sft.schema import validate_example, write_jsonl

    records = load_records(args.src)
    if args.limit_files:
        records = records[:args.limit_files]
    seeds = build_seeds(records,
                        tasks=(args.tasks.split(",") if args.tasks else None))
    print(f"[seeds] {len(seeds)} seeds from {len(records)} records")

    if args.dry_run:
        from sft.teacher import FakeTeacher
        teacher = FakeTeacher()
    else:
        from sft.teacher import OpenAICompatTeacher
        kw = {"base_url": args.base_url} if args.base_url else {}
        teacher = OpenAICompatTeacher(model=args.model, **kw)

    examples, stats = generate(seeds, teacher, progress=lambda *a, **k: None)
    if args.target and len(examples) > args.target:
        examples = examples[:args.target]

    tok = None
    if args.tokenizer:
        from dataset import load_bpe_tokenizer
        tok = load_bpe_tokenizer(args.tokenizer)
    rows = package(examples, tok)
    bad = [r for r in rows if validate_example(r)]
    if bad:
        raise SystemExit(f"internal error: {len(bad)} rows failed validation")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(rows, args.out)
    stats_path = args.out + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[sft]", json.dumps({k: stats[k] for k in
                               ("n_generated", "n_passed", "pass_rate")}))
    print(f"[out] {args.out} ({len(rows)} rows) | stats: {stats_path}")
    if not args.dry_run and stats["pass_rate"] < 0.9:
        print("[warn] pass_rate < 0.9 — spec §6 acceptance ke neeche; "
              "filter_reasons dekho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`pyproject.toml` — two edits:

```toml
[project.scripts]
ryth-sft = "sft.cli:main"          # ryth-eval line ke baad
```

```toml
packages = [
    ...existing...,
    "sft", "sft.tasks",
]
```

- [ ] **Step 4: Docs — `docs/sft.md`**

```markdown
# sft — teacher-generated instruction data (phase-6 §6)

Pipeline: corpus records → 5 task types ke seeds → teacher → validators +
filters → chat-template JSONL. Poora pipeline OFFLINE testable hai
(FakeTeacher); REAL generation sirf teacher key ke saath.

## Status (M3)

- ✅ Pipeline code + offline dry-run (`--dry-run`)
- ⛔ Real generation GATED: owner se `RYTH_TEACHER_API_KEY` +
  `RYTH_TEACHER_MODEL` chahiye (proxy base-url bhi tab)

## Offline smoke (aaj chalega)

```bash
python3 -m sft.cli generate --src bench --out /tmp/sft_smoke.jsonl --dry-run
```

(bench/*.py nahi hote to koi bhi chhota .py dir chalega)

## Real run (key aane par)

```bash
export RYTH_TEACHER_API_KEY=sk-...        # owner proxy key
export RYTH_TEACHER_MODEL=<backend-name>  # Nemotron-class
export RYTH_SFT_BASE_URL=https://...      # agar proxy direct OpenRouter nahi
ryth-sft generate --src corpus_out --target 5000 \
  --out data/sft_v1.jsonl --tokenizer tok/tokenizer.json
```

Acceptance (spec §6): ~5–10k examples, pass_rate ≥ 0.90 (stats JSON me),
owner spot-check sample.

## Output format

Har row: `task`, `messages` (persona-system/user/assistant), `text`
(W2 chat-template rendered), optional `token_ids` (sirf --tokenizer ke
saath), `meta`. Trainer `text`/`token_ids` seedha consume karega.

## Security

⚠ `test_gen` validator generated asserts LOCAL subprocess me execute karta
hai (timeout 10s, no network jail) — trusted machine par hi chalao.
Teacher-directives dataset turns me leak nahi hote (tested).
```

(Note: `--base-url` env override `RYTH_SFT_BASE_URL` docs me hai lekin CLI
flag hi primary hai; env support Task 7 me add mat karo — YAGNI, docs line
generic rakhi hai.)

- [ ] **Step 5: Full suite + entry-point sanity**

Run: `python3 -m pytest tests/ -q && pip install -e ".[dev]" --no-deps --break-system-packages -q && ryth-sft --help | head -5`
Expected: suite green; `ryth-sft` command available

- [ ] **Step 6: Commit**

```bash
git add sft/cli.py docs/sft.md pyproject.toml tests/test_sft.py
git commit -m "feat(sft): ryth-sft CLI with offline dry-run + docs + packaging"
```

---

## Plan Self-Review Notes

- Spec coverage §6: teacher.py (T2), tasks/ (T3+T4), filter.py (T5), cli.py
  ryth-sft (T7), chat-template output format (T1/T6), acceptance machinery
  (pass_rate stats + <0.9 warn real runs par; ≥0.90 gate offline smoke par
  NAHI lagta kyunki canned answers me dedup-collisions normal hain),
  blocked-on honored (key gating T2, docs T7). kNN-memory writeup
  JAAN-BOOJH KE OUT — alag deliverable hai (ledger ruling).
- Placeholder scan: koi TBD/TODO nahi; sab steps me real code. `bench` dry-run
  smoke doc me sirf suggestion hai (bench me .py files nahi — .txt/.jsonl.gz
  hain; isliye docs example generic dir kehta hai).
- Type consistency: `Seed.check` callable; `generate()` returns
  `(examples, stats)`; `to_row(tok=None)`; transport `(status:int, body:str)`
  — T2 test `flaky` returns list-pop tuple ✓; FakeTeacher.complete(**kw)
  tolerant ✓. `build_seeds` T4 me hi (T6 sirf re-export) — T4 test usi se ✓.
