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


# --------------------------------------------------------------------- #
# tasks/builders — corpus records se 5 seed types
# --------------------------------------------------------------------- #

from sft.tasks.builders import ALL_BUILDERS, build_seeds

_FUNC_SRC = '''def add_two(a, b):
    """Return the sum of two numbers, supporting ints and floats."""
    return a + b


def shout(s):
    """Uppercase the string and append an exclamation mark for excitement."""
    return s.upper() + "!"
'''


def _rec(content=_FUNC_SRC, language="python", path="mod/math.py",
         h="deadbeef01"):
    # NOTE: corpus.tasks.builders._example() ko repository/split/license bhi
    # mangta hai — full record duck-type yahi hai (ledger ruling).
    return types.SimpleNamespace(content=content, language=language,
                                 path=path, hash=h, repository="r", split="train",
                                 license="MIT")


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
    b = [s.id for s in build_seeds([_rec(h="bb"), _rec(h="aa")])]
    assert a == b and len(a) == len(set(a))         # sorted + unique
    assert build_seeds([_rec(content=""), _rec(language="c")]) == []


def test_test_gen_validator_runs_asserts_locally():
    seeds = build_seeds([_rec()])
    tst = next(s for s in seeds if s.task == "test_gen")
    i2c = next(s for s in seeds if s.task == "instruction_to_code")
    assert tst.validate("assert add_two(1, 2) == 3\nassert add_two(0, 0) == 0") == []
    probs = tst.validate("assert add_two(1, 2) == 99")
    assert probs and "fail" in probs[0]
    assert i2c.validate("def add_two(a, b):\n    return a + b") == []
    assert i2c.validate("def broken(:\n")
