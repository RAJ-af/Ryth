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
    if rec.language != "python":
        return []
    out = []
    for fn in extract_python_functions(rec.content or ""):
        if not fn["docstring"] or len(fn["body"]) <= len(fn["docstring"]) + 20:
            continue
        stub = (f"{fn['header']}\n"
                f'    """{fn["docstring"]}"""\n'
                "    ...  # TODO: implement")
        user = (f"Implement this Python function:\n\n{stub}\n\n"
                "The docstring is the specification.")
        out.append(_seed(_sid("i2c", rec, fn["name"]),
                         "instruction_to_code", rec, user, _check_compiles()))
    return out


def bug_fix_seeds(rec) -> list:
    """Corpus ka hi deterministic operator-mutation; fix compile-checked."""
    if rec.language != "python":
        return []
    out = []
    for ex in bug_fixing_examples(rec):
        buggy = ex["input"].split("\n\n", 1)[1]
        user = ("This Python function has a subtle bug. Find and fix it."
                f"\n\n{buggy}")
        out.append(_seed(f"bug:{getattr(rec, 'hash', '')[:12]}",
                         "bug_fix", rec, user, _check_compiles()))
    return out


def docstring_to_code_seeds(rec) -> list:
    """Body completion — pretrain signal ka chat-format cousin."""
    if rec.language != "python":
        return []
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
    if rec.language != "python":
        return []
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
    if rec.language != "python":
        return []
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
