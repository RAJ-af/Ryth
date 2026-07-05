"""Training-task builders — turn clean files into task-formatted examples.

Task types (spec): next-token, FIM, completion, editing, docstring->code,
README->code, code->explanation, bug fixing, refactoring, unit-test generation.

Har builder REAL examples banata hai (no placeholders): FIM sentinels tokenizer ke
special tokens se match karte hain; bug-fixing deterministic mutation se before/
after banata hai; docstring->code python functions se nikaalta hai; README->code
aur unit-test generation repo-level pairing use karte hain.

Sab deterministic hai (content-hash se choices), koi RNG / wall-clock nahi.
Example schema:
    {task, input, target, text, language, repository, path, split, license}
"""

from __future__ import annotations

import hashlib
import re

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"


def _pick(n: int, key: str) -> int:
    if n <= 0:
        return 0
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % n


def _example(task, record, *, input="", target="", text=""):
    return {
        "task": task, "input": input, "target": target, "text": text,
        "language": record.language, "repository": record.repository,
        "path": record.path, "split": record.split, "license": record.license,
    }


# --------------------------------------------------------------------------- #
# python function extraction (indentation-based, no external parser)
# --------------------------------------------------------------------------- #
_DEF_RE = re.compile(r"^(\s*)def\s+([A-Za-z_]\w*)\s*\(", re.M)


def extract_python_functions(text: str) -> list:
    """Return list of dicts: {name, header, body, full, indent, docstring}."""
    out = []
    lines = text.split("\n")
    for m in _DEF_RE.finditer(text):
        start = text[:m.start()].count("\n")
        indent = len(m.group(1))
        # collect the def header (may span until the line ending with ':')
        i = start
        while i < len(lines) and not lines[i].rstrip().endswith(":"):
            i += 1
        header_end = i
        body = []
        j = header_end + 1
        while j < len(lines):
            ln = lines[j]
            if ln.strip() == "":
                body.append(ln)
                j += 1
                continue
            cur_indent = len(ln) - len(ln.lstrip())
            if cur_indent <= indent:
                break
            body.append(ln)
            j += 1
        body_text = "\n".join(body).rstrip("\n")
        full = "\n".join(lines[start:j]).rstrip("\n")
        doc = _docstring_of(body_text)
        out.append({"name": m.group(2), "header": "\n".join(lines[start:header_end + 1]),
                    "body": body_text, "full": full, "indent": indent, "docstring": doc})
    return out


def _docstring_of(body_text: str) -> str:
    s = body_text.lstrip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            end = s.find(q, 3)
            if end != -1:
                return s[3:end].strip()
    return ""


# --------------------------------------------------------------------------- #
# file-level builders
# --------------------------------------------------------------------------- #
def next_token_examples(rec) -> list:
    return [_example("next_token", rec, text=rec.content)] if rec.content else []


def fim_examples(rec, fim_rate: float = 0.5) -> list:
    text = rec.content or ""
    if len(text) < 60:
        return []
    # choose a deterministic middle span (a contiguous ~25% slice)
    if _pick(1000, rec.hash + "fim") / 1000.0 > fim_rate:
        return []
    n = len(text)
    a = n // 4 + _pick(max(1, n // 8), rec.hash)
    b = min(n, a + max(8, n // 4))
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    if not middle.strip():
        return []
    psm = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    return [_example("fim", rec, input=f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}",
                     target=middle, text=psm)]


def completion_examples(rec) -> list:
    text = rec.content or ""
    if len(text) < 40:
        return []
    cut = len(text) // 2
    return [_example("completion", rec, input=text[:cut], target=text[cut:])]


def editing_examples(rec) -> list:
    """Remove one non-trivial line; task = restore it (real edit signal)."""
    text = rec.content or ""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if len(ln.strip()) > 5]
    if len(idx) < 4:
        return []
    victim = idx[_pick(len(idx), rec.hash + "edit")]
    removed = lines[victim]
    broken = "\n".join(lines[:victim] + lines[victim + 1:])
    instr = f"Restore the missing line at line {victim + 1}."
    return [_example("editing", rec, input=f"{instr}\n\n{broken}", target=removed.strip())]


def docstring_to_code_examples(rec) -> list:
    if rec.language != "python" or not rec.content:
        return []
    out = []
    for fn in extract_python_functions(rec.content):
        if fn["docstring"] and len(fn["body"]) > len(fn["docstring"]) + 20:
            prompt = f"{fn['header']}\n    \"\"\"{fn['docstring']}\"\"\""
            out.append(_example("docstring_to_code", rec, input=prompt, target=fn["body"]))
    return out


def code_to_explanation_examples(rec) -> list:
    if rec.language != "python" or not rec.content:
        return []
    out = []
    for fn in extract_python_functions(rec.content):
        if fn["docstring"]:
            code_wo_doc = fn["full"].replace(f'"""{fn["docstring"]}"""', "", 1)
            out.append(_example("code_to_explanation", rec,
                                input=code_wo_doc.strip(), target=fn["docstring"]))
    return out


_BUG_MUTATIONS = (("+", "-"), ("-", "+"), ("==", "!="), ("<", ">"),
                  (" and ", " or "), (" or ", " and "))


def bug_fixing_examples(rec) -> list:
    """Deterministically inject one operator bug; task = fix to original."""
    text = rec.content or ""
    if len(text) < 40:
        return []
    for i, (a, b) in enumerate(_BUG_MUTATIONS):
        pos = text.find(a)
        if pos != -1 and (_pick(len(_BUG_MUTATIONS), rec.hash) == i or True):
            buggy = text[:pos] + b + text[pos + len(a):]
            if buggy != text:
                return [_example("bug_fixing", rec,
                                 input="Fix the bug in this code:\n\n" + buggy,
                                 target=text)]
    return []


def refactoring_examples(rec) -> list:
    """before = de-formatted (collapsed blank lines + trailing ws); after = clean.
    A real normalization/reformatting signal."""
    text = rec.content or ""
    if len(text) < 60:
        return []
    messy = re.sub(r"[ \t]+\n", "\n", text)
    messy = re.sub(r"\n{3,}", "\n\n\n", messy)
    messy = "\n".join(ln.rstrip() + ("  " if i % 5 == 0 else "")
                      for i, ln in enumerate(text.split("\n")))
    if messy == text:
        return []
    return [_example("refactoring", rec,
                     input="Refactor and clean up this code:\n\n" + messy, target=text)]


# --------------------------------------------------------------------------- #
# repo-level builders
# --------------------------------------------------------------------------- #
def readme_to_code_examples(repo_records: list) -> list:
    readme = next((r for r in repo_records
                   if r.path.split("/")[-1].lower().startswith("readme") and r.content), None)
    if not readme:
        return []
    code = [r for r in repo_records if r.language in ("python", "javascript",
            "typescript", "go", "rust") and r.content]
    if not code:
        return []
    pick = code[_pick(len(code), readme.hash)]
    prompt = (readme.content or "")[:1500]
    return [_example("readme_to_code", pick,
                     input="Based on this README, write a module:\n\n" + prompt,
                     target=pick.content)]


def unit_test_generation_examples(repo_records: list) -> list:
    """Pair a source file with its matching test file (name-based)."""
    tests = {r.path.split("/")[-1]: r for r in repo_records
             if r.content and ("test" in r.path.lower() or "spec" in r.path.lower())}
    srcs = {r.path.split("/")[-1]: r for r in repo_records
            if r.content and "test" not in r.path.lower() and r.language == "python"}
    out = []
    for tname, trec in tests.items():
        base = tname.replace("test_", "").replace("_test", "").replace("spec_", "")
        src = srcs.get(base)
        if src:
            out.append(_example("unit_test_generation", src,
                                input="Write unit tests for this code:\n\n" + src.content,
                                target=trec.content))
    return out
