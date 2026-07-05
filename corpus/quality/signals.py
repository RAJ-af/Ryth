"""Quality signals for a repository (each normalized to 0..1).

Signals (spec): syntax validity, documentation, tests, project structure,
comments, complexity, maintainability, duplicate ratio. Sab kuch kept FileRecords
(content ke saath) se compute hota hai. Pure standard library, deterministic.
"""

from __future__ import annotations

_COMMENT_PREFIX = {
    "python": ("#",), "bash": ("#",), "yaml": ("#",), "sql": ("--",),
    "javascript": ("//",), "typescript": ("//",), "rust": ("//",),
    "go": ("//",), "cpp": ("//",), "java": ("//",), "css": ("/*",),
}
_CODE_LANGS = set(_COMMENT_PREFIX) | {"html"}
_STRUCTURE_MARKERS = (
    "readme", "license", "pyproject.toml", "setup.py", "package.json",
    "cargo.toml", "go.mod", "pom.xml", "build.gradle", "cmakelists.txt",
    "makefile", "dockerfile",
)


def _base(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1].lower()


def _is_test(path: str) -> bool:
    b, p = _base(path), path.lower()
    return (b.startswith("test_") or b.endswith("_test.py") or ".test." in b
            or ".spec." in b or "_test.go" in b or "/tests/" in p or "/test/" in p
            or b.endswith("test.java"))


def check_syntax(language: str, text: str) -> bool | None:
    """True/False if checkable, None if we can't judge this language."""
    if language == "python":
        try:
            compile(text, "<corpus>", "exec")
            return True
        except (SyntaxError, ValueError):
            return False
    if language in ("javascript", "typescript", "rust", "go", "cpp", "java",
                    "json", "css"):
        return _balanced_brackets(text)
    return None


def _balanced_brackets(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _comment_ratio(language: str, text: str) -> float:
    prefixes = _COMMENT_PREFIX.get(language)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    if not prefixes:
        return 0.0
    c = sum(1 for ln in lines if any(ln.startswith(p) for p in prefixes))
    return c / len(lines)


def _has_docstring(language: str, text: str) -> bool:
    if language == "python":
        return '"""' in text or "'''" in text
    return "/**" in text or _comment_ratio(language, text) > 0.02


def repo_signals(records: list) -> dict:
    """Compute the 0..1 signal dict for a repo from its kept FileRecords."""
    code = [r for r in records if r.language in _CODE_LANGS and r.content]
    paths = [r.path for r in records]

    # --- syntax validity ---
    checks = [check_syntax(r.language, r.content) for r in code]
    checks = [c for c in checks if c is not None]
    syntax = (sum(checks) / len(checks)) if checks else 0.7

    # --- documentation ---
    readme = 1.0 if any(_base(p).startswith("readme") for p in paths) else 0.0
    doc_ratio = (sum(_has_docstring(r.language, r.content) for r in code) / len(code)
                 if code else 0.0)
    documentation = 0.5 * readme + 0.5 * doc_ratio

    # --- tests ---
    n_tests = sum(1 for p in paths if _is_test(p))
    tests = min(1.0, (n_tests / max(1, len(code))) * 3.0) if n_tests else 0.0

    # --- structure ---
    have = sum(1 for m in _STRUCTURE_MARKERS
               if any(_base(p) == m or _base(p).startswith(m) for p in paths))
    has_src = any(("/src/" in p or "/lib/" in p or p.startswith("src/")
                   or p.startswith("lib/")) for p in paths)
    structure = min(1.0, have / 4.0 + (0.2 if has_src else 0.0))

    # --- comments ---
    cratios = [_comment_ratio(r.language, r.content) for r in code]
    avg_c = sum(cratios) / len(cratios) if cratios else 0.0
    comments = min(1.0, avg_c / 0.20)

    # --- complexity (function-length health) ---
    healths = []
    for r in code:
        lines = [ln for ln in r.content.split("\n") if ln.strip()]
        n_def = r.content.count("def ") + r.content.count("function ") \
            + r.content.count("fn ")
        avg_len = len(lines) / max(1, n_def)
        healths.append(1.0 if avg_len <= 40 else max(0.2, 40.0 / avg_len))
    complexity = sum(healths) / len(healths) if healths else 0.7

    # --- maintainability ---
    if code:
        good_lines, total_lines = 0, 0
        for r in code:
            for ln in r.content.split("\n"):
                total_lines += 1
                if len(ln) <= 120:
                    good_lines += 1
        line_health = good_lines / max(1, total_lines)
        size_health = sum(1 for r in code if r.size < 200_000) / len(code)
        maintainability = 0.5 * line_health + 0.5 * size_health
    else:
        maintainability = 0.5

    # --- duplicate ratio (cleanliness) ---
    hashes = [r.hash for r in records if r.hash]
    if hashes:
        dup = 1.0 - len(set(hashes)) / len(hashes)
    else:
        dup = 0.0
    duplicate_ratio = 1.0 - dup

    return {
        "syntax_validity": round(syntax, 4),
        "documentation": round(documentation, 4),
        "tests": round(tests, 4),
        "structure": round(structure, 4),
        "comments": round(comments, 4),
        "complexity": round(complexity, 4),
        "maintainability": round(maintainability, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
    }
