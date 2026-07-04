"""Module 3 — Quality Analyzer.

Har file ko 0–100 quality score deta hai. Signals:
  Readability • Complexity • Comments • Type Hints • Tests • README • License

Python files ke liye AST se detailed analysis; baaki languages ke liye lightweight
heuristics. Score se hi Module 9 (Curriculum) difficulty nikaalta hai.
"""

from __future__ import annotations

import ast

from .record import FileRecord


def _ast_depth(node, depth=0) -> int:
    """Maximum nesting depth of the AST (structural complexity signal)."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_ast_depth(c, depth + 1) for c in children)


def _python_signals(text: str) -> dict:
    sig = {"functions": 0, "classes": 0, "typed_args": 0, "total_args": 0,
           "branches": 0, "has_tests": False, "imports": [],
           "ast_depth": 0, "cyclomatic": 1, "async": 0}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return sig
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig["functions"] += 1
            for a in node.args.args:
                sig["total_args"] += 1
                if a.annotation is not None:
                    sig["typed_args"] += 1
            if node.name.startswith("test_"):
                sig["has_tests"] = True
            if isinstance(node, ast.AsyncFunctionDef):
                sig["async"] += 1
        elif isinstance(node, ast.ClassDef):
            sig["classes"] += 1
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                               ast.BoolOp, ast.ExceptHandler, ast.IfExp,
                               ast.comprehension)):
            sig["branches"] += 1                       # cyclomatic decision points
        elif isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            sig["async"] += 1
        elif isinstance(node, ast.Import):
            sig["imports"] += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            sig["imports"].append(node.module or "")
    sig["cyclomatic"] = sig["branches"] + 1            # McCabe-style approximation
    sig["ast_depth"] = _ast_depth(tree)
    if "assert" in text or "import pytest" in text or "unittest" in text:
        sig["has_tests"] = True
    return sig


class QualityAnalyzer:
    def __init__(self):
        pass

    def score(self, rec: FileRecord, repo_meta: dict | None = None) -> int:
        repo_meta = repo_meta or {}
        text = rec.text
        lines = text.splitlines() or [""]
        n_lines = len(lines)

        # --- Readability: line length + blank ratio ---
        avg_len = sum(len(ln) for ln in lines) / n_lines
        readability = 100
        if avg_len > 100:
            readability -= min(40, (avg_len - 100) * 0.5)
        long_lines = sum(1 for ln in lines if len(ln) > 120)
        readability -= min(30, long_lines / n_lines * 100)
        readability = max(0, readability)

        # --- Comments ratio ---
        comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
        comment_ratio = comment_lines / n_lines
        comments = min(100, comment_ratio * 500)     # ~20% comments => full
        rec.meta["comment_lines"] = comment_lines
        rec.meta["n_lines"] = n_lines

        if rec.language == "python":
            s = _python_signals(text)
            rec.meta["functions"] = s["functions"]
            rec.meta["classes"] = s["classes"]
            rec.meta["imports"] = sorted(set(s["imports"]))[:32]
            # Smart-curriculum signals (Module 9)
            rec.meta["ast_depth"] = s["ast_depth"]
            rec.meta["cyclomatic"] = s["cyclomatic"]
            rec.meta["async"] = s["async"]

            type_cov = (s["typed_args"] / s["total_args"]) if s["total_args"] else 0.0
            type_hints = type_cov * 100
            # complexity: branches-per-function (kam = simple = achha, ek limit tak)
            bpf = s["branches"] / max(1, s["functions"])
            complexity = max(0, 100 - abs(bpf - 3) * 12)   # ~3 sweet spot
            tests = 100 if s["has_tests"] else 0
        else:
            type_hints = 50            # non-python: neutral
            complexity = 70
            tests = 0

        readme = 100 if repo_meta.get("has_readme") else 0
        license_ = 100 if repo_meta.get("has_license") else 0

        # weighted blend
        score = (
            0.22 * readability +
            0.18 * complexity +
            0.12 * comments +
            0.15 * type_hints +
            0.13 * tests +
            0.10 * readme +
            0.10 * license_
        )
        rec.quality = int(round(max(0, min(100, score))))
        return rec.quality
