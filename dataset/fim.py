"""Module 4 (v1.1) — FIM Builder (Fill-In-the-Middle).

Coding models ke liye bahut valuable: model ko sikhata hai ki code ke *beech* me
missing part predict kare (jaise editor autocomplete/infill). Har Python function
ko todkar (prefix / middle / suffix) PSM-format training text banate hain:

    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}

Deterministic hai (path-derived seed) taaki dataset replay ho sake.
"""

from __future__ import annotations

import ast
import hashlib
import random

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def _path_seed(path: str, base_seed: int) -> int:
    """Path se ek deterministic seed (Python ke randomized hash() se bacho)."""
    h = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed ^ h) & 0xFFFFFFFF


def extract_functions(text: str) -> list[str]:
    """AST se har function/async-function ka source segment nikaalo."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    segs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(text, node)
            if seg:
                segs.append(seg)
    return segs


def make_fim(code: str, rng: random.Random) -> str | None:
    """Ek code snippet ko PSM FIM example me badlo (random split, seeded)."""
    n = len(code)
    if n < 40:
        return None
    a = rng.randint(int(n * 0.20), int(n * 0.50))          # prefix end
    b = rng.randint(a + 1, min(n, a + int(n * 0.40) + 1))  # middle end
    prefix, middle, suffix = code[:a], code[a:b], code[b:]
    if not middle.strip():
        return None
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


class FIMBuilder:
    def __init__(self, config):
        self.cfg = config

    def build_for(self, rec) -> list[str]:
        """Ek FileRecord se FIM texts ki list (khali agar python nahi / disabled)."""
        if not getattr(self.cfg, "fim_enabled", True):
            return []
        if rec.language != "python":
            return []
        rng = random.Random(_path_seed(rec.path, self.cfg.seed))
        out = []
        for seg in extract_functions(rec.text):
            if len(seg) < self.cfg.fim_min_chars:
                continue
            if rng.random() > self.cfg.fim_rate:
                continue
            fim = make_fim(seg, rng)
            if fim:
                out.append(fim)
        return out
