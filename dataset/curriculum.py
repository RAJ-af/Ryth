"""Module 9 — Smart Curriculum Builder.

Difficulty ab sirf file size se nahi — multiple code-complexity signals combine
karke nikalti hai:

    AST depth • Cyclomatic complexity • Imports • Classes • Async usage • Size

Records ko easy -> medium -> hard order me arrange karta hai taaki model pehle
simple code seekhe, phir complex (jaise insaan seekhte hain). FIM-generated
records apne source ki difficulty inherit karte hain.
"""

from __future__ import annotations

from .record import FileRecord

_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def complexity_score(rec: FileRecord) -> float:
    """Weighted multi-signal complexity (Smart Curriculum ka core)."""
    m = rec.meta
    n_tokens = m.get("n_tokens", len(rec.token_ids))
    return (
        1.6 * m.get("ast_depth", 0) +
        0.8 * m.get("cyclomatic", 1) +
        0.5 * len(m.get("imports", [])) +
        2.0 * m.get("classes", 0) +
        1.2 * m.get("async", 0) +
        1.0 * m.get("functions", 0) +
        n_tokens / 600.0
    )


def assign_difficulty(rec: FileRecord) -> str:
    # FIM records apne source ki difficulty le kar aate hain — dobara compute mat karo
    if rec.meta.get("kind") == "fim" and rec.difficulty in _ORDER:
        return rec.difficulty
    c = complexity_score(rec)
    rec.meta["complexity"] = round(c, 2)
    if c < 8:
        rec.difficulty = "easy"
    elif c < 22:
        rec.difficulty = "medium"
    else:
        rec.difficulty = "hard"
    return rec.difficulty


class CurriculumBuilder:
    def order(self, records: list[FileRecord]) -> list[FileRecord]:
        for rec in records:
            assign_difficulty(rec)
        # easy -> hard; tie par higher quality pehle. Deterministic (stable sort).
        return sorted(records, key=lambda r: (_ORDER[r.difficulty], -r.quality,
                                              r.repo, r.path))
