"""Curriculum learning — RDE difficulty integration.

RDE har chunk ke saath difficulty metadata (easy/medium/hard) store karta hai
(Smart Curriculum, dataset engine me). Yahan hum us metadata ko use karke training
order easy -> medium -> hard banate hain, taaki model gradually seekhe.
"""

from __future__ import annotations

_RANK = {"easy": 0, "medium": 1, "hard": 2}


def difficulty_of(rds, idx: int) -> str:
    return rds.meta(idx).get("difficulty", "medium")


def curriculum_order(rds, indices=None) -> list[int]:
    """Indices ko easy -> medium -> hard order me arrange karo (stable)."""
    if indices is None:
        indices = list(range(len(rds)))
    return sorted(indices, key=lambda i: _RANK.get(difficulty_of(rds, i), 1))


def difficulty_histogram(rds, indices=None) -> dict:
    """Kitne chunks har difficulty me hain (reporting ke liye)."""
    if indices is None:
        indices = list(range(len(rds)))
    hist = {"easy": 0, "medium": 0, "hard": 0}
    for i in indices:
        d = difficulty_of(rds, i)
        hist[d] = hist.get(d, 0) + 1
    return hist
