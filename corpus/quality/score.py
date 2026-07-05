"""Aggregate quality signals into a single 0..100 score.

Weights configurable hain. Default weights spec ke signals ko balance karte hain.
`score_repo` ek repo ke records se (score, signals) dono deta hai.
"""

from __future__ import annotations

from .signals import repo_signals

DEFAULT_WEIGHTS = {
    "syntax_validity": 0.20,
    "documentation": 0.15,
    "tests": 0.15,
    "structure": 0.12,
    "comments": 0.10,
    "complexity": 0.12,
    "maintainability": 0.10,
    "duplicate_ratio": 0.06,
}


def quality_score(signals: dict, weights: dict | None = None) -> float:
    """Weighted 0..100 score from a signal dict."""
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.get(k, 0.0) for k in signals)
    if total_w <= 0:
        return 0.0
    s = sum(signals.get(k, 0.0) * w.get(k, 0.0) for k in signals)
    return round(100.0 * s / total_w, 2)


def score_repo(records: list, weights: dict | None = None) -> tuple:
    """Return (score_0_100, signals_dict) for a repo's records."""
    signals = repo_signals(records)
    return quality_score(signals, weights), signals
