"""Repository quality scoring (0..100) from structural + content signals."""

from .score import DEFAULT_WEIGHTS, quality_score, score_repo
from .signals import check_syntax, repo_signals

__all__ = [
    "repo_signals", "quality_score", "score_repo", "check_syntax",
    "DEFAULT_WEIGHTS",
]
