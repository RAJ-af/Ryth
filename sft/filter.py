"""Quality gate — rule filters + dedup (+ optional teacher self-verification).

Spec §6 acceptance: v1 dataset me >=90% examples rule filters pass karein.
Rule reasons human-readable hain — stats me aggregate hoke dikhte hain.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


@dataclass
class FilterConfig:
    min_user_chars: int = 20
    min_assistant_chars: int = 20
    max_total_chars: int = 8000


def normalize_for_dedup(text: str) -> str:
    """Whitespace-normalized dedup key — CASE-PRESERVING jaan-boojh ke.

    Code me `Add` aur `add` alag identifiers hain; lower() karne par
    legitimately alag solutions galti se merge ho jaate the (review fix).
    """
    return _WS_RE.sub(" ", text).strip()


class Deduper:
    """First-wins dedup — normalized assistant text ka sha256."""

    def __init__(self):
        self.seen: set = set()

    def duplicate(self, text: str) -> bool:
        h = hashlib.sha256(normalize_for_dedup(text).encode()).hexdigest()
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


def rule_check(seed, assistant_text: str,
               cfg: FilterConfig | None = None) -> list[str]:
    """Per-task validator PEHLE, phir generic length rules."""
    cfg = cfg or FilterConfig()
    reasons = list(seed.validate(assistant_text))
    if len(seed.user_prompt) < cfg.min_user_chars:
        reasons.append(f"user prompt < {cfg.min_user_chars} chars")
    if len(assistant_text.strip()) < cfg.min_assistant_chars:
        reasons.append(f"assistant < {cfg.min_assistant_chars} chars")
    if len(seed.user_prompt) + len(assistant_text) > cfg.max_total_chars:
        reasons.append(f"total > {cfg.max_total_chars} chars")
    return reasons


def self_verify(seed, assistant_text: str, teacher) -> list[str]:
    """OPTIONAL second pass: teacher khud reviewer banke yes/no bole."""
    q = (f"Task given to an engineer:\n\n{seed.user_prompt}\n\n"
         f"Their answer:\n\n{assistant_text}\n\n"
         "Is the answer correct and complete? Reply with exactly yes or no.")
    reply = teacher.complete(
        "You are a strict code reviewer. Reply with exactly one word: "
        "yes or no.", q, max_tokens=8, temperature=0.0).strip().lower()
    if reply.startswith("yes"):
        return []
    return [f"self_verify said: {reply[:40]}"]
