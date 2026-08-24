"""Secret / API-key detection + redaction.

Regex-based detectors for common credential shapes: AWS keys, Google/GitHub/
Slack tokens, private-key blocks, Bearer headers, aur generic `key = "value"`
assignments. Redaction se cleaned corpus me live secrets nahi jaate.
Pure standard library.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Har pattern: (kind, regex, value_group)
#   value_group = None -> poora match redact hota hai
#   value_group = int  -> sirf us capture-group ka span redact hota hai
#                         (assignment me naam bachta hai, sirf value jaata hai
#                         — training data zyada useful rehta hai)
_PATTERNS: list[tuple[str, re.Pattern, int | None]] = [
    # --- well-known token shapes ---
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), None),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), None),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), None),
    ("private_key_block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"), None),
    ("bearer_header", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"), None),

    # --- assignment-style ---
    ("aws_secret_assignment", re.compile(
        r"\baws[_\-]?secret[_\-]?access[_\-]?key\b\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9/+=]{40}", re.IGNORECASE), None),
    ("credential_assignment", re.compile(
        r"\b(api[_\-]?key|apikey|access[_\-]?token|auth[_\-]?token|secret|"
        r"client[_\-]?secret|password|passwd|pwd|token|credentials?)\b"
        r"\s*[:=]\s*([\"'])([^\"']{8,})\2", re.IGNORECASE), 3),
]


def _find_spans(text: str) -> list[tuple[int, int, str]]:
    """Saare secret spans collect karo, phir overlapping merges resolve karo."""
    raw: list[tuple[int, int, str]] = []
    for kind, rx, grp in _PATTERNS:
        for m in rx.finditer(text):
            if grp is not None:
                raw.append((m.start(grp), m.end(grp), kind))
            else:
                raw.append((m.span()[0], m.span()[1], kind))
    if not raw:
        return []
    raw.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in raw:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:                 # overlap: lamba span jeete
                merged[-1] = (merged[-1][0], end, merged[-1][2])
            continue
        merged.append((start, end, kind))
    return merged


def find_secrets(text: str) -> list[tuple[str, str]]:
    """Detected secrets ki (kind, snippet) list — khali list matlab clean."""
    return [(kind, text[s:e]) for s, e, kind in _find_spans(text)]


def has_secret(text: str) -> bool:
    """Kya text me koi detectable secret hai?"""
    return bool(_find_spans(text))


def redact_secrets(text: str) -> tuple[str, int]:
    """Secret values ko REDACTED se replace karo. Returns (text, count).

    Overlaps pehle merge ho jaate hain, isliye count = distinct spans."""
    spans = _find_spans(text)
    if not spans:
        return text, 0
    out = text
    for start, end, _kind in reversed(spans):
        out = out[:start] + REDACTED + out[end:]
    return out, len(spans)
