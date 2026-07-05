"""Language filtering + deterministic ratio balancing.

Detection `corpus/languages.py` me hai; ye module usse filtering/balancing policy
banata hai. Ratio balancing deterministic hai (record hash se ordering), koi
wall-clock / RNG nahi.
"""

from __future__ import annotations

import hashlib

from ..languages import detect_language, is_priority


def _rank(record) -> str:
    """Stable per-record key for deterministic selection."""
    return record.hash or hashlib.sha256(
        (record.repository + "/" + record.path).encode()).hexdigest()


def keep_language(record, allowed=None) -> bool:
    lang = record.language or "unknown"
    if allowed is not None:
        return lang in set(allowed)
    return is_priority(lang)


def annotate_language(record) -> None:
    """Fill record.language in place from path + content."""
    if not record.language or record.language == "unknown":
        record.language = detect_language(record.path, record.content)


def balance_language_ratios(records: list, ratios: dict, *, drop_others=True) -> list:
    """Downsample `records` so per-language counts approximate `ratios`.

    Deterministic: within a language, records are ordered by hash and the first
    `target` kept. Returns the kept subset (order preserved from input).
    """
    langs = {k: v for k, v in ratios.items() if v > 0}
    by_lang: dict = {}
    for r in records:
        by_lang.setdefault(r.language, []).append(r)

    avail = {lang: len(by_lang.get(lang, [])) for lang in langs}
    if not any(avail.values()):
        return list(records)

    # Largest total N such that floor(ratio*N) <= available for every language.
    def feasible(N: int) -> bool:
        return all(int(langs[l] * N) <= avail[l] for l in langs)

    lo, hi = 0, sum(avail.values()) * 2 + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    N = lo

    target = {l: int(langs[l] * N) for l in langs}
    keep_ids = set()
    for lang, tgt in target.items():
        chosen = sorted(by_lang.get(lang, []), key=_rank)[:tgt]
        keep_ids.update(id(r) for r in chosen)

    out = []
    for r in records:
        if r.language in langs:
            if id(r) in keep_ids:
                out.append(r)
        elif not drop_others:
            out.append(r)
    return out
