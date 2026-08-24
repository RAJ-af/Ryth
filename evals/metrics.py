"""pass@k — Codex paper (Chen et al. 2021) ka unbiased estimator.

Factorials se overflow hota hai, isliye stable product form:
    pass@k = 1 - prod_{i=n-c+1..n} (1 - k/i)
Pure standard library.
"""

from __future__ import annotations


def pass_at_k(n: int, c: int, k: int) -> float:
    """n samples me c correct hon par k draws me kam-se-kam 1 correct hone ki probability."""
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    k = min(k, n)
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def aggregate(results: list[dict], ks=(1, 5, 10)) -> dict[str, float]:
    """Per-task pass@k ka macro-average. Item: {task_id, n, n_passed}."""
    out: dict[str, float] = {}
    for k in ks:
        vals = [pass_at_k(r["n"], r["n_passed"], k) for r in results]
        out[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out
