"""Profiler — torch.profiler ka lightweight wrapper.

Kuch steps profile karke ek summary table deta hai (kahan time ja raha hai).
Optional — trainer/benchmark isse use karte hain. CPU + CUDA dono support.
"""

from __future__ import annotations

import torch


def _activities():
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return acts


def profile_steps(step_fn, n: int = 5, row_limit: int = 12,
                  sort_by: str = "cpu_time_total") -> str:
    """`step_fn()` ko n baar chala ke profiler summary table (str) return karo."""
    with torch.profiler.profile(activities=_activities()) as prof:
        for _ in range(n):
            step_fn()
    return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
