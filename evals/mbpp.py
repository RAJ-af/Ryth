"""MBPP-style runner — assert-based tests, koi check() wrapper nahi.

Real MBPP rows `text` + `test_list` laate hain; load_mbpp adapter unhe
canonical Problem (prompt/test) me badalta hai. Pehle se canonical rows
(prompt/test) bhi tolerate hote hain.
"""

from __future__ import annotations

import gzip
import json

from .datasets import Problem
from .generation import extract_code
from .execution import run_program
from .metrics import aggregate


def load_mbpp(path: str) -> list[Problem]:
    """MBPP rows (text/test_list) -> canonical Problem."""
    opener = gzip.open if path.endswith(".gz") else open
    problems = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" in d and "test" in d:            # already canonical
                problems.append(Problem(
                    task_id=str(d["task_id"]), prompt=d["prompt"], test=d["test"],
                    entry_point=d.get("entry_point", ""),
                    canonical_solution=d.get("canonical_solution", "")))
                continue
            problems.append(Problem(
                task_id=str(d.get("task_id") or d.get("code", "")),
                prompt=d["text"],
                test="\n".join(d["test_list"]),
                canonical_solution=d.get("canonical_solution",
                                         d.get("code", ""))))
    return problems


def evaluate(problems, *, sampler=None, model=None, tok=None,
             n_samples: int = 20, mode: str = "base", temperature: float = 0.8,
             top_k: int | None = 40, max_new_tokens: int = 256,
             timeout_s: float = 10.0, seed: int = 1234, ks=(1, 5, 10),
             progress=print) -> dict:
    """Same contract as evals.humaneval.evaluate, par program = code + asserts."""
    if sampler is None:
        if model is None or tok is None:
            raise ValueError("sampler YA (model+tok) dono me se ek do")
        import torch
        torch.manual_seed(seed)
        from .generation import sample_completion

        def sampler(prompt: str) -> str:
            return sample_completion(model, tok, prompt, mode=mode,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k)

    tasks = []
    for p in problems:
        n_passed = 0
        for _ in range(n_samples):
            completion = sampler(p.prompt)
            program = f"{extract_code(completion)}\n{p.test}\n"
            n_passed += int(run_program(program, timeout_s=timeout_s).ok)
        tasks.append({"task_id": p.task_id, "n": n_samples,
                      "n_passed": n_passed})
        progress(f"[mbpp] {p.task_id}: {n_passed}/{n_samples}")
    return {"meta": {"task": "mbpp", "mode": mode, "n_samples": n_samples,
                     "temperature": temperature, "top_k": top_k,
                     "max_new_tokens": max_new_tokens, "seed": seed},
            "tasks": tasks, "pass_at_k": aggregate(tasks, ks=ks)}
