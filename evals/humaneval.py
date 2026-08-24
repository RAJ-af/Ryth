"""HumanEval-style pass@k runner.

Har problem ke liye n samples generate karo, phir HAR sample ko alag
subprocess me `prompt + completion + test + check(entry_point)` ke saath
chalao. Sampler dependency-injected hai — tests bina model ke chalte hain.

⚠️ SECURITY: generated code LOCAL subprocess me EXECUTE hota hai (timeout ke
alawa koi jail nahi) — trusted machine pe hi chalao (see evals/execution.py).
"""

from __future__ import annotations

import json

from .datasets import Problem
from .execution import run_program
from .generation import extract_code
from .metrics import aggregate


def _build_program(p: Problem, completion: str) -> str:
    code = extract_code(completion)
    return f"{p.prompt}{code}\n{p.test}\ncheck({p.entry_point})\n"


def evaluate(problems: list[Problem], *, sampler=None, model=None, tok=None,
             n_samples: int = 20, mode: str = "base", temperature: float = 0.8,
             top_k: int | None = 40, max_new_tokens: int = 256,
             timeout_s: float = 10.0, seed: int = 1234, ks=(1, 5, 10),
             progress=print) -> dict:
    """pass@k eval — sampler YA (model+tok) dono me se ek zaroori hai."""
    if sampler is None:
        if model is None or tok is None:
            raise ValueError("sampler YA (model+tok) dono me se ek do")
        import torch
        torch.manual_seed(seed)                      # reproducible sampling
        from .generation import sample_completion

        def sampler(prompt: str) -> str:
            return sample_completion(model, tok, prompt, mode=mode,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k)

    tasks = []
    for p in problems:
        n_passed = 0
        samples_ok = []
        for _ in range(n_samples):
            completion = sampler(p.prompt)
            program = _build_program(p, completion)
            r = run_program(program, timeout_s=timeout_s)
            samples_ok.append(bool(r.ok))
            n_passed += int(r.ok)
        tasks.append({"task_id": p.task_id, "n": n_samples,
                      "n_passed": n_passed, "samples_ok": samples_ok})
        progress(f"[humaneval] {p.task_id}: {n_passed}/{n_samples}")
    res = {"meta": {"mode": mode, "n_samples": n_samples,
                    "temperature": temperature, "top_k": top_k,
                    "max_new_tokens": max_new_tokens, "seed": seed,
                    "task": "humaneval"},
           "tasks": tasks,
           "pass_at_k": aggregate(tasks, ks=ks)}
    return res


def save_results(res: dict, out_path: str) -> None:
    """Pretty-JSON results file (runs ke beech compare karne ke liye)."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
