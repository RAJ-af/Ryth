"""Generation loop — seeds x teacher -> filtered Examples + stats.

Offline dry-run: FakeTeacher inject karo (CLI --dry-run isi class ko use
karta hai) — bina network/key ke poora pipeline validate hota hai.
Failures pipeline ko nahi rokte — reason record hoke stats me jaata hai.
"""

from __future__ import annotations

from sft.filter import Deduper, FilterConfig, rule_check, self_verify
from sft.schema import Example
from sft.tasks.builders import build_seeds          # noqa: F401 (re-export)
from sft.tasks.prompts import SYSTEM_PROMPT, TEACHER_SYSTEM, teacher_user


def generate(seeds, teacher, *, cfg: FilterConfig | None = None,
             dedup: bool = True, verify_teacher=None, limit: int | None = None,
             progress=print) -> tuple:
    """Returns (examples, stats). Ek seed fail => reason logged, aage badho."""
    cfg = cfg or FilterConfig()
    dd = Deduper() if dedup else None
    examples: list[Example] = []
    reasons: dict[str, int] = {}
    per_task: dict[str, dict[str, int]] = {}
    n_gen = 0

    for s in seeds:
        if limit and n_gen >= limit:
            break
        pt = per_task.setdefault(s.task, {"generated": 0, "passed": 0})
        pt["generated"] += 1
        n_gen += 1
        raw = teacher.complete(TEACHER_SYSTEM,
                               teacher_user(s.user_prompt, s.task))
        bad = rule_check(s, raw, cfg)
        if not bad and dd is not None and dd.duplicate(raw):
            bad = ["duplicate"]
        if not bad and verify_teacher is not None:
            bad = self_verify(s, raw, verify_teacher)
        if bad:
            for r in bad:
                reasons[r] = reasons.get(r, 0) + 1
            continue
        pt["passed"] += 1
        examples.append(Example(
            task=s.task,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": s.user_prompt},
                      {"role": "assistant", "content": raw}],
            meta={"source_path": s.source_path, "language": s.language}))
        progress(f"[sft] {s.task} ok ({len(examples)})")

    stats = {"n_seeds": len(seeds), "n_generated": n_gen,
             "n_passed": len(examples),
             "pass_rate": round(len(examples) / n_gen, 4) if n_gen else 0.0,
             "per_task": per_task,
             "filter_reasons": dict(sorted(reasons.items(),
                                           key=lambda kv: -kv[1]))}
    return examples, stats


def package(examples, tok=None) -> list[dict]:
    """Example list -> JSONL-ready rows (tok diye to token_ids bhi)."""
    return [e.to_row(tok) for e in examples]
