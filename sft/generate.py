"""Generation loop — seeds x teacher -> filtered Examples + stats.

Offline dry-run: FakeTeacher inject karo (CLI --dry-run isi class ko use
karta hai) — bina network/key ke poora pipeline validate hota hai.
Failures pipeline ko nahi rokte — reason record hoke stats me jaata hai.
Teacher-call failures bhi soft hain (teacher_error reason); `target_passed`
diye to jitne examples pass ho utne par ruk jaate hain (API spend cap).
"""

from __future__ import annotations

from sft.filter import Deduper, FilterConfig, rule_check, self_verify
from sft.schema import Example
from sft.tasks.builders import build_seeds          # noqa: F401 (re-export)
from sft.tasks.prompts import SYSTEM_PROMPT, TEACHER_SYSTEM, teacher_user
from sft.teacher import TeacherConfigError, TeacherError


def generate(seeds, teacher, *, cfg: FilterConfig | None = None,
             dedup: bool = True, verify_teacher=None, limit: int | None = None,
             target_passed: int | None = None, on_example=None,
             progress=print) -> tuple:
    """Returns (examples, stats). Ek seed fail => reason logged, aage badho.

    target_passed: n_passed is tak pahunchte hi ruk jao (CLI --target isi
    ko use karta hai — warna poora seed-pool jal jaata tha).
    on_example: har PASS ke turant baad callback(example) — CLI incremental
    JSONL write ke liye; crash par bhi likha hua bach jaata hai.
    """
    cfg = cfg or FilterConfig()
    dd = Deduper() if dedup else None
    examples: list[Example] = []
    reasons: dict[str, int] = {}
    per_task: dict[str, dict[str, int]] = {}
    n_gen = 0

    def _fail(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for s in seeds:
        if limit and n_gen >= limit:
            break
        if target_passed and len(examples) >= target_passed:
            break
        pt = per_task.setdefault(s.task, {"generated": 0, "passed": 0})
        pt["generated"] += 1
        n_gen += 1
        try:
            raw = teacher.complete(TEACHER_SYSTEM,
                                   teacher_user(s.user_prompt, s.task))
        except TeacherConfigError:
            raise                                  # setup galat — LOUD raho
        except TeacherError as e:
            _fail("teacher_error")                 # transient/network — soft
            progress(f"[sft] {s.task} teacher fail: {str(e)[:80]}")
            continue
        bad = rule_check(s, raw, cfg)
        if not bad and dd is not None and dd.duplicate(raw):
            bad = ["duplicate"]
        if not bad and verify_teacher is not None:
            try:
                bad = self_verify(s, raw, verify_teacher)
            except TeacherConfigError:
                raise
            except TeacherError as e:
                _fail("teacher_error")
                progress(f"[sft] {s.task} verify fail: {str(e)[:80]}")
                continue
        if bad:
            for r in bad:
                _fail(r)
            continue
        pt["passed"] += 1
        ex = Example(
            task=s.task,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": s.user_prompt},
                      {"role": "assistant", "content": raw}],
            meta={"source_path": s.source_path, "language": s.language})
        examples.append(ex)
        if on_example is not None:
            on_example(ex)
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
