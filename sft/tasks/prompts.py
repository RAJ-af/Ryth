"""Prompt templates — TEACHER instructions vs DATASET turns alag hain.

Design ruling: stored user turn waisa hi hai jo Ryth ka REAL user poochta;
teacher-directive sirf teacher call me bracketed appendix hota hai taaki
meta-text dataset ke messages me kabhi leak na ho.
"""

from __future__ import annotations

KNOWN_TASKS = ("instruction_to_code", "bug_fix", "docstring_to_code",
               "explain_code", "test_gen")

# STORED hota hai har example me — yahi Ryth ki persona banega
SYSTEM_PROMPT = "You are Ryth, a concise and correct coding assistant."

# SIRF teacher ko jaata hai — dataset me KABHI nahi jaata
TEACHER_SYSTEM = ("You are an expert engineer generating high-quality SFT "
                  "training data. Follow the format directive exactly; "
                  "output ONLY the requested content.")

_DIRECTIVES = {
    "instruction_to_code":
        "Respond with a single complete Python function. No prose, no fences.",
    "bug_fix":
        "Respond with the corrected full function only. No prose, no fences.",
    "docstring_to_code":
        "Respond with the function BODY (correctly indented), no prose.",
    "explain_code":
        "Explain in 3-6 short sentences what the code does and why.",
    "test_gen":
        "Respond with Python assert statements that test the function. "
        "One per line, no imports, no fences.",
}


def directive_for(task: str) -> str:
    try:
        return _DIRECTIVES[task]
    except KeyError:
        raise ValueError(f"unknown sft task {task!r}; "
                         f"known: {sorted(_DIRECTIVES)}") from None


def teacher_user(user_prompt: str, task: str) -> str:
    """Stored-turn + directive — SIRF teacher.complete() me jaata hai."""
    return f"{user_prompt}\n\n[{directive_for(task)}]"
