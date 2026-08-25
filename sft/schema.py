"""SFT example schema — Seed (teacher request ka blueprint) + Example (row).

JSONL row shape:
    {"task": ..., "messages": [{role, content}, ...],
     "text": "<|system|>...", "token_ids": [...]?, "meta": {...}}
text/token_ids packaging-time pe render hote hain (W2 chat template).
token_ids OPTIONAL hai — provisional tokenizer bake karne se bacho; real
24k tokenizer (post-W1) aane par hi ids pack karo.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field


@dataclass
class Seed:
    """Ek teacher request: user turn + per-task validator closure."""

    id: str
    task: str
    language: str
    user_prompt: str               # STORED user turn (Ryth-user ki awaaz)
    teacher_directive: str = ""    # sirf teacher call me jaata hai
    check: object = None           # callable(assistant_text) -> list[str]
    source_path: str = ""

    def validate(self, assistant_text: str) -> list[str]:
        if self.check is None:
            return []
        return list(self.check(assistant_text) or [])


@dataclass
class Example:
    task: str
    messages: list                 # persona system + user + assistant
    meta: dict = field(default_factory=dict)

    def to_row(self, tok=None) -> dict:
        from evals.chat_template import register_chat_tokens, render

        if tok is not None:
            register_chat_tokens(tok)
            text = render(self.messages)
            return {"task": self.task, "messages": self.messages,
                    "text": text, "token_ids": tok.encode(text),
                    "meta": self.meta}
        return {"task": self.task, "messages": self.messages,
                "text": render(self.messages), "meta": self.meta}


def validate_example(row: dict) -> list[str]:
    """Loader-side structural QA (har committed dataset pe chalega)."""
    problems = []
    for key in ("task", "messages", "text"):
        if key not in row:
            problems.append(f"missing key {key!r}")
    msgs = row.get("messages") or []
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if len(roles) < 3:
        problems.append(f"need >=3 messages, got {len(roles)}")
    if roles and (roles[0] != "system" or roles[-1] != "assistant"):
        problems.append(f"bad role sequence {roles}")
    if "token_ids" in row and not (isinstance(row["token_ids"], list)
                                   and all(isinstance(i, int)
                                           for i in row["token_ids"])):
        problems.append("token_ids must be a list of ints")
    return problems


def write_jsonl(rows: list, path: str) -> int:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
