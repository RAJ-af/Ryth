"""Module 2 — Validator.

Ye checks karta hai (aur reason ke saath flag karta hai):
  Syntax • Encoding • Language • File Type • File Size • Extension

Python file agar compile hi nahi hoti -> `syntax` flag. (Encoding pehle Cleaner
me ho chuka, yahan double-safe.)
"""

from __future__ import annotations

import os

from .config import RDEConfig
from .record import FileRecord


class Validator:
    def __init__(self, config: RDEConfig):
        self.cfg = config
        self.stats = {"ok": 0, "size": 0, "extension": 0, "language": 0,
                      "syntax": 0, "encoding": 0}

    def validate(self, rec: FileRecord) -> bool:
        # size
        n = len(rec.text.encode("utf-8"))
        if n < self.cfg.min_file_bytes or n > self.cfg.max_file_bytes:
            rec.drop("size"); self.stats["size"] += 1; return False

        # extension allow-list (dockerfile ka koi ext nahi -> language se pass)
        _, ext = os.path.splitext(rec.path.lower())
        if ext and ext not in self.cfg.allowed_ext:
            rec.drop("extension"); self.stats["extension"] += 1; return False

        # language known hona chahiye
        if rec.language == "unknown":
            rec.drop("language"); self.stats["language"] += 1; return False

        # encoding (re-check)
        try:
            rec.text.encode("utf-8")
        except UnicodeEncodeError:
            rec.drop("encoding"); self.stats["encoding"] += 1; return False

        # syntax (sirf python)
        if rec.language == "python":
            try:
                compile(rec.text, rec.path, "exec")
            except SyntaxError:
                rec.drop("syntax"); self.stats["syntax"] += 1; return False

        self.stats["ok"] += 1
        return True
