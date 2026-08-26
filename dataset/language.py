"""Module 5 — Language Detector.

Extension + content heuristics se file ka type detect karta hai.
W1-revision: 13 code languages (C/C++/Java/JS/TS/Rust/Go/Kotlin/C#/SQL/
HTML/CSS) + prose support — pack stage inhe drop na kare.
"""

from __future__ import annotations

import json
import os

_EXT_MAP = {
    ".py": "python", ".md": "markdown", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".sh": "bash", ".bash": "bash", ".txt": "text",
    ".cfg": "ini", ".ini": "ini",
    # W1 code languages
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go", ".kt": "kotlin", ".kts": "kotlin",
    ".cs": "csharp", ".sql": "sql",
    ".html": "html", ".htm": "html", ".css": "css",
}


def detect_language(path: str, text: str) -> str:
    name = os.path.basename(path).lower()
    _, ext = os.path.splitext(name)

    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    if ext in _EXT_MAP:
        lang = _EXT_MAP[ext]
        # JSON confirm: parse hoke dikhaye tabhi json
        if lang == "json":
            try:
                json.loads(text)
            except Exception:
                return "text"
        return lang

    # extension-less content sniff
    head = text.lstrip()[:200]
    if head.startswith("#!") and "python" in head:
        return "python"
    if head.startswith("#!") and ("bash" in head or "/sh" in head):
        return "bash"
    if head.upper().startswith("FROM ") and "\n" in text:
        return "dockerfile"
    return "unknown"
