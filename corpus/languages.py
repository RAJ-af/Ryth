"""Language detection by extension + filename + light content heuristics.

Corpus ke priority languages ka mapping. RDE ke language detector se alag hai
(usse chhedna mana hai) — ye corpus-specific hai aur zyada languages cover karta.
"""

from __future__ import annotations

# Priority coding languages (spec order).
LANGUAGES = (
    "python", "javascript", "typescript", "rust", "go", "cpp", "java",
    "bash", "sql", "html", "css", "json", "yaml", "dockerfile", "markdown",
)

_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".h": "cpp", ".c": "cpp",
    ".java": "java",
    ".sh": "bash", ".bash": "bash",
    ".sql": "sql",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "css", ".sass": "css",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".markdown": "markdown",
}

_FILENAMES = {
    "dockerfile": "dockerfile", "makefile": "bash", "pkgbuild": "bash",
    ".bashrc": "bash", ".zshrc": "bash",
}


def detect_language(path: str, content: str | None = None) -> str:
    """Best-effort language id for a path (+ optional content). 'unknown' if none."""
    base = path.rsplit("/", 1)[-1].lower()
    if base in _FILENAMES:
        return _FILENAMES[base]
    if base.startswith("dockerfile"):
        return "dockerfile"
    dot = base.rfind(".")
    if dot != -1:
        ext = base[dot:]
        if ext in _EXT:
            lang = _EXT[ext]
            # .h/.c can be C++ or C; treat as cpp bucket. shebang refines bash.
            return lang
    if content:
        head = content[:200].lstrip()
        if head.startswith("#!") and ("bash" in head[:40] or "/sh" in head[:40]):
            return "bash"
    return "unknown"


def is_priority(language: str) -> bool:
    return language in LANGUAGES
