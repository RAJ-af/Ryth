"""Module 4 — Repository Analyzer.

Repo-level info store karta hai (future me repository-aware training ke liye):
  Repository • Stars • License • Branch • Commit • Folder Structure • README/License flags
"""

from __future__ import annotations

import os


_LICENSE_HINTS = ("mit license", "apache license", "bsd ", "mozilla public",
                  "gnu general public", "isc license", "the unlicense")


class RepositoryInfo:
    def __init__(self, name: str, root: str):
        self.name = name
        self.root = root
        self.stars = 0
        self.license = "unknown"
        self.branch = ""
        self.commit = ""
        self.has_readme = False
        self.has_license = False
        self.folders: list[str] = []

    def to_meta(self) -> dict:
        return {
            "repo": self.name, "stars": self.stars, "license": self.license,
            "branch": self.branch, "commit": self.commit,
            "has_readme": self.has_readme, "has_license": self.has_license,
        }


class RepositoryAnalyzer:
    def analyze(self, name: str, root: str) -> RepositoryInfo:
        info = RepositoryInfo(name, root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            if rel != ".":
                info.folders.append(rel)
            for fn in filenames:
                low = fn.lower()
                if low.startswith("readme"):
                    info.has_readme = True
                if low.startswith("license") or low.startswith("licence") \
                        or low == "copying":
                    info.has_license = True
                    info.license = self._sniff_license(os.path.join(dirpath, fn))
        return info

    def _sniff_license(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                head = f.read(800).lower()
        except Exception:
            return "unknown"
        for hint in _LICENSE_HINTS:
            if hint in head:
                return hint.strip().split(" license")[0].upper() or "unknown"
        return "custom"
