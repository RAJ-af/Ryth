"""Hugging Face downloader — materialize a HF dataset's code into files.

Optional dependency: `datasets`. Agar installed nahi hai to clear ImportError
milega (pipeline gracefully skip kar sakta hai). Har example se code/text field
nikaal ke stage_dir me ek file likhta hai.
"""

from __future__ import annotations

import os

from .base import Downloader, DownloadError, StagedRepo

# Common text/code column names across popular code datasets.
_CODE_FIELDS = ("content", "code", "func_code_string", "whole_func_string",
                "text", "source", "body")
_EXT = {"python": ".py", "javascript": ".js", "typescript": ".ts", "go": ".go",
        "rust": ".rs", "java": ".java", "cpp": ".cpp", "markdown": ".md"}


class HuggingFaceDownloader(Downloader):
    kind = "huggingface"

    def __init__(self, split: str = "train", max_examples: int = 5000):
        self.split = split
        self.max_examples = max_examples

    def available(self) -> bool:
        try:
            import datasets  # noqa: F401
            return True
        except Exception:
            return False

    def fetch(self, source, stage_dir: str) -> StagedRepo:
        if not self.available():
            raise DownloadError(
                "huggingface source needs the `datasets` package: "
                "pip install 'ryth[corpus-hf]'")
        import datasets  # local import (optional dep)

        try:
            ds = datasets.load_dataset(source.location, split=self.split,
                                       streaming=True)
        except Exception as e:                          # pragma: no cover - network
            raise DownloadError(f"hf load_dataset failed for {source.location!r}: {e}")

        repo = source.id.replace(":", "_").replace("/", "_")
        dest = os.path.join(stage_dir, repo)
        os.makedirs(dest, exist_ok=True)
        lang = source.languages[0] if source.languages else "python"
        ext = _EXT.get(lang, ".txt")

        n = 0
        for ex in ds:                                   # pragma: no cover - network
            field = next((f for f in _CODE_FIELDS if ex.get(f)), None)
            if not field:
                continue
            text = ex[field]
            if not isinstance(text, str) or not text.strip():
                continue
            with open(os.path.join(dest, f"example_{n:06d}{ext}"), "w",
                      encoding="utf-8") as f:
                f.write(text)
            n += 1
            if n >= self.max_examples:
                break
        if n == 0:                                      # pragma: no cover - network
            raise DownloadError(f"no code fields found in {source.location!r}")
        return StagedRepo(repo=source.id, source="huggingface", root=dest,
                          license_hint=source.license_hint)
