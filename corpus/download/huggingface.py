"""Hugging Face downloader — materialize a HF dataset's code into files.

Optional dependency: `datasets`. Agar installed nahi hai to clear ImportError
milega (pipeline gracefully skip kar sakta hai). Har example se code/text field
nikaal ke stage_dir me ek file likhta hai.

GATED datasets ke liye `open_streaming` fallback-chain support karta hai:
primary fail (gate/auth) ho to entry ke 'fallbacks' try hote hain — jaise
bigcode/the-stack-dedup (gated=auto) -> codeparrot/github-code @
refs/convert/parquet (ungated, verified anonymous streaming).
"""

from __future__ import annotations

import os

from .base import Downloader, DownloadError, StagedRepo

# Common text/code column names across popular code datasets.
_CODE_FIELDS = ("content", "code", "func_code_string", "whole_func_string",
                "text", "source", "body")
_EXT = {"python": ".py", "javascript": ".js", "typescript": ".ts", "go": ".go",
        "rust": ".rs", "java": ".java", "cpp": ".cpp", "c": ".c",
        "markdown": ".md"}

# gate/auth failure markers (case-insensitive substring match)
_GATE_MARKERS = ("gated", "authentication", "401", "403")


def _is_gate_error(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return any(m in s for m in _GATE_MARKERS)


def open_streaming(entry: dict, split: str = "train"):
    """entry {location, subpath?, revision?, fallbacks?} -> (ds, served_name).

    Primary load_dataset gated/auth/other-error par aage ki 'fallbacks'
    chain try hoti hai. Sab fail -> actionable DownloadError (HF_TOKEN hint
    + ungated-alternative hint). Returns the raw streaming dataset.
    """
    attempts = [{"location": entry.get("location"),
                 "subpath": entry.get("subpath") or None,
                 "revision": entry.get("revision")}]
    for fb in entry.get("fallbacks") or []:
        attempts.append({"location": fb.get("location"),
                         "subpath": fb.get("subpath") or None,
                         "revision": fb.get("revision")})
    import datasets  # local import (optional dep)

    errors: list[str] = []
    ds = None
    served = None
    for i, at in enumerate(attempts):
        try:
            ds = datasets.load_dataset(at["location"], split=split,
                                       streaming=True, data_dir=at["subpath"],
                                       revision=at["revision"])
            served = at["location"]
            break
        except Exception as e:                          # pragma: no cover - network
            gated = _is_gate_error(e)
            errors.append(f"{at['location']}: {type(e).__name__}: "
                          f"{str(e)[:160]}")
            left = len(attempts) - i - 1
            print(f"[hf] {at['location']} FAIL"
                  f"{' (GATED/auth)' if gated else ''}"
                  f"{f' -> next fallback ({left} bache)' if left else ' — koi fallback nahi'}",
                  flush=True)
            ds = None
    if ds is None:
        raise DownloadError(
            "hf load_dataset sab attempts me fail:\n  - "
            + "\n  - ".join(errors)
            + "\n[hint] GATED dataset hai? HF_TOKEN export karo YA config "
              "entry ke 'fallbacks' me ungated source do (jaise "
              "codeparrot/github-code @ refs/convert/parquet).")
    return ds, served


class HuggingFaceDownloader(Downloader):
    kind = "huggingface"

    def __init__(self, split: str = "train", max_examples: int | None = None,
                 max_bytes: int | None = None):
        self.split = split
        # NOTE: pehle default 5000 tha — W1-scale budget (~1GB+) ko chupchaap
        # ~50MB par kaat deta. Default ab UNLIMITED; byte-budget hi cap hai.
        self.max_examples = max_examples
        # byte budget: staged output is se bada hua to streaming rok do
        # (W1 scale pe poori dataset kabhi nahi utarti — budget chahiye)
        self.max_bytes = max_bytes

    def available(self) -> bool:
        try:
            import datasets  # noqa: F401
            return True
        except Exception:
            return False

    def fetch(self, source, stage_dir: str,
              fallbacks: list[dict] | None = None) -> StagedRepo:
        if not self.available():
            raise DownloadError(
                "huggingface source needs the `datasets` package: "
                "pip install 'ryth[corpus-hf]'")

        revision = getattr(source, "ref", "HEAD")
        entry = {"location": source.location,
                 "subpath": getattr(source, "subpath", ""),
                 "revision": None if revision in ("HEAD", "", None) else revision,
                 "fallbacks": fallbacks or []}
        ds, served = open_streaming(entry, split=self.split)

        repo = source.id.replace(":", "_").replace("/", "_")
        dest = os.path.join(stage_dir, repo)
        os.makedirs(dest, exist_ok=True)
        lang = source.languages[0] if source.languages else "python"
        ext = _EXT.get(lang, ".txt")

        n = 0
        staged_bytes = 0
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
            staged_bytes += len(text)
            if self.max_bytes is not None and staged_bytes >= self.max_bytes:
                break
            if self.max_examples is not None and n >= self.max_examples:
                break
        if n == 0:                                      # pragma: no cover - network
            raise DownloadError(f"no code fields found in {served!r}")
        return StagedRepo(repo=source.id, source="huggingface", root=dest,
                          license_hint=source.license_hint)
