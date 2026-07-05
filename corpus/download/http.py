"""HTTP downloader — docs / markdown / tutorials over HTTP(S).

Ek URL fetch karke stage_dir me save karta hai. HTML ko `.html`, baaki ko content
type / extension ke hisaab se save karta hai. Standard library only.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

from .base import Downloader, DownloadError, StagedRepo

_UA = {"User-Agent": "ryth-corpus/1.0 (+https://github.com/RAJ-af/Ryth)"}


class HTTPDownloader(Downloader):
    kind = "http"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def available(self) -> bool:
        return True

    def fetch(self, source, stage_dir: str) -> StagedRepo:
        url = source.location
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                ctype = resp.headers.get("Content-Type", "")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            raise DownloadError(f"http fetch failed for {url!r}: {e}")

        repo = source.id.replace(":", "_").replace("/", "_")
        dest = os.path.join(stage_dir, repo)
        os.makedirs(dest, exist_ok=True)

        name = url.rstrip("/").split("/")[-1] or "index"
        if "." not in name:
            ext = ".html" if "html" in ctype else ".md" if "markdown" in ctype else ".txt"
            name += ext
        with open(os.path.join(dest, name), "wb") as f:
            f.write(data)

        return StagedRepo(repo=source.id, source="http", root=dest,
                          license_hint=source.license_hint)
