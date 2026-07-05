"""Local downloader — ingest a folder already on disk.

Ye hamesha chalta hai (no network, no deps). Do use-cases:
  * aapke paas already code hai (attach a Kaggle dataset / clone manually)
  * tests + offline pipeline runs

`source.location` ek directory path hai. `source.subpath` diya ho to sirf woh
subdir ingest hoti hai.
"""

from __future__ import annotations

import os

from .base import Downloader, DownloadError, StagedRepo


class LocalDownloader(Downloader):
    kind = "local"

    def available(self) -> bool:
        return True

    def fetch(self, source, stage_dir: str) -> StagedRepo:
        root = source.location
        if source.subpath:
            root = os.path.join(root, source.subpath)
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise DownloadError(f"local source path not found: {root!r}")
        repo = source.location.rstrip("/").replace(os.sep, "/").split("/")[-1]
        repo = repo or source.id
        return StagedRepo(repo=repo, source="local", root=root,
                          license_hint=source.license_hint)
