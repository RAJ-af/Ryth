"""GitHub downloader — fetch a public repo as a zip via codeload.

Network use hota hai (koi API token nahi chahiye public repos ke liye). `datasets`
ya `git` par depend nahi karta — sirf standard library (`urllib`, `zipfile`).

Zip ko stage_dir me extract karta hai, top-level folder strip karke.
"""

from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
import zipfile

from .base import Downloader, DownloadError, StagedRepo

_UA = {"User-Agent": "ryth-corpus/1.0 (+https://github.com/RAJ-af/Ryth)"}


class GitHubDownloader(Downloader):
    kind = "github"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def available(self) -> bool:
        return True                       # urllib always present; network at fetch

    def fetch(self, source, stage_dir: str) -> StagedRepo:
        owner_name = source.location.strip("/")
        ref = source.ref or "HEAD"
        url = f"https://codeload.github.com/{owner_name}/zip/{ref}"
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                blob = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            raise DownloadError(f"github fetch failed for {owner_name}@{ref}: {e}")

        dest = os.path.join(stage_dir, owner_name.replace("/", "__"))
        os.makedirs(dest, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                names = zf.namelist()
                top = names[0].split("/", 1)[0] + "/" if names else ""
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    rel = info.filename[len(top):] if info.filename.startswith(top) \
                        else info.filename
                    if source.subpath and not rel.startswith(source.subpath.rstrip("/") + "/"):
                        continue
                    if not rel:
                        continue
                    out = os.path.join(dest, rel)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with zf.open(info) as src, open(out, "wb") as dst:
                        dst.write(src.read())
        except zipfile.BadZipFile as e:
            raise DownloadError(f"github zip corrupt for {owner_name}: {e}")

        return StagedRepo(repo=owner_name, source="github", root=dest,
                          license_hint=source.license_hint)
