"""Downloaders — materialize sources into local staging dirs.

`resolve_downloader(kind)` factory se sahi downloader milta hai. Local hamesha
available; github/http network par; huggingface `datasets` par (optional).
"""

from .base import Downloader, DownloadError, StagedRepo
from .local import LocalDownloader
from .github import GitHubDownloader
from .http import HTTPDownloader
from .huggingface import HuggingFaceDownloader

_REGISTRY = {
    "local": LocalDownloader,
    "github": GitHubDownloader,
    "http": HTTPDownloader,
    "huggingface": HuggingFaceDownloader,
}


def resolve_downloader(kind: str, **kw) -> Downloader:
    """Return a downloader instance for a source kind."""
    if kind not in _REGISTRY:
        raise ValueError(f"no downloader for kind {kind!r} (have {list(_REGISTRY)})")
    return _REGISTRY[kind](**kw)


__all__ = [
    "Downloader", "DownloadError", "StagedRepo", "resolve_downloader",
    "LocalDownloader", "GitHubDownloader", "HTTPDownloader", "HuggingFaceDownloader",
]
