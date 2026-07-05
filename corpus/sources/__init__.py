"""Corpus source definitions (declarative; downloaded by corpus/download)."""

from .registry import (CATEGORIES, KINDS, Source, SourceList, default_sources)

__all__ = ["Source", "SourceList", "default_sources", "KINDS", "CATEGORIES"]
