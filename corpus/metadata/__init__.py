"""Corpus metadata — records + JSONL store."""

from .record import FileRecord, RepoRecord, content_hash
from .store import RecordStore, read_repo_records, write_repo_records

__all__ = [
    "FileRecord", "RepoRecord", "content_hash",
    "RecordStore", "read_repo_records", "write_repo_records",
]
