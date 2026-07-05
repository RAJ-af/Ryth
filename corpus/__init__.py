"""Ryth Corpus v1.0 — a corpus engineering system for coding LLMs.

Raw code repositories se ek world-class, license-clean, deduplicated, quality-
scored, task-formatted training dataset banata hai — 30M se 1B tak ke Ryth models
ke liye. Tokenizer / RDE / model / training engine ko chhue bina, ek alag package.

Public API:
    CorpusConfig        — build ko control karne wali config
    CorpusPipeline      — full pipeline orchestrator
    Source, SourceList, default_sources
    FileRecord, RepoRecord
    build_task_dataset  — task-formatted examples (FIM, bugfix, …)
    build_report / write_json_report / write_html_report
    exporters (export_raw / _jsonl / _parquet / _rds)
"""

from .config import CorpusConfig
from .metadata import FileRecord, RepoRecord
from .pipeline import CorpusPipeline, CorpusResult
from .report import build_report, write_html_report, write_json_report
from .sources import Source, SourceList, default_sources
from .tasks import build_task_dataset

__version__ = "1.0.0"

__all__ = [
    "CorpusConfig", "CorpusPipeline", "CorpusResult",
    "Source", "SourceList", "default_sources",
    "FileRecord", "RepoRecord", "build_task_dataset",
    "build_report", "write_json_report", "write_html_report",
    "__version__",
]
