"""Ryth Corpus configuration — single source of truth for a corpus build.

Ek `CorpusConfig` object poore corpus pipeline ko control karta hai: language
ratios, task ratios, quality threshold, license policy, split ratios, dedup
thresholds, aur size caps. Isi ko badal ke 30M se 1B tak ke liye dataset banao.

Pure standard library. Koi bhi field ko `configs/` me JSON se load kiya ja sakta
hai (`CorpusConfig.from_dict`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# Priority coding languages. Ratios sum ko 1.0 pe normalize kiya jaata hai.
DEFAULT_LANGUAGE_RATIOS = {
    "python": 0.28, "javascript": 0.12, "typescript": 0.10, "rust": 0.08,
    "go": 0.08, "cpp": 0.07, "java": 0.07, "bash": 0.03, "sql": 0.03,
    "html": 0.03, "css": 0.02, "json": 0.03, "yaml": 0.02, "markdown": 0.03,
    "dockerfile": 0.01,
}

# Training task mix. Ratios normalize hote hain.
DEFAULT_TASK_RATIOS = {
    "next_token": 0.40, "fim": 0.15, "completion": 0.10, "editing": 0.05,
    "docstring_to_code": 0.06, "readme_to_code": 0.04, "code_to_explanation": 0.06,
    "bug_fixing": 0.05, "refactoring": 0.04, "unit_test_generation": 0.05,
}

# Permissive licenses accept-list (SPDX ids). GPL / unknown reject by default.
DEFAULT_ALLOWED_LICENSES = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
                            "ISC", "MPL-2.0", "Unlicense", "0BSD")


def _normalize(ratios: dict) -> dict:
    total = float(sum(ratios.values()))
    if total <= 0:
        raise ValueError("ratio map must have a positive sum")
    return {k: v / total for k, v in ratios.items()}


@dataclass
class CorpusConfig:
    # --- identity ---
    name: str = "ryth-corpus"
    version: str = "1.0.0"

    # --- languages ---
    language_ratios: dict = field(
        default_factory=lambda: dict(DEFAULT_LANGUAGE_RATIOS))
    enforce_language_ratios: bool = False   # True -> downsample to hit ratios

    # --- licenses ---
    allowed_licenses: tuple = DEFAULT_ALLOWED_LICENSES
    allow_unknown_licenses: bool = False    # reject unknown/GPL unless True
    allow_copyleft: bool = False            # allow GPL/AGPL/LGPL if True

    # --- cleaning / size caps ---
    max_file_bytes: int = 1 * 1024 ** 2     # 1 MB per file
    min_file_bytes: int = 16
    max_line_length: int = 1000             # minified/generated heuristic
    strip_secrets: bool = True              # redact detected secrets
    drop_secret_files: bool = False         # drop instead of redact
    strip_notebook_outputs: bool = True

    # --- dedup ---
    dedup_files: bool = True
    dedup_repos: bool = True
    near_dedup: bool = True
    near_dup_threshold: float = 0.85        # Jaccard >= -> near-duplicate
    minhash_perms: int = 64
    minhash_bands: int = 16

    # --- quality ---
    min_quality: int = 40                   # 0..100; repos below are dropped
    min_file_quality: int = 0

    # --- splits (repo-level, deterministic) ---
    split_ratios: dict = field(
        default_factory=lambda: {"train": 0.98, "validation": 0.01, "test": 0.01})

    # --- tasks ---
    task_ratios: dict = field(default_factory=lambda: dict(DEFAULT_TASK_RATIOS))
    fim_rate: float = 0.5                   # fraction of functions turned into FIM
    seq_char_budget: int = 4000             # soft per-example char budget for tasks

    # --- determinism ---
    seed: int = 20240 + 1                   # deterministic (no Date/random needed)

    def __post_init__(self):
        self.language_ratios = _normalize(self.language_ratios)
        self.task_ratios = _normalize(self.task_ratios)
        self.split_ratios = _normalize(self.split_ratios)
        assert 0.0 <= self.near_dup_threshold <= 1.0
        assert 0 <= self.min_quality <= 100
        assert self.minhash_bands >= 1 and self.minhash_perms % self.minhash_bands == 0, \
            "minhash_perms must be divisible by minhash_bands"
        assert self.max_file_bytes > self.min_file_bytes

    # ---- serialization ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusConfig":
        known = {f for f in cls.__dataclass_fields__}          # noqa: E1133
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def rows_per_band(self) -> int:
        return self.minhash_perms // self.minhash_bands
