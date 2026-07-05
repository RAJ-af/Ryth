"""Deterministic, leakage-free repo-level dataset splitting."""

from .splitter import (SPLITS, assign_split, split_records, verify_no_leakage)

__all__ = ["assign_split", "split_records", "verify_no_leakage", "SPLITS"]
