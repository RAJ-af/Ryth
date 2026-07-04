"""Manifest Lock — exact-reproducibility metadata for a dataset build.

Manifest ke saath ek "lock" store hota hai jisse aap poora dataset bit-for-bit
reproduce kar sakte ho (ya kam se kam verify kar sakte ho ki kaunse inputs/config
se bana tha). Ye dataset ko science banata hai, jugaad nahi.

Fields:
    rds_version, dataset_version, tokenizer_version, tokenizer_hash,
    model_version, creation_time (UTC ISO), git_commit, python_version,
    platform, config (full RDEConfig snapshot), seed, source_root
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone


def _git_commit(cwd: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def tokenizer_hash(tokenizer) -> str:
    """Tokenizer ki identity ka stable hash (merges + special tokens se).

    Alag tokenizer => alag hash => dataset mismatch turant pakda jaayega.
    """
    h = hashlib.sha256()
    h.update(str(getattr(tokenizer, "version", 0)).encode())
    h.update(str(getattr(tokenizer, "vocab_size", 0)).encode())
    # scratch BPE ke andar tak pahunch ke merges hash karo (agar mile)
    inner = getattr(tokenizer, "_t", tokenizer)
    merges = getattr(inner, "merges", None)
    if merges is not None:
        for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
            h.update(f"{a},{b},{idx};".encode())
    special = getattr(inner, "special_tokens", {})
    for k, v in sorted(special.items()):
        h.update(f"{k}={v};".encode())
    return h.hexdigest()


def _config_snapshot(config) -> dict:
    """RDEConfig ko JSON-safe dict me badlo (sets -> sorted lists)."""
    try:
        raw = asdict(config)
    except TypeError:
        raw = {k: getattr(config, k) for k in dir(config)
               if not k.startswith("_")}
    clean = {}
    for k, v in raw.items():
        if isinstance(v, set):
            clean[k] = sorted(v)
        elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
            clean[k] = v
    return clean


def build_lock(config, tokenizer, *, dataset_version: str,
               model_version: str, source_root: str,
               now_iso: str | None = None) -> dict:
    """Reproducibility lock banao. `now_iso` inject kar sakte ho (deterministic tests)."""
    return {
        "dataset_version": dataset_version,
        "rds_version": config.rds_version,
        "tokenizer_version": getattr(tokenizer, "version",
                                     config.tokenizer_version),
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "model_version": model_version,
        "creation_time": now_iso or datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": config.seed,
        "source_root": os.path.abspath(source_root),
        "config": _config_snapshot(config),
    }


def verify_lock(lock: dict, config, tokenizer) -> list[str]:
    """Current config/tokenizer lock se match karte hain? Mismatches ki list do."""
    problems = []
    if lock.get("tokenizer_hash") != tokenizer_hash(tokenizer):
        problems.append("tokenizer_hash mismatch (alag tokenizer)")
    if lock.get("seed") != config.seed:
        problems.append(f"seed mismatch ({lock.get('seed')} != {config.seed})")
    if lock.get("rds_version") != config.rds_version:
        problems.append("rds_version mismatch")
    snap = lock.get("config", {})
    for key in ("seq_len", "vocab_size", "overlap", "add_bos", "add_eos"):
        if snap.get(key) != getattr(config, key, None):
            problems.append(f"config.{key} mismatch")
    return problems
