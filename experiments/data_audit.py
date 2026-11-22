"""Streaming exact document/provenance audit for Ryth JSONL exports.

Example: python experiments/data_audit.py --train train.jsonl --validation val.jsonl
    --tokenizer tokenizer.json

Each JSONL object needs string `content` (corpus export) or `text` (dataset
record). Optional metadata is read from the top level: repository (or repo),
path, source, license, language, revision and license_evidence. Split files
are supplied explicitly; any nonempty split field must agree. Documents are
compared case- and whitespace-sensitively after NFC normalization, matching
Ryth's tokenizer. Only full-document exact matches are checked; this is not
benchmark decontamination or a near-duplicate, substring or family scan.
License strings are self-reported metadata, not legal determinations.
Results go to stdout; inputs are never modified. SQLite scratch data, which
contains document text, is placed in a private temporary directory and deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.bpe import BPETokenizer

FIELDS = ("repository", "path", "source", "license", "language", "revision", "license_evidence")
UNKNOWN = {"unknown", "none", "null", "n/a", "na", "unspecified", "noassertion"}


def iter_records(path):
    """Read one JSON object per nonblank UTF-8 line, with actionable errors."""
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, record


def text_of(record, location):
    values = [record[k] for k in ("content", "text") if k in record]
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{location}: content or text must be a string")
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError(f"{location}: conflicting content and text")
    return values[0]


def metadata(record, field):
    values = [record[k] for k in (("repository", "repo") if field == "repository" else (field,)) if k in record]
    if not values or all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
        return "missing", None
    if any(not isinstance(v, str) or not v.strip() or v.strip().lower() in UNKNOWN for v in values):
        return "ambiguous", None
    values = [v.strip() for v in values]
    if len(set(values)) != 1:
        return "ambiguous", None
    # Expressions, multiple values and unreviewed free text are not parsed or approved.
    if field == "license" and (any(c in values[0] for c in "(),;/") or
                               len(values[0].split()) != 1):
        return "ambiguous", None
    return "present", values[0]


def audit(train, validation, tokenizer=None):
    """Return aggregate statistics; memory is bounded by one record and SQLite cache."""
    stats = {s: {"records": 0, "utf8_bytes": 0, "normalized_utf8_bytes": 0,
                 "tokens": 0 if tokenizer else None,
                 "metadata": {f: {"missing": 0, "ambiguous": 0, "present": 0} for f in FIELDS}}
             for s in ("train", "validation")}
    with tempfile.TemporaryDirectory(prefix="ryth-audit-") as tmp:
        with sqlite3.connect(os.path.join(tmp, "audit.sqlite")) as db:
            db.executescript("""
                PRAGMA cache_size = -4096;
                CREATE TABLE documents (digest BLOB NOT NULL, content BLOB NOT NULL,
                    train_count INTEGER NOT NULL DEFAULT 0, validation_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (digest, content)) WITHOUT ROWID;
                CREATE TABLE metadata_values (split TEXT, field TEXT, value TEXT, n INTEGER,
                    PRIMARY KEY (split, field, value)) WITHOUT ROWID;
            """)
            for split, path in (("train", train), ("validation", validation)):
                stat = stats[split]
                for line_no, record in iter_records(path):
                    location = f"{path}:{line_no}"
                    given_split = record.get("split")
                    if given_split not in (None, "", split):
                        raise ValueError(f"{location}: split must be {split!r}, got {given_split!r}")
                    text = text_of(record, location)
                    raw = text.encode("utf-8")
                    normalized = unicodedata.normalize("NFC", text)
                    data = normalized.encode("utf-8")
                    stat["records"] += 1
                    stat["utf8_bytes"] += len(raw)
                    stat["normalized_utf8_bytes"] += len(data)
                    if tokenizer:
                        stat["tokens"] += len(tokenizer.encode(normalized))
                    digest = hashlib.sha256(data).digest()
                    db.execute(f"INSERT INTO documents (digest, content, {split}_count) VALUES (?, ?, 1) "
                               f"ON CONFLICT(digest, content) DO UPDATE SET {split}_count = {split}_count + 1",
                               (digest, data))
                    for field in FIELDS:
                        status, value = metadata(record, field)
                        stat["metadata"][field][status] += 1
                        if value is not None and field in ("repository", "source", "license", "language"):
                            db.execute("INSERT INTO metadata_values VALUES (?, ?, ?, 1) "
                                       "ON CONFLICT(split, field, value) DO UPDATE SET n = n + 1",
                                       (split, field, value))
                db.commit()
                unique = db.execute(f"SELECT COUNT(*) FROM documents WHERE {split}_count > 0").fetchone()[0]
                stat["unique_documents"] = unique
                stat["duplicate_records"] = stat["records"] - unique
                stat["metadata_values"] = {}
                for field in ("repository", "source", "license", "language"):
                    distinct = db.execute("SELECT COUNT(*) FROM metadata_values WHERE split=? AND field=?", (split, field)).fetchone()[0]
                    top = db.execute("SELECT value, n FROM metadata_values WHERE split=? AND field=? ORDER BY n DESC, value LIMIT 50", (split, field)).fetchall()
                    stat["metadata_values"][field] = {"distinct": distinct, "top_50": dict(top)}
            overlap = db.execute("SELECT COUNT(*), COALESCE(SUM(train_count),0), COALESCE(SUM(validation_count),0) FROM documents WHERE train_count > 0 AND validation_count > 0").fetchone()
            total_unique = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    return {"normalization": "NFC; case and whitespace preserved", "scope": "exact full-document matches only; no near-duplicate, substring, benchmark or repository-family scan",
            "license_notice": "Metadata only; presence or a label is not evidence of rights or legal approval.",
            "tokenizer": {"sha256": tokenizer.compute_hash(), "vocab_size": tokenizer.vocab_size} if tokenizer else None,
            "splits": stats, "unique_documents_all_splits": total_unique,
            "overlap": {"unique_documents": overlap[0], "train_records": overlap[1], "validation_records": overlap[2]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="train JSONL")
    parser.add_argument("--validation", required=True, help="validation JSONL")
    parser.add_argument("--tokenizer", help="optional Ryth tokenizer JSON for token counts")
    args = parser.parse_args(argv)
    try:
        tok = BPETokenizer.load(args.tokenizer) if args.tokenizer else None
        result = audit(args.train, args.validation, tok)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (OSError, ValueError, UnicodeError, sqlite3.Error, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
