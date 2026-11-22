"""Measure a saved Ryth tokenizer on held-out multilingual JSONL samples.

Example: python experiments/tokenizer_benchmark.py --tokenizer tokenizer.json --input val.jsonl
Each nonblank line is an object with string `language` and `text` (or corpus
export `content`). Use held-out samples representative of each language, not
samples used to train the tokenizer. Metrics are descriptive, not model quality.
Tokenization uses Ryth's NFC normalization: raw and normalized byte counts and
roundtrip changes are reported separately. Throughput times encoding only and
includes per-sample timer overhead, not file I/O or decoding. JSON goes to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.data_audit import iter_records, text_of
from tokenizer.bpe import BPETokenizer


def benchmark(path, tokenizer):
    languages = {}
    for line_no, record in iter_records(path):
        location = f"{path}:{line_no}"
        language = record.get("language")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"{location}: language must be a nonempty string")
        language = language.strip()
        text = text_of(record, location)
        normalized = unicodedata.normalize("NFC", text)
        row = languages.setdefault(language, {"samples": 0, "utf8_bytes": 0,
            "normalized_utf8_bytes": 0, "tokens": 0, "encode_seconds": 0.0,
            "nfc_changed_samples": 0, "original_roundtrip_mismatches": 0,
            "normalized_roundtrip_mismatches": 0, "special_token_text_samples": 0})
        raw_bytes = len(text.encode("utf-8"))
        norm_bytes = len(normalized.encode("utf-8"))
        start = time.perf_counter()
        ids = tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        decoded = tokenizer.decode(ids)
        row["samples"] += 1
        row["utf8_bytes"] += raw_bytes
        row["normalized_utf8_bytes"] += norm_bytes
        row["tokens"] += len(ids)
        row["encode_seconds"] += elapsed
        row["nfc_changed_samples"] += text != normalized
        row["original_roundtrip_mismatches"] += decoded != text
        row["normalized_roundtrip_mismatches"] += decoded != normalized
        row["special_token_text_samples"] += any(s in normalized for s in tokenizer.special_tokens)
    for row in languages.values():
        tokens, seconds = row["tokens"], row["encode_seconds"]
        row["utf8_bytes_per_token"] = row["utf8_bytes"] / tokens if tokens else None
        row["normalized_utf8_bytes_per_token"] = row["normalized_utf8_bytes"] / tokens if tokens else None
        row["tokens_per_second"] = tokens / seconds if seconds else None
        row["utf8_bytes_per_second"] = row["utf8_bytes"] / seconds if seconds else None
    return {"tokenizer": {"sha256": tokenizer.compute_hash(), "vocab_size": tokenizer.vocab_size},
            "normalization": "NFC in Ryth encode; raw and normalized bytes shown separately",
            "throughput_note": "Encoding only, per-sample timed; hardware and input-dependent.",
            "languages": dict(sorted(languages.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="saved Ryth tokenizer JSON")
    parser.add_argument("--input", required=True, help="held-out language/text JSONL")
    args = parser.parse_args(argv)
    try:
        tokenizer = BPETokenizer.load(args.tokenizer)
        result = benchmark(args.input, tokenizer)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
