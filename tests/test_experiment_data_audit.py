"""Offline tests for the exact audit and multilingual tokenizer benchmark."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.data_audit import audit
from experiments.tokenizer_benchmark import benchmark
from tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path, records):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    return path


def tokenizer(tmp_path):
    tok = BPETokenizer().train(["def add(x): return x + 1"] * 3, vocab_size=270)
    tok.add_special_tokens(["<|fim_prefix|>"])
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    return BPETokenizer.load(path), path


def cli(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "experiments" / script), *map(str, args)],
                          text=True, capture_output=True, cwd=ROOT)


def test_duplicates_overlap_provenance_and_tokens(tmp_path):
    tok, _ = tokenizer(tmp_path)
    train = jsonl(tmp_path / "train.jsonl", [
        {"content": "café\n", "repository": "a/r", "path": "x.py", "source": "local", "license": "MIT", "language": "python", "split": "train"},
        {"text": "cafe\u0301\n", "repo": "a/r", "license": "UNKNOWN"},
        {"text": "other", "repository": "a/r", "repo": "b/r", "license": "MIT OR Apache-2.0"},
        {"text": "CAFÉ\n", "license": ["MIT", "GPL-3.0"]},
    ])
    val = jsonl(tmp_path / "val.jsonl", [
        {"content": "café\n", "license": "MIT", "split": "validation"},
        {"text": "cafe\u0301\n"},
        {"text": "distinct"},
    ])
    result = audit(train, val, tok)
    t = result["splits"]["train"]
    v = result["splits"]["validation"]
    assert (t["records"], t["unique_documents"], t["duplicate_records"]) == (4, 3, 1)
    assert (v["unique_documents"], v["duplicate_records"]) == (2, 1)
    assert result["unique_documents_all_splits"] == 4
    assert result["overlap"] == {"unique_documents": 1, "train_records": 2, "validation_records": 2}
    assert t["metadata"]["license"] == {"present": 1, "missing": 0, "ambiguous": 3}
    assert t["metadata"]["repository"] == {"present": 2, "missing": 1, "ambiguous": 1}
    assert t["metadata_values"]["license"]["top_50"] == {"MIT": 1}
    assert t["tokens"] == sum(len(tok.encode(x)) for x in ["café\n", "cafe\u0301\n", "other", "CAFÉ\n"])
    assert t["utf8_bytes"] > t["normalized_utf8_bytes"]
    assert result["tokenizer"]["sha256"] == tok.compute_hash()


def test_multilingual_benchmark_reports_normalization_and_special_tokens(tmp_path):
    tok, _ = tokenizer(tmp_path)
    path = jsonl(tmp_path / "samples.jsonl", [
        {"language": "Hindi", "text": "नमस्ते"},
        {"language": "Python", "text": "cafe\u0301 <|fim_prefix|>"},
        {"language": "Python", "text": ""},
        {"language": "English", "text": "hello"},
    ])
    rows = benchmark(path, tok)["languages"]
    hindi = rows["Hindi"]
    assert hindi["samples"] == 1
    assert hindi["utf8_bytes_per_token"] == len("नमस्ते".encode()) / len(tok.encode("नमस्ते"))
    assert hindi["tokens_per_second"] > 0
    python = rows["Python"]
    assert python["samples"] == 2
    assert python["nfc_changed_samples"] == 1
    assert python["original_roundtrip_mismatches"] == 1
    assert python["normalized_roundtrip_mismatches"] == 0
    assert python["special_token_text_samples"] == 1


def test_cli_success_help_and_bad_input(tmp_path):
    _, tokpath = tokenizer(tmp_path)
    train = jsonl(tmp_path / "train.jsonl", [{"text": "one", "language": "English"}])
    val = jsonl(tmp_path / "val.jsonl", [{"text": "one", "language": "English"}])
    for script in ("data_audit.py", "tokenizer_benchmark.py"):
        help_result = cli(script, "--help")
        assert help_result.returncode == 0 and "usage:" in help_result.stdout
    audit_result = cli("data_audit.py", "--train", train, "--validation", val, "--tokenizer", tokpath)
    assert audit_result.returncode == 0, audit_result.stderr
    assert json.loads(audit_result.stdout)["overlap"]["unique_documents"] == 1
    bench_result = cli("tokenizer_benchmark.py", "--input", val, "--tokenizer", tokpath)
    assert bench_result.returncode == 0, bench_result.stderr
    assert json.loads(bench_result.stdout)["languages"]["English"]["samples"] == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text":\n', encoding="utf-8")
    result = cli("data_audit.py", "--train", bad, "--validation", val)
    assert result.returncode != 0 and "bad.jsonl:1" in result.stderr
    assert result.stdout == ""
    missing_language = cli("tokenizer_benchmark.py", "--input", jsonl(bad, [{"text": "x"}]), "--tokenizer", tokpath)
    assert missing_language.returncode != 0 and "language" in missing_language.stderr


def test_input_validation_and_empty_files(tmp_path):
    empty = jsonl(tmp_path / "empty", [])
    result = audit(empty, empty)
    assert result["unique_documents_all_splits"] == 0
    assert result["splits"]["train"]["tokens"] is None
    bad = jsonl(tmp_path / "bad", [{"text": "x", "split": "test"}])
    with pytest.raises(ValueError, match="split"):
        audit(bad, empty)
    jsonl(bad, [{"text": "a", "content": "b"}])
    with pytest.raises(ValueError, match="conflicting"):
        audit(bad, empty)
    jsonl(bad, [42])
    with pytest.raises(ValueError, match="object"):
        audit(bad, empty)
