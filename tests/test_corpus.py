"""Unit tests for the Ryth Corpus engineering pipeline (corpus/ package).

Sab kuch offline hai (LocalDownloader + synthetic repos) — koi network nahi. Har
module cover hota hai: config, metadata, licenses, languages, sources, download,
cleaners, filters, dedup, quality, split, tasks, exporters, report, pipeline, cli.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from corpus import (CorpusConfig, CorpusPipeline, FileRecord, RepoRecord,
                    Source, SourceList, build_task_dataset)
from corpus.metadata import RecordStore, content_hash
from corpus.metadata.store import read_repo_records, write_repo_records


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
MIT = ("MIT License\n\nPermission is hereby granted, free of charge, to any "
       "person obtaining a copy of this software.")
APACHE = "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/"
GPL = ("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
       "This program is free software.")

GOOD_PY = ('"""A small utility module."""\n\n\n'
           'def add(a: int, b: int) -> int:\n'
           '    """Return the sum of two integers."""\n'
           '    return a + b\n\n\n'
           'def clamp(x, lo, hi):\n'
           '    # keep within range\n'
           '    if x < lo:\n        return lo\n'
           '    if x > hi:\n        return hi\n    return x\n')
TEST_PY = "from mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _write(path, content, binary=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if binary else "w"
    with open(path, mode) as f:
        f.write(content)


def build_repo(root, name="alpha", license_text=MIT, extra=None):
    r = os.path.join(root, name)
    _write(os.path.join(r, "LICENSE"), license_text)
    _write(os.path.join(r, "mod.py"), GOOD_PY)
    _write(os.path.join(r, "test_mod.py"), TEST_PY)
    _write(os.path.join(r, "README.md"), f"# {name}\nA demo library for adding numbers.\n")
    for rel, content in (extra or {}).items():
        _write(os.path.join(r, rel), content)
    return r


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_normalizes_ratios():
    c = CorpusConfig()
    assert abs(sum(c.language_ratios.values()) - 1.0) < 1e-9
    assert abs(sum(c.task_ratios.values()) - 1.0) < 1e-9
    assert abs(sum(c.split_ratios.values()) - 1.0) < 1e-9


def test_config_rows_per_band_and_roundtrip():
    c = CorpusConfig(minhash_perms=64, minhash_bands=16)
    assert c.rows_per_band == 4
    d = c.to_dict()
    c2 = CorpusConfig.from_dict(d)
    assert c2.minhash_perms == 64


def test_config_bad_band_divisor():
    with pytest.raises(AssertionError):
        CorpusConfig(minhash_perms=10, minhash_bands=3)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def test_filerecord_from_bytes_and_hash():
    data = b"print('hi')\n"
    r = FileRecord.from_bytes("owner/repo", "a.py", data, source="local")
    assert r.hash == content_hash(data)
    assert r.size == len(data)
    assert r.content == "print('hi')\n"
    assert "content" not in r.to_dict()          # excluded by default
    assert r.kept


def test_record_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "recs.jsonl")
        recs = [FileRecord("r", "a.py", hash="h1", size=3),
                FileRecord("r", "b.py", hash="h2", size=4)]
        n = RecordStore(p).write(recs)
        assert n == 2
        back = list(RecordStore(p).read())
        assert [x.path for x in back] == ["a.py", "b.py"]


def test_repo_records_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "repos.jsonl")
        write_repo_records(p, [RepoRecord("r", quality_score=80.0, n_files=3)])
        back = list(read_repo_records(p))
        assert back[0].quality_score == 80.0


# --------------------------------------------------------------------------- #
# licenses
# --------------------------------------------------------------------------- #
def test_license_detection():
    from corpus.licenses import detect_license_text
    assert detect_license_text(MIT) == "MIT"
    assert detect_license_text(APACHE) == "Apache-2.0"
    assert detect_license_text(GPL).startswith("GPL-3.0")
    assert detect_license_text("random text") == "UNKNOWN"
    assert detect_license_text("SPDX-License-Identifier: ISC") == "ISC"


def test_license_policy():
    from corpus.licenses import is_allowed
    allowed = ("MIT", "Apache-2.0")
    assert is_allowed("MIT", allowed=allowed)
    assert not is_allowed("GPL-3.0-only", allowed=allowed)
    assert not is_allowed("UNKNOWN", allowed=allowed)
    assert is_allowed("GPL-3.0-only", allowed=allowed, allow_copyleft=True)
    assert is_allowed("UNKNOWN", allowed=allowed, allow_unknown=True)


def test_detect_repo_license():
    from corpus.licenses import detect_repo_license
    assert detect_repo_license({"LICENSE": MIT, "a.py": "x=1"}) == "MIT"
    assert detect_repo_license({"a.py": "# SPDX-License-Identifier: BSD-3-Clause"}) \
        == "BSD-3-Clause"


# --------------------------------------------------------------------------- #
# languages
# --------------------------------------------------------------------------- #
def test_language_detection():
    from corpus.languages import detect_language, is_priority
    assert detect_language("a/b.py") == "python"
    assert detect_language("x.ts") == "typescript"
    assert detect_language("main.go") == "go"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("run.sh", "#!/bin/bash\n") == "bash"
    assert detect_language("weird.xyz") == "unknown"
    assert is_priority("python") and not is_priority("cobol")


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_source_validation_and_list():
    with pytest.raises(ValueError):
        Source(id="x", kind="ftp", location="y")
    sl = SourceList([Source("a", "local", "/tmp/a"),
                     Source("b", "github", "o/n", enabled=False)])
    assert len(sl.enabled()) == 1
    assert sl.by_kind("local")[0].id == "a"
    assert Source.from_dict(sl.to_list()[0]).id == "a"


def test_default_sources_permissive():
    from corpus.sources import default_sources
    sl = default_sources()
    assert all(s.kind in ("github", "huggingface", "http", "local")
               for s in sl.sources)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def test_local_downloader_iter_files():
    from corpus.download import LocalDownloader, resolve_downloader
    with tempfile.TemporaryDirectory() as tmp:
        r = build_repo(tmp, "alpha")
        staged = LocalDownloader().fetch(Source("local:alpha", "local", r), tmp)
        files = dict(staged.iter_files())
        assert "mod.py" in files and isinstance(files["mod.py"], bytes)
        assert resolve_downloader("github").kind == "github"


def test_download_availability():
    from corpus.download import (GitHubDownloader, HTTPDownloader,
                                 HuggingFaceDownloader)
    assert GitHubDownloader().available()          # urllib always present
    assert HTTPDownloader().available()
    # hf availability depends on `datasets`; just ensure it returns a bool
    assert isinstance(HuggingFaceDownloader().available(), bool)


# --------------------------------------------------------------------------- #
# cleaners
# --------------------------------------------------------------------------- #
def test_cleaner_rules():
    from corpus.cleaners import rules
    assert rules.is_vendor_path("a/node_modules/x.js")
    assert rules.is_lockfile("yarn.lock")
    assert rules.is_binary(b"\x00\x01\x02BIN")
    assert rules.has_corrupted_encoding(b"\xff\xfe\x00bad")
    assert rules.is_minified("bundle.min.js", "var a=1;" * 500)
    assert rules.is_generated("// @generated do not edit\ncode")
    assert rules.is_too_large(2000, 1000)


def test_notebook_output_stripping():
    from corpus.cleaners import rules
    nb = json.dumps({"cells": [{"cell_type": "code", "source": ["print(1)"],
                                "outputs": [{"text": "1"}], "execution_count": 5}]})
    out = json.loads(rules.strip_notebook_outputs(nb))
    assert out["cells"][0]["outputs"] == []
    assert out["cells"][0]["execution_count"] is None


def test_secret_detection_and_redaction():
    from corpus.cleaners import find_secrets, redact_secrets
    text = 'AKIAIOSFODNN7EXAMPLE1\napi_key = "supersecretvalue123"\n'
    assert find_secrets(text)
    red, n = redact_secrets(text)
    assert n >= 1 and "REDACTED" in red and "supersecretvalue123" not in red


def test_cleaner_pipeline_decisions():
    from corpus.cleaners import Cleaner
    cl = Cleaner(CorpusConfig())
    assert not cl.inspect("a/node_modules/x.js", b"x=1").keep
    assert not cl.inspect("blob.bin", b"\x00\x01\x02").keep
    res = cl.inspect("s.py", b'token = "abcdefgh12345678xyz"\n')
    assert res.keep and res.secrets_redacted >= 1


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def test_language_balance():
    from corpus.filters import balance_language_ratios
    recs = ([FileRecord("r", f"p{i}.py", language="python", hash=f"py{i}")
             for i in range(10)]
            + [FileRecord("r", f"g{i}.go", language="go", hash=f"go{i}")
               for i in range(2)])
    kept = balance_language_ratios(recs, {"python": 0.5, "go": 0.5})
    langs = [r.language for r in kept]
    assert langs.count("python") == langs.count("go")   # balanced


def test_filter_rules():
    from corpus.filters import LicenseFilter, LanguageFilter, SizeFilter
    cfg = CorpusConfig()
    assert LicenseFilter(cfg).allows("MIT")
    assert not LicenseFilter(cfg).allows("GPL-3.0-only")
    assert LanguageFilter().check(FileRecord("r", "a.py", language="python"))[0]
    assert not LanguageFilter().check(FileRecord("r", "a", language="unknown"))[0]
    big = FileRecord("r", "a.py", language="python", size=10 ** 8)
    assert SizeFilter(cfg).check(big) == (False, "too_large")


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_exact_file_dedup():
    from corpus.dedup import dedupe_files
    recs = [FileRecord("r", "a.py", hash="h"), FileRecord("r", "b.py", hash="h"),
            FileRecord("r", "c.py", hash="k")]
    kept, dropped = dedupe_files(recs)
    assert len(kept) == 2 and dropped[0].drop_reason == "duplicate_file"


def test_repo_dedup():
    from corpus.dedup import dedupe_repos, repo_signature
    keep = dedupe_repos({"a": ["h1", "h2"], "b": ["h2", "h1"], "c": ["h3"]})
    assert sum(keep.values()) == 2          # a==b collapse
    assert repo_signature(["h1", "h2"]) == repo_signature(["h2", "h1"])


def test_near_dedup():
    from corpus.dedup import minhash_signature, jaccard_estimate, dedupe_near
    a = ("def alpha(x):\n    total = x + 1\n    return total\n\n"
         "def beta(y):\n    result = y * 2\n    return result\n\n"
         "def gamma(z):\n    return z - 3\n")
    b = a.replace("z - 3", "z - 4")          # tiny in-place edit -> near-dup
    sig_a, sig_b = minhash_signature(a, 64), minhash_signature(b, 64)
    assert jaccard_estimate(sig_a, sig_a) == 1.0
    assert jaccard_estimate(sig_a, sig_b) > 0.7
    recs = [FileRecord("r", "a.py", hash="1", content=a),
            FileRecord("r", "b.py", hash="2", content=b),
            FileRecord("r", "c.py", hash="3",
                       content="wholly unrelated prose about weather and gardens")]
    kept, dropped = dedupe_near(recs, 64, 16, 0.7)
    assert len(kept) == 2 and dropped[0].drop_reason == "near_duplicate"


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
def test_quality_signals_and_score():
    from corpus.quality import check_syntax, repo_signals, score_repo
    assert check_syntax("python", "def f():\n    return 1\n") is True
    assert check_syntax("python", "def f(:\n") is False
    assert check_syntax("go", "func x() {}") is True
    recs = [FileRecord("r", "mod.py", language="python", hash="1", content=GOOD_PY),
            FileRecord("r", "test_mod.py", language="python", hash="2", content=TEST_PY),
            FileRecord("r", "README.md", language="markdown", hash="3", content="# r\ndocs")]
    sig = repo_signals(recs)
    assert 0.0 <= sig["syntax_validity"] <= 1.0
    assert sig["tests"] > 0                          # has a test file
    score, _ = score_repo(recs)
    assert 0 <= score <= 100 and score > 20


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_split_determinism_and_no_leakage():
    from corpus.split import assign_split, split_records, verify_no_leakage
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    s1 = assign_split("owner/repo", ratios, seed=7)
    s2 = assign_split("owner/repo", ratios, seed=7)
    assert s1 == s2 and s1 in ("train", "validation", "test")
    recs = [FileRecord("repoA", f"{i}.py") for i in range(3)] + \
           [FileRecord("repoB", f"{i}.py") for i in range(3)]
    split_records(recs, ratios, seed=1)
    assert verify_no_leakage(recs)
    # all files of a repo share a split
    assert len({r.split for r in recs if r.repository == "repoA"}) == 1


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def test_python_function_extraction():
    from corpus.tasks.builders import extract_python_functions
    fns = extract_python_functions(GOOD_PY)
    names = {f["name"] for f in fns}
    assert {"add", "clamp"} <= names
    add = next(f for f in fns if f["name"] == "add")
    assert "sum of two integers" in add["docstring"]


def test_task_builders_cover_all_types():
    from corpus.tasks import builders as B
    rec = FileRecord("r", "mod.py", language="python", hash="deadbeef", content=GOOD_PY)
    rec.split = "train"
    assert B.next_token_examples(rec)
    assert B.completion_examples(rec)
    assert B.editing_examples(rec)
    assert B.docstring_to_code_examples(rec)
    assert B.code_to_explanation_examples(rec)
    assert B.bug_fixing_examples(rec)
    assert B.refactoring_examples(rec)
    fim = B.fim_examples(rec, fim_rate=1.0)
    assert fim and B.FIM_MIDDLE in fim[0]["text"]


def test_task_mixer_ratatios_and_distribution():
    from corpus.tasks import build_task_dataset, task_distribution
    recs = []
    for i in range(6):
        r = FileRecord("r", f"m{i}.py", language="python", hash=f"h{i}", content=GOOD_PY)
        r.split = "train"
        recs.append(r)
    cfg = CorpusConfig(task_ratios={"next_token": 0.5, "completion": 0.5})
    ex = build_task_dataset(recs, cfg, enforce_ratios=True)
    dist = task_distribution(ex)
    assert set(dist) <= {"next_token", "completion"}
    if ex:
        assert dist.get("next_token", 0) == dist.get("completion", 0)


# --------------------------------------------------------------------------- #
# exporters
# --------------------------------------------------------------------------- #
def _sample_records():
    recs = []
    for i in range(2):
        r = FileRecord("r", f"m{i}.py", language="python", hash=f"h{i}",
                       size=len(GOOD_PY), content=GOOD_PY, license="MIT")
        r.split = "train" if i == 0 else "validation"
        recs.append(r)
    return recs


def test_export_raw_and_jsonl():
    from corpus.exporters import export_raw, export_records_by_split
    with tempfile.TemporaryDirectory() as tmp:
        n = export_raw(_sample_records(), os.path.join(tmp, "raw"))
        assert n == 2
        assert os.path.exists(os.path.join(tmp, "raw", "train", "r", "m0.py"))
        counts = export_records_by_split(_sample_records(), os.path.join(tmp, "js"))
        assert counts["train"] == 1 and counts["validation"] == 1


def test_export_rds_uses_rde():
    from corpus.exporters import export_rds
    with tempfile.TemporaryDirectory() as tmp:
        mans = export_rds(_sample_records(), os.path.join(tmp, "rds"), seq_len=32)
        assert "train" in mans
        assert os.path.exists(os.path.join(tmp, "rds", "train", "manifest.json"))


def test_export_parquet_optional():
    from corpus.exporters import export_parquet, pyarrow_available
    if not pyarrow_available():
        pytest.skip("pyarrow not installed")
    with tempfile.TemporaryDirectory() as tmp:
        n = export_parquet(_sample_records(), os.path.join(tmp, "c.parquet"))
        assert n == 2


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_build_and_write():
    from corpus.report import build_report, write_html_report, write_json_report
    recs = _sample_records()
    repos = [RepoRecord("r", license="MIT", quality_score=75.0, n_files=2)]
    rep = build_report(recs, repos, drops={"vendor": 3})
    assert rep["dataset_size"]["files"] == 2
    assert "python" in rep["language_distribution"]
    assert rep["duplicate_statistics"]["duplicate_file"] == 0
    with tempfile.TemporaryDirectory() as tmp:
        write_json_report(rep, os.path.join(tmp, "r.json"))
        write_html_report(rep, os.path.join(tmp, "r.html"))
        assert os.path.getsize(os.path.join(tmp, "r.html")) > 0


# --------------------------------------------------------------------------- #
# pipeline (end-to-end)
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end():
    from corpus.split import verify_no_leakage
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "repos")
        build_repo(root, "alpha", MIT, extra={
            "node_modules/x.js": "var a=1",             # vendor -> drop
        })
        # binary file (bytes)
        _write(os.path.join(root, "alpha", "blob.bin"), bytes([0, 1, 2, 255]), binary=True)
        build_repo(root, "beta", APACHE, extra={"util.py": "def g():\n    return 2\n"})
        build_repo(root, "gplrepo", GPL)                 # license -> drop

        cfg = CorpusConfig(min_quality=0)
        pipe = CorpusPipeline(cfg)
        sl = SourceList([Source(f"local:{n}", "local", os.path.join(root, n))
                         for n in ("alpha", "beta", "gplrepo")])
        res = pipe.build(sl.enabled(), os.path.join(tmp, "stage"),
                         created_at="2026-07-05T00:00:00Z")

        assert res.n_files > 0
        assert verify_no_leakage(res.records)
        licenses = {r.license for r in res.records}
        assert "GPL-3.0-or-later" not in licenses         # gpl dropped
        assert res.drops.get("vendor", 0) >= 1
        assert res.drops.get("binary", 0) >= 1
        assert res.drops.get("license", 0) >= 1           # gpl repo files dropped
        assert all(r.quality_score >= 0 for r in res.records)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_cli_build_and_stats(capsys):
    from corpus import cli
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "repos")
        build_repo(root, "alpha", MIT)
        build_repo(root, "beta", APACHE, extra={"util.py": "def g():\n    return 2\n"})
        out = os.path.join(tmp, "out")
        rc = cli.main(["build", "--input", root, "--out", out, "--min-quality", "0",
                       "--tasks"])
        assert rc == 0
        assert os.path.exists(os.path.join(out, "records.jsonl"))
        assert os.path.exists(os.path.join(out, "report.html"))
        assert os.path.exists(os.path.join(out, "tasks.jsonl"))
        rc = cli.main(["stats", "--records", os.path.join(out, "records.jsonl")])
        assert rc == 0
        rc = cli.main(["export", "--records", os.path.join(out, "records.jsonl"),
                       "--format", "jsonl", "--out", os.path.join(tmp, "exp")])
        assert rc == 0


if __name__ == "__main__":       # allow `python tests/test_corpus.py`
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
