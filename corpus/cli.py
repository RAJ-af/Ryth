"""ryth-corpus — command line for the Ryth Corpus engineering pipeline.

Commands:
    download   sources ko staging dir me fetch karo (local/github/http/hf)
    clean      ek input tree ko ingest + clean karke raw output likho
    score      per-repo quality scores print karo
    build      full pipeline -> records.jsonl + repos.jsonl + report(.json/.html)
    stats      records.jsonl se statistics print/report karo
    export     records.jsonl ko raw/jsonl/parquet/rds me export karo

Most commands ek local input directory par kaam karte hain (offline). Har
top-level subdir ek repository maani jaati hai (RDE jaisa layout).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from .config import CorpusConfig
from .metadata import RecordStore, write_repo_records
from .pipeline import CorpusPipeline
from .quality import score_repo
from .report import build_report, write_html_report, write_json_report
from .sources import Source, SourceList
from .tasks import build_task_dataset


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: str | None, **overrides) -> CorpusConfig:
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.update({k: v for k, v in overrides.items() if v is not None})
    return CorpusConfig.from_dict(data)


def _local_sources(input_dir: str) -> SourceList:
    """One local Source per top-level subdirectory (else the dir itself)."""
    input_dir = os.path.abspath(input_dir)
    subs = [d for d in sorted(os.listdir(input_dir))
            if os.path.isdir(os.path.join(input_dir, d))] if os.path.isdir(input_dir) else []
    sl = SourceList()
    if subs:
        for d in subs:
            sl.add(Source(id=f"local:{d}", kind="local",
                          location=os.path.join(input_dir, d)))
    else:
        sl.add(Source(id="local:root", kind="local", location=input_dir))
    return sl


# --------------------------------------------------------------------------- #
def cmd_download(args):
    from .download import DownloadError, resolve_downloader
    if args.source_file:
        with open(args.source_file, encoding="utf-8") as f:
            sources = SourceList.from_list(json.load(f)).enabled()
    else:
        sources = _local_sources(args.input).enabled()
    os.makedirs(args.stage, exist_ok=True)
    ok = 0
    for s in sources:
        dl = resolve_downloader(s.kind)
        if not dl.available():
            print(f"  skip {s.id} (downloader unavailable)")
            continue
        try:
            staged = dl.fetch(s, args.stage)
        except DownloadError as e:
            print(f"  fail {s.id}: {e}")
            continue
        n = sum(1 for _ in staged.iter_files())
        print(f"  staged {staged.repo}: {n} files -> {staged.root}")
        ok += 1
    print(f"[download] {ok}/{len(sources)} sources staged")
    return 0


def cmd_clean(args):
    from .exporters import export_raw
    cfg = _load_config(args.config)
    pipe = CorpusPipeline(cfg)
    records = pipe.ingest(_local_sources(args.input).enabled(), args.stage or args.input, _now())
    kept = [r for r in records if not r.drop_reason]
    for r in kept:
        r.split = "train"
    n = export_raw(kept, os.path.join(args.out, "raw"))
    drops: dict = {}
    for r in records:
        if r.drop_reason:
            drops[r.drop_reason] = drops.get(r.drop_reason, 0) + 1
    print(f"[clean] kept {n} / {len(records)} files")
    for reason, c in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"    dropped {c:6d}  {reason}")
    return 0


def cmd_score(args):
    pipe = CorpusPipeline(_load_config(args.config))
    records = pipe.ingest(_local_sources(args.input).enabled(), args.stage or args.input, _now())
    by_repo: dict = {}
    for r in records:
        if not r.drop_reason:
            r.language = r.language if r.language != "unknown" else r.language
            by_repo.setdefault(r.repository, []).append(r)
    from .filters import annotate_language
    rows = []
    for repo, recs in by_repo.items():
        for r in recs:
            annotate_language(r)
        score, signals = score_repo(recs)
        rows.append((score, repo, len(recs), signals))
    rows.sort(key=lambda t: -t[0])
    if args.json:
        print(json.dumps([{"repository": r, "quality_score": s, "n_files": n,
                           "signals": sig} for s, r, n, sig in rows], indent=2))
    else:
        for s, repo, n, _sig in rows:
            print(f"  {s:6.2f}  {repo}  ({n} files)")
    return 0


def cmd_build(args):
    cfg = _load_config(args.config, min_quality=args.min_quality, seed=args.seed)
    pipe = CorpusPipeline(cfg)
    result = pipe.build(_local_sources(args.input).enabled(), args.stage or args.input, _now())

    os.makedirs(args.out, exist_ok=True)
    RecordStore(os.path.join(args.out, "records.jsonl")).write(
        result.records, include_content=True)
    write_repo_records(os.path.join(args.out, "repos.jsonl"), result.repos)

    examples = None
    if args.tasks:
        examples = build_task_dataset(result.records, cfg)
        from .exporters import export_tasks_jsonl
        export_tasks_jsonl(examples, os.path.join(args.out, "tasks.jsonl"))

    report = build_report(result.records, result.repos, examples=examples,
                          drops=result.drops, config=cfg)
    write_json_report(report, os.path.join(args.out, "report.json"))
    write_html_report(report, os.path.join(args.out, "report.html"))

    ds = report["dataset_size"]
    print(f"[build] {ds['files']} files / {ds['repositories']} repos / "
          f"{ds['megabytes']} MB")
    print(f"    splits: {report['split_distribution']}")
    if examples is not None:
        print(f"    tasks : {report['task_distribution']}")
    print(f"    output: {args.out}/ (records.jsonl, repos.jsonl, report.json/html"
          f"{', tasks.jsonl' if examples is not None else ''})")
    return 0


def cmd_stats(args):
    records = list(RecordStore(args.records).read())
    repos = []
    if args.repos and os.path.exists(args.repos):
        from .metadata import read_repo_records
        repos = list(read_repo_records(args.repos))
    report = build_report(records, repos)
    if args.html:
        write_html_report(report, args.html)
        print(f"[stats] wrote {args.html}")
    print(json.dumps({k: report[k] for k in
                      ("dataset_size", "language_distribution",
                       "license_distribution", "split_distribution")}, indent=2))
    return 0


def cmd_export(args):
    records = list(RecordStore(args.records).read())
    fmt = args.format
    if fmt == "raw":
        from .exporters import export_raw
        n = export_raw(records, args.out)
        print(f"[export] raw: {n} files -> {args.out}")
    elif fmt == "jsonl":
        from .exporters import export_records_by_split
        counts = export_records_by_split(records, args.out)
        print(f"[export] jsonl by split: {counts}")
    elif fmt == "parquet":
        from .exporters import export_parquet
        n = export_parquet(records, os.path.join(args.out, "corpus.parquet"))
        print(f"[export] parquet: {n} rows -> {args.out}/corpus.parquet")
    elif fmt == "rds":
        from .exporters import export_rds
        tok = None
        if args.tokenizer:
            from dataset import load_bpe_tokenizer
            tok = load_bpe_tokenizer(args.tokenizer)
        manifests = export_rds(records, args.out, tokenizer=tok, seq_len=args.seq_len)
        print(f"[export] rds splits: {list(manifests)} -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ryth-corpus",
                                description="Ryth Corpus engineering pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch sources into a staging dir")
    d.add_argument("--input", default=".", help="local input dir (local sources)")
    d.add_argument("--source-file", help="JSON list of source dicts")
    d.add_argument("--stage", default="corpus_stage")
    d.set_defaults(func=cmd_download)

    c = sub.add_parser("clean", help="ingest + clean a tree -> raw output")
    c.add_argument("--input", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--stage")
    c.add_argument("--config")
    c.set_defaults(func=cmd_clean)

    s = sub.add_parser("score", help="print per-repo quality scores")
    s.add_argument("--input", required=True)
    s.add_argument("--stage")
    s.add_argument("--config")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)

    b = sub.add_parser("build", help="run the full pipeline")
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--stage")
    b.add_argument("--config")
    b.add_argument("--min-quality", type=int)
    b.add_argument("--seed", type=int)
    b.add_argument("--tasks", action="store_true", help="also build task examples")
    b.set_defaults(func=cmd_build)

    st = sub.add_parser("stats", help="statistics from a records.jsonl")
    st.add_argument("--records", required=True)
    st.add_argument("--repos")
    st.add_argument("--html")
    st.set_defaults(func=cmd_stats)

    e = sub.add_parser("export", help="export records to a format")
    e.add_argument("--records", required=True)
    e.add_argument("--format", required=True,
                   choices=["raw", "jsonl", "parquet", "rds"])
    e.add_argument("--out", required=True)
    e.add_argument("--tokenizer", help="BPE tokenizer.json (rds export)")
    e.add_argument("--seq-len", type=int, default=1024)
    e.set_defaults(func=cmd_export)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
