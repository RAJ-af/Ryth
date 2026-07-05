"""Corpus reports — statistics + HTML/JSON output.

Spec reports: language distribution, license distribution, duplicate statistics,
quality histogram, repository rankings, dataset size, task distribution. Sab kuch
kept records + repo records + drop counts + task examples se compute hota hai.
Pure standard library (koi templating lib nahi).
"""

from __future__ import annotations

import html
import json
import os


def _count_by(records, key) -> dict:
    out: dict = {}
    for r in records:
        k = getattr(r, key)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def quality_histogram(repos, bins=10) -> dict:
    hist = {f"{i*10}-{i*10+10}": 0 for i in range(bins)}
    labels = list(hist)
    for rp in repos:
        idx = min(bins - 1, int(rp.quality_score // (100 / bins)))
        hist[labels[idx]] += 1
    return hist


def build_report(records, repos, *, examples=None, drops=None,
                 config=None) -> dict:
    """Assemble the full stats dict. `records` = kept FileRecords."""
    kept = [r for r in records if not r.drop_reason]
    total_bytes = sum(r.size for r in kept)
    splits = _count_by(kept, "split")

    rankings = sorted(
        ({"repository": rp.repository, "quality_score": rp.quality_score,
          "n_files": rp.n_files, "license": rp.license, "split": rp.split}
         for rp in repos),
        key=lambda d: (-d["quality_score"], d["repository"]))

    dup = dict(drops or {})
    return {
        "name": getattr(config, "name", "ryth-corpus"),
        "version": getattr(config, "version", "1.0.0"),
        "dataset_size": {
            "files": len(kept),
            "repositories": len(repos),
            "bytes": total_bytes,
            "megabytes": round(total_bytes / 1e6, 3),
        },
        "language_distribution": _count_by(kept, "language"),
        "license_distribution": _count_by(kept, "license"),
        "split_distribution": splits,
        "duplicate_statistics": {
            "duplicate_file": dup.get("duplicate_file", 0),
            "near_duplicate": dup.get("near_duplicate", 0),
            "duplicate_repo": dup.get("duplicate_repo", 0),
        },
        "drop_reasons": dup,
        "quality_histogram": quality_histogram(repos),
        "repository_rankings": rankings[:100],
        "task_distribution": _task_dist(examples) if examples else {},
    }


def _task_dist(examples) -> dict:
    out: dict = {}
    for e in examples:
        out[e["task"]] = out.get(e["task"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def write_json_report(report: dict, path: str) -> str:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def _table(title, mapping) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in mapping.items())
    return (f"<h2>{html.escape(title)}</h2><table><thead>"
            f"<tr><th>key</th><th>count</th></tr></thead><tbody>{rows}</tbody></table>")


def _bars(title, mapping) -> str:
    mx = max(mapping.values(), default=1) or 1
    rows = ""
    for k, v in mapping.items():
        w = int(100 * v / mx)
        rows += (f"<div class='bar'><span class='lbl'>{html.escape(str(k))}</span>"
                 f"<span class='track'><span class='fill' style='width:{w}%'></span></span>"
                 f"<span class='val'>{v}</span></div>")
    return f"<h2>{html.escape(title)}</h2>{rows}"


def write_html_report(report: dict, path: str) -> str:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    ds = report["dataset_size"]
    rank_rows = "".join(
        f"<tr><td>{i+1}</td><td>{html.escape(r['repository'])}</td>"
        f"<td>{r['quality_score']}</td><td>{r['n_files']}</td>"
        f"<td>{html.escape(r['license'])}</td><td>{html.escape(r['split'])}</td></tr>"
        for i, r in enumerate(report["repository_rankings"][:50]))
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(report['name'])} report</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a2e;max-width:960px}}
h1{{margin-bottom:0}} .sub{{color:#666}} h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e3e3e3;padding:.35rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f6f6fb}} .cards{{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem}}
.card{{background:#f6f6fb;border-radius:10px;padding:1rem 1.4rem;min-width:130px}}
.card .n{{font-size:1.6rem;font-weight:700}} .card .k{{color:#666;font-size:.8rem}}
.bar{{display:flex;align-items:center;gap:.5rem;margin:.2rem 0;font-size:.85rem}}
.bar .lbl{{width:150px}} .bar .track{{flex:1;background:#eee;border-radius:4px;height:12px;overflow:hidden}}
.bar .fill{{display:block;height:100%;background:#5b5be6}} .bar .val{{width:60px;text-align:right}}
</style></head><body>
<h1>{html.escape(report['name'])} <span class="sub">v{html.escape(str(report['version']))}</span></h1>
<div class="cards">
<div class="card"><div class="n">{ds['files']:,}</div><div class="k">files</div></div>
<div class="card"><div class="n">{ds['repositories']:,}</div><div class="k">repositories</div></div>
<div class="card"><div class="n">{ds['megabytes']:,}</div><div class="k">MB</div></div>
</div>
{_bars('Language distribution', report['language_distribution'])}
{_bars('License distribution', report['license_distribution'])}
{_table('Split distribution', report['split_distribution'])}
{_table('Duplicate statistics', report['duplicate_statistics'])}
{_bars('Quality histogram', report['quality_histogram'])}
{_bars('Task distribution', report['task_distribution']) if report['task_distribution'] else ''}
<h2>Repository rankings (top 50)</h2>
<table><thead><tr><th>#</th><th>repository</th><th>quality</th><th>files</th>
<th>license</th><th>split</th></tr></thead><tbody>{rank_rows}</tbody></table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path
