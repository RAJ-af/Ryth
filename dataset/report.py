"""Module 10 (v1.1) — Validation Report.

Pipeline ke end me ek human-readable report auto-generate karta hai (JSON + HTML).
Isse ek nazar me pata chal jaata hai ki dataset healthy hai ya nahi:

    Files processed • Duplicates removed • Generated removed • Encoding failures
    Average quality/functions/imports/comments • Longest sequence
    Packing efficiency • Padding % • Compression ratio • Difficulty split
"""

from __future__ import annotations

import json
import os


def build_report(manifest: dict, cleaner_stats: dict) -> dict:
    """Manifest.stats + cleaner drops ko ek flat report dict me merge karo."""
    s = manifest["stats"]
    pk = s["packing"]
    return {
        "dataset_version": manifest.get("lock", {}).get("dataset_version"),
        "created": manifest.get("lock", {}).get("creation_time"),
        "files_processed": s["total_files_seen"],
        "files_kept": s["files_kept"],
        "fim_docs": s["fim_docs"],
        "duplicates_removed": cleaner_stats.get("duplicate", 0),
        "generated_removed": cleaner_stats.get("generated", 0),
        "vendor_removed": cleaner_stats.get("vendor", 0),
        "binary_removed": cleaner_stats.get("binary", 0),
        "empty_removed": cleaner_stats.get("empty", 0),
        "encoding_failures": cleaner_stats.get("non_utf8", 0),
        "validate_drops": s["file_drop_reasons"],
        "avg_quality": s["avg_quality"],
        "avg_functions": s["avg_functions"],
        "avg_imports": s["avg_imports"],
        "avg_comments": s["avg_comments"],
        "longest_sequence": s["longest_sequence"],
        "languages": s["languages"],
        "difficulty_split": s["difficulty_split"],
        "total_tokens": s["total_tokens"],
        "total_chunks": s["total_chunks"],
        "chunk_duplicate_pct": s["chunk_duplicate_pct"],
        "packing_efficiency_pct": pk["packing_efficiency_pct"],
        "padding_pct": pk["padding_pct"],
        "avg_context_usage": pk["avg_context_usage"],
        "seq_len": pk["seq_len"],
        "compression_ratio": s["compression_ratio"],
        "n_shards": manifest["n_shards"],
    }


def _row(label, value):
    return f"    <tr><td>{label}</td><td>{value}</td></tr>\n"


def _health_flags(r: dict) -> list[tuple[str, str]]:
    """Simple health checks -> (level, message). level: ok | warn | bad."""
    flags = []
    eff = r["packing_efficiency_pct"]
    if eff >= 95:
        flags.append(("ok", f"Packing efficiency {eff}% — excellent"))
    elif eff >= 80:
        flags.append(("warn", f"Packing efficiency {eff}% — thoda padding waste"))
    else:
        flags.append(("bad", f"Packing efficiency {eff}% — bahut padding waste!"))
    if r["files_kept"] == 0:
        flags.append(("bad", "Koi file nahi bachi — filters bahut aggressive hain"))
    if r["chunk_duplicate_pct"] > 30:
        flags.append(("warn", f"Chunk duplicate {r['chunk_duplicate_pct']}% — high"))
    if r["avg_quality"] < 40:
        flags.append(("warn", f"Average quality {r['avg_quality']} — low"))
    return flags


def write_report(report: dict, out_dir: str) -> tuple[str, str]:
    """report.json + report.html likho, dono ke paths return karo."""
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    flags = _health_flags(report)
    flag_html = "\n".join(
        f'    <li class="{lvl}">{msg}</li>' for lvl, msg in flags)
    rows = "".join(_row(k.replace("_", " ").title(), v)
                   for k, v in report.items()
                   if not isinstance(v, dict))
    dict_sections = ""
    for k, v in report.items():
        if isinstance(v, dict):
            inner = "".join(_row(ik, iv) for ik, iv in v.items())
            dict_sections += (f"  <h3>{k.replace('_',' ').title()}</h3>\n"
                              f"  <table>\n{inner}  </table>\n")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RDE Validation Report</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;color:#1a1a2e}}
 h1{{border-bottom:3px solid #6c5ce7;padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}
 td{{border:1px solid #ddd;padding:.4rem .7rem}} td:first-child{{font-weight:600;width:55%}}
 ul{{list-style:none;padding:0}} li{{padding:.5rem .8rem;border-radius:6px;margin:.3rem 0}}
 .ok{{background:#d4f8e8}} .warn{{background:#fff3cd}} .bad{{background:#ffd6d6}}
</style></head><body>
 <h1>Ryth Data Engine — Validation Report</h1>
 <h3>Health</h3>
 <ul>
{flag_html}
 </ul>
 <h3>Summary</h3>
 <table>
{rows} </table>
{dict_sections}
</body></html>"""
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return json_path, html_path
