"""Corpus source probe — bulk download se PEHLE chalao (measure-first).

configs/w1_sources.json ke hf entries me se `--subset` language wala source
uthata hai (single source of truth — build isi config se chalta hai) aur
pehli `limit` rows stream karke batata hai: kaunse columns hain, license
values ka histogram kya hai. Isi se license-policy decide hoti hai.

GATED primary par fallbacks khud try hote hain (downloader jaisa hi chain).
NETWORK TOOL — tests sirf fake-datasets inject karke chalate hain.
"""

from __future__ import annotations

import json
import os

# --- make the repo importable when run from a clone (scripts/..) ------------
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
                    "isc", "mpl-2.0"}

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "w1_sources.json")


def _entry_for(subset: str, config: str) -> dict:
    entries = json.load(open(config, encoding="utf-8"))
    for e in entries:
        if e.get("kind") == "huggingface" and subset in (e.get("languages")
                                                         or []):
            return e
    raise SystemExit(f"[probe] '{subset}' language ka huggingface source "
                     f"{config} me nahi mila")


def w1_probe_stack(subset: str, limit: int = 200,
                   config: str = _DEFAULT_CONFIG) -> dict:
    from corpus.download.huggingface import open_streaming

    e = dict(_entry_for(subset, config))
    ref = e.pop("ref", None)                      # Source.ref == HF revision
    if ref and ref != "HEAD":
        e.setdefault("revision", ref)
    ds, served = open_streaming(e)

    cols: list[str] = []
    hist: dict[str, int] = {}
    rows = 0
    for ex in ds:
        if not cols:
            cols = sorted(ex.keys())
        lic = ex.get("license") or "unknown"
        hist[lic] = hist.get(lic, 0) + 1
        rows += 1
        if rows >= limit:
            break
    return {"subset": subset, "rows": rows, "columns": cols,
            "license_histogram": hist, "served_location": served}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="e.g. python | c")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="w1_sources.json — build isi se chalta hai")
    a = p.parse_args(argv)
    out = w1_probe_stack(a.subset, limit=a.limit, config=a.config)
    print(json.dumps(out, indent=2))
    # streaming-iterator threads ka shutdown race rc=134 de sakta tha
    # (JSON ke BAAD abort) — clean rc ke liye controlled hard-exit
    from corpus.download.huggingface import teardown_safe_exit

    teardown_safe_exit(0)
    return 0  # pragma: no cover (upar se _exit ho jata hai)


if __name__ == "__main__":
    raise SystemExit(main())
