"""The Stack-dedup schema probe — bulk download se PEHLE chalao (measure-first).

Ek subset ki pehle `limit` rows stream karke batata hai: kaunse columns hain,
license values ka histogram kya hai. Isi se license-policy decide hoti hai
(allowlist ya unknown-keep — runbook troubleshooting me documented).
NETWORK TOOL — tests sirf fake-datasets inject karke chalate hain.
"""

from __future__ import annotations


ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
                    "isc", "mpl-2.0"}


def w1_probe_stack(subset: str, limit: int = 200) -> dict:
    import datasets  # optional dep — probe/network machine pe hi zaroori

    ds = datasets.load_dataset("bigcode/the-stack-dedup", split="train",
                               streaming=True, data_dir=f"data/{subset}")
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
            "license_histogram": hist}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="e.g. python | c")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args(argv)
    out = w1_probe_stack(a.subset, limit=a.limit)
    import json

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
