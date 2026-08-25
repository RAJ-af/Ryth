"""W1 RDS packing — raw tree -> seq_len chunks, PARTS me todkar (resume-safe).

Har part apna RDEPipeline.run() karta hai (apna out_dir + `_DONE` marker),
phir merge_manifests() sab shards ko ek final data_dir me jodta hai jo
`RDSDataset(training)` seedha padh sakta hai (manifest["shards"] list native).
Common manifest fields (seq_len, dtype, vocab_size…) pehle part se aate hain,
`extra_meta` se override ho sakte hain.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


# ye keys pehle valid part manifest se copy hoti hain (RDSDataset inhe padhta hai)
_PROPAGATE_KEYS = ("format", "rds_version", "tokenizer_version",
                   "vocab_size", "seq_len", "dtype")


def merge_manifests(part_dirs: list[str], out_dir: str,
                    extra_meta: dict | None = None) -> dict:
    """Part manifests ko ek final manifest me jodo (shard files copy hote hain)."""
    os.makedirs(out_dir, exist_ok=True)
    shards: list[dict] = []
    stats_total: dict = {}
    seen_names: set[str] = set()
    propagated: dict = {}
    for pd in part_dirs:
        mf = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
        if not propagated:
            propagated = {k: mf[k] for k in _PROPAGATE_KEYS if k in mf}
        for sh in mf.get("shards", []):
            fname = sh["file"]
            while fname in seen_names:                  # naam takraaye toh prefix
                fname = f"{os.path.basename(pd)}_{fname}"
            seen_names.add(fname)
            shutil.copy2(os.path.join(pd, sh["file"]),
                         os.path.join(out_dir, fname))
            shards.append({**sh, "file": fname})
        for k, v in (mf.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                stats_total[k] = stats_total.get(k, 0) + v
    manifest = {**propagated, "n_shards": len(shards),
                "shards": shards, "stats": stats_total}
    manifest.update(extra_meta or {})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _discover_repo_dirs(raw_root: str) -> list[str]:
    """Stage tree ke andar repo-level dirs (jahan files hain)."""
    repos = []
    for dp, fns, _ in os.walk(raw_root):
        if any(not d.startswith("_") and os.path.isfile(os.path.join(dp, d))
               for d in fns):
            repos.append(dp)
    return sorted(repos)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="corpus_out/stage")
    p.add_argument("--tok", default="tok/tokenizer.json")
    p.add_argument("--parts", type=int, default=4,
                   help="kitne part-chunks me pipeline chalana hai (resume unit)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--out", default="rds_w1")
    a = p.parse_args(argv)

    from dataset import load_bpe_tokenizer
    from dataset.pipeline import RDEPipeline

    tok = load_bpe_tokenizer(a.tok)

    repos = _discover_repo_dirs(a.raw)
    if not repos:
        raise SystemExit(f"[pack] koi repo files nahi mili {a.raw!r} me")
    k = max(1, a.parts)
    groups = [repos[i::k] for i in range(k)]             # round-robin split

    part_dirs: list[str] = []
    for gi, group in enumerate(groups):
        if not group:
            continue
        part_in = os.path.join(a.out, f"in_{gi:02d}")
        part_out = os.path.join(a.out, f"part_{gi:02d}")
        done = os.path.join(part_out, "_DONE")
        os.makedirs(part_in, exist_ok=True)
        for r in group:                                  # symlink: disk bachao
            dst = os.path.join(
                part_in,
                os.path.basename(r) or f"repo_{len(os.listdir(part_in))}")
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(r), dst)
                except OSError:                          # FS symlink support nahi
                    shutil.copytree(r, dst)
        if not os.path.exists(done):
            pipe = RDEPipeline(tok)
            pipe.run(part_in, part_out, verbose=True)
            with open(done, "w", encoding="utf-8") as f:
                f.write("{}")
        else:
            print(f"[pack] part_{gi:02d} pehle se DONE — skip")
        part_dirs.append(part_out)

    mm = merge_manifests(part_dirs, os.path.join(a.out, "final"),
                         extra_meta={"seq_len": a.seq_len, "w1": True})
    n_chunks = sum(sh.get("chunks", 0) for sh in mm["shards"])
    print(f"[pack] shards={len(mm['shards'])} chunks={n_chunks} "
          f"-> {os.path.join(a.out, 'final')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
