"""`ryth-rde` command-line interface.

Subcommands:
    build      raw repos -> RDS dataset (full pipeline)
    inspect    dataset summary / one chunk (optionally decoded)
    verify     verify shard checksums
    stats      print dataset statistics
    manifest   print manifest + reproducibility lock
"""

from __future__ import annotations

import argparse
import json

import sys

from .config import RDEConfig
from .dataset import RDSDataset
from .pipeline import RDEPipeline, EmptyDatasetError
from .tokenizer_adapter import ByteTokenizer, load_bpe_tokenizer


def _load_tokenizer(path):
    return load_bpe_tokenizer(path) if path else ByteTokenizer()


# --------------------------------------------------------------------------- #
def _cmd_build(args):
    tok = _load_tokenizer(args.tokenizer)
    cfg = RDEConfig(seq_len=args.seq_len, vocab_size=tok.vocab_size,
                    shard_max_bytes=args.shard_mb * 1024 * 1024)
    try:
        RDEPipeline(tok, cfg).run(args.root, args.out, verbose=True,
                                  debug=args.debug)
    except EmptyDatasetError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_inspect(args):
    ds = RDSDataset(args.data_dir)
    m, s = ds.manifest, ds.manifest.get("stats", {})
    lock, pk = m.get("lock", {}), s.get("packing", {})
    print(f"RDS dataset: {args.data_dir}")
    print(f"  format         : RDS v{m['rds_version']} | tokenizer v{m['tokenizer_version']}")
    print(f"  dataset_version: {lock.get('dataset_version','?')}  model: {lock.get('model_version','?')}")
    print(f"  vocab/seq_len  : {m['vocab_size']} / {m['seq_len']}  dtype={m['dtype']}")
    print(f"  shards/chunks  : {m['n_shards']} / {len(ds)}")
    print(f"  languages      : {s.get('languages',{})}")
    print(f"  difficulty     : {s.get('difficulty_split',{})}")
    print(f"  packing eff.   : {pk.get('packing_efficiency_pct','?')}%  "
          f"padding {pk.get('padding_pct','?')}%")
    if args.chunk is not None:
        if not (0 <= args.chunk < len(ds)):
            print(f"  chunk {args.chunk} out of range (0..{len(ds)-1})")
        else:
            chunk, meta = ds[args.chunk], ds.meta(args.chunk)
            print(f"\n--- chunk[{args.chunk}] len={len(chunk)} ---")
            print(f"  meta: {json.dumps(meta, ensure_ascii=False)}")
            print(f"  ids : {list(chunk[:24])}{' ...' if len(chunk) > 24 else ''}")
            if args.decode:
                tok = _load_tokenizer(args.tokenizer)
                print("  decoded:\n" + "-" * 40)
                print(tok.decode(list(chunk))[:600])
    ds.close()
    return 0


def _cmd_verify(args):
    ds = RDSDataset(args.data_dir)
    ok = ds.verify()
    print(f"checksum verify: {'ALL OK' if ok else 'CORRUPTION DETECTED'} "
          f"({ds.manifest['n_shards']} shards)")
    ds.close()
    return 0 if ok else 1


def _cmd_stats(args):
    ds = RDSDataset(args.data_dir)
    print(json.dumps(ds.manifest.get("stats", {}), indent=2, ensure_ascii=False))
    ds.close()
    return 0


def _cmd_manifest(args):
    ds = RDSDataset(args.data_dir)
    m = dict(ds.manifest)
    if not args.full:
        m.pop("stats", None)            # lock + shard summary short view
    print(json.dumps(m, indent=2, ensure_ascii=False))
    ds.close()
    return 0


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(prog="ryth-rde",
                                 description="Ryth Data Engine CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build an RDS dataset from raw repos")
    p.add_argument("root",
                   help="input folder: <root>/<repo>/<files...> or <root>/<files...>")
    p.add_argument("out", help="output dataset folder")
    p.add_argument("--tokenizer", default=None,
                   help="BPE tokenizer.json (default: byte-level)")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="chunk length in tokens (default: 1024)")
    p.add_argument("--shard_mb", type=int, default=256, help="shard size (MB)")
    p.add_argument("--debug", action="store_true",
                   help="print every pipeline stage (discovery, cleaner, "
                        "validator, language, encoder, chunker)")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("inspect", help="inspect a dataset / chunk")
    p.add_argument("data_dir")
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--decode", action="store_true")
    p.add_argument("--tokenizer", default=None)
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("verify", help="verify shard checksums")
    p.add_argument("data_dir")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("stats", help="print dataset statistics")
    p.add_argument("data_dir")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("manifest", help="print manifest + reproducibility lock")
    p.add_argument("data_dir")
    p.add_argument("--full", action="store_true", help="include full stats block")
    p.set_defaults(func=_cmd_manifest)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
