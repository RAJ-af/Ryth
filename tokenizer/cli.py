"""`ryth-tokenizer` command-line interface.

Subcommands:
    train    corpus se ek BPE tokenizer train karo
    encode   text ko token ids me badlo
    decode   token ids ko wapas text me badlo
"""

from __future__ import annotations

import argparse
import sys

from .bpe import BPETokenizer
from .train import iter_jsonl_corpus, iter_text_files, train_tokenizer


def _cmd_train(args):
    if args.files:
        texts = list(iter_text_files(args.files))
    else:
        texts = list(iter_jsonl_corpus(args.data))
    if not texts:
        print("error: no corpus text found "
              "(--data <jsonl dir> ya --files '<glob>' do)", file=sys.stderr)
        return 1
    total = sum(len(t) for t in texts)
    print(f"corpus: {len(texts)} documents, {total:,} chars")
    train_tokenizer(texts, vocab_size=args.vocab, out_dir=args.out, verbose=True)
    return 0


def _cmd_encode(args):
    tok = BPETokenizer.load(args.tokenizer)
    text = args.text if args.text is not None else _read_input(args.input)
    ids = tok.encode(text)
    print(f"tokens: {len(ids)}  ({len(text)/max(1,len(ids)):.2f} chars/token)")
    print(ids)
    return 0


def _cmd_decode(args):
    tok = BPETokenizer.load(args.tokenizer)
    ids = [int(x) for x in args.ids.replace(",", " ").split()]
    print(tok.decode(ids))
    return 0


def _read_input(path):
    if not path or path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_parser():
    ap = argparse.ArgumentParser(prog="ryth-tokenizer",
                                 description="Ryth scratch BPE tokenizer CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train a BPE tokenizer on a corpus")
    p.add_argument("--data", default="data", help="jsonl corpus folder")
    p.add_argument("--files", default=None,
                   help="glob for plain text/code files (e.g. 'src/**/*.py')")
    p.add_argument("--vocab", type=int, default=1024, help="target vocab size")
    p.add_argument("--out", default="tokenizer_out", help="output folder")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("encode", help="encode text to token ids")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    p.add_argument("--text", default=None, help="text to encode")
    p.add_argument("--input", default=None, help="file to encode ('-' for stdin)")
    p.set_defaults(func=_cmd_encode)

    p = sub.add_parser("decode", help="decode token ids to text")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    p.add_argument("--ids", required=True, help="space/comma separated token ids")
    p.set_defaults(func=_cmd_decode)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
