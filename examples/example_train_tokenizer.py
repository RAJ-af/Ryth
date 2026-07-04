"""Example: train the scratch byte-level BPE tokenizer on sample code.

Run:  python examples/example_train_tokenizer.py
"""

import os
import sys
import tempfile

# Make the repo importable when running from a checkout (not required if installed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS
from tests.sample_data import build_sample
from tokenizer.train import iter_text_files


def main():
    tmp = tempfile.mkdtemp(prefix="ryth_tok_")
    build_sample(tmp)                                   # writes sample .py files

    texts = list(iter_text_files(os.path.join(tmp, "**", "*.py")))
    print(f"corpus: {len(texts)} files, {sum(len(t) for t in texts):,} chars")

    tok = BPETokenizer()
    tok.train(texts, vocab_size=400, verbose=False)
    tok.add_special_tokens(DEFAULT_SPECIAL_TOKENS)

    out = os.path.join(tmp, "tokenizer.json")
    tok.save(out)
    print(f"vocab_size = {tok.vocab_size} | merges = {len(tok.merges)}")
    print(f"saved -> {out}")

    # a few learned multi-character tokens
    learned = [bytes(v).decode("utf-8", "replace")
               for k, v in tok.vocab.items() if k >= 256 and len(v) >= 3]
    print("sample learned tokens:", [repr(x) for x in learned[:12]])


if __name__ == "__main__":
    main()
