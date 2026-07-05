"""Export to Ryth RDS using the existing Ryth Data Engine (RDE) — unchanged.

Kept records ko ek temp raw-repo layout me likhte hain (`<root>/<repo>/<path>`),
phir `dataset.RDEPipeline` chalate hain — RDE ki koi line nahi badalti. Har split
ka alag RDS dataset banta hai (`<out>/<split>/`).

RDE optional dependency-free hai (ByteTokenizer fallback), par aap apna trained
BPE tokenizer bhi de sakte ho.
"""

from __future__ import annotations

import os
import tempfile

from .raw import export_raw


def export_rds(records: list, out_dir: str, *, tokenizer=None, seq_len: int = 1024,
               splits=("train", "validation", "test")) -> dict:
    """Build one RDS dataset per split via RDE. Returns {split: manifest}.

    `tokenizer`: an object with encode/decode/vocab_size (e.g. a loaded BPE
    tokenizer). None -> RDE's built-in ByteTokenizer.
    """
    # Imported here so `corpus` stays importable without the model/RDE torch deps
    # being required for non-RDS exports.
    from dataset import RDEConfig, RDEPipeline
    from dataset.tokenizer_adapter import ByteTokenizer

    tok = tokenizer or ByteTokenizer()
    manifests: dict = {}
    for split in splits:
        split_recs = [r for r in records if not r.drop_reason and r.split == split
                      and r.content is not None]
        if not split_recs:
            continue
        with tempfile.TemporaryDirectory(prefix=f"ryth-rds-{split}-") as tmp:
            export_raw(split_recs, tmp, by_split=False)
            cfg = RDEConfig(seq_len=seq_len, vocab_size=tok.vocab_size,
                            tokenizer_version=getattr(tok, "version", 1))
            dst = os.path.join(out_dir, split)
            manifests[split] = RDEPipeline(tok, cfg).run(tmp, dst, verbose=False)
    return manifests
