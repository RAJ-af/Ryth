"""Ryth Data Engine (RDE) — a data operating system for training Ryth.

Ye sirf encoder nahi; poora pipeline hai: clean -> validate -> score -> analyze
-> tokenize -> encode -> chunk -> curriculum -> dedup -> shard -> RDS format,
with mmap streaming, random access, checksums, versioning, and statistics.

Module map (spec -> file):
   1 Cleaner            cleaner.py
   2 Validator          validator.py
   3 Quality Analyzer   quality.py
   4 Repository         repository.py
   5 Language Detector  language.py
   6 Tokenizer          tokenizer_adapter.py  (plug your scratch BPE)
   7 Encoder            encoder.py
   8 Chunk Builder      chunker.py
   9 Curriculum         curriculum.py
  10 Metadata           record.py + rds.py
  11 Deduplication      dedup.py
  12 Compression        rds.py (uint16)
  13 Streaming          dataset.py
  14 Memory Mapping     dataset.py / rds.py (mmap)
  15 Sharding           sharding.py
  16 Random Access      rds.py (RDSReader.__getitem__)
  17 Versioning         rds.py header + manifest
  18 Checksum           rds.py footer
  19 Statistics         stats.py
  20 Future             record.meta / reserved header space
"""

from .config import RDEConfig
from .dataset import RDSDataset
from .lock import build_lock, verify_lock, tokenizer_hash
from .pipeline import RDEPipeline, EmptyDatasetError
from .rds import RDSReader, ShardWriter, SUPPORTED_VERSIONS
from .tokenizer_adapter import ByteTokenizer, load_bpe_tokenizer

__all__ = [
    "RDEConfig", "RDEPipeline", "EmptyDatasetError",
    "RDSDataset", "RDSReader", "ShardWriter",
    "ByteTokenizer", "load_bpe_tokenizer", "SUPPORTED_VERSIONS",
    "build_lock", "verify_lock", "tokenizer_hash",
]
