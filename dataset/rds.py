"""RDS binary format — reader + writer (single shard file).

Covers modules:
  10 Metadata   12 Compression (uint16)   14 Memory-mapping   16 Random access
  17 Versioning 18 Checksum

File layout (little-endian):

    [HEADER  64 bytes]
        magic "RDS1" | version | tok_version | vocab_size | seq_len
        dtype_flag(0=uint16,1=uint32) | flags | n_chunks
        data_offset | index_offset | meta_offset | footer_offset
    [DATA]      har chunk ke token ids, back-to-back (uint16/uint32)
    [INDEX]     n_chunks x (data_offset: uint64, length_tokens: uint32)
    [METADATA]  uint64 length + UTF-8 JSON (per-chunk meta list) — Module 10
    [FOOTER]    sha256 digest (32 bytes) + magic "RDSE"  — Module 18

Header offsets ki wajah se reader mmap karke seedha kisi bhi chunk par jump kar
sakta hai (Module 16), bina poori file padhe.

VERSIONING (backward compatibility):
  Header ka `version` field format-version batata hai. `RDSReader` is version par
  dispatch karta hai (`_PARSERS` registry). Naya format (v2/v3) add karte waqt bas
  ek naya parser register karo — purane shards padhte rehte hain. "Kabhi mat maano
  ki format nahi badlega." Isliye reader hamesha version-aware hai.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from array import array

MAGIC = b"RDS1"
FOOTER_MAGIC = b"RDSE"
HEADER_SIZE = 64
HEADER_FMT = "<4sHHIIBBIQQQQ"          # 54 bytes, header padded to 64
INDEX_FMT = "<QI"                       # (offset, length) = 12 bytes
INDEX_SIZE = struct.calcsize(INDEX_FMT)


class ShardWriter:
    """Ek RDS shard ko streaming tarike se likhता hai (data disk par, index RAM me)."""

    def __init__(self, path: str, *, version: int, tok_version: int,
                 vocab_size: int, seq_len: int, dtype_flag: int):
        self.path = path
        self.version = version
        self.tok_version = tok_version
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dtype_flag = dtype_flag
        self.typecode = "H" if dtype_flag == 0 else "I"
        self._f = open(path, "wb+")            # wb+ => checksum ke liye read bhi kar saken
        self._f.write(b"\x00" * HEADER_SIZE)        # placeholder header
        self._index: list[tuple[int, int]] = []
        self._meta: list[dict] = []
        self.n_tokens = 0

    def add_chunk(self, token_ids, meta: dict) -> None:
        arr = array(self.typecode, token_ids)
        offset = self._f.tell()
        self._f.write(arr.tobytes())
        self._index.append((offset, len(token_ids)))
        self._meta.append(meta)
        self.n_tokens += len(token_ids)

    @property
    def n_chunks(self) -> int:
        return len(self._index)

    @property
    def data_bytes(self) -> int:
        return self.n_tokens * (2 if self.dtype_flag == 0 else 4)

    def close(self) -> None:
        f = self._f
        data_offset = HEADER_SIZE

        # INDEX
        index_offset = f.tell()
        for off, length in self._index:
            f.write(struct.pack(INDEX_FMT, off, length))

        # METADATA (Module 10)
        meta_offset = f.tell()
        mjson = json.dumps(self._meta, ensure_ascii=False).encode("utf-8")
        f.write(struct.pack("<Q", len(mjson)))
        f.write(mjson)

        footer_offset = f.tell()

        # patch header with real offsets
        header = struct.pack(
            HEADER_FMT, MAGIC, self.version, self.tok_version, self.vocab_size,
            self.seq_len, self.dtype_flag, 0, self.n_chunks,
            data_offset, index_offset, meta_offset, footer_offset,
        ).ljust(HEADER_SIZE, b"\x00")
        f.seek(0)
        f.write(header)
        f.flush()

        # FOOTER checksum over [0, footer_offset)  (Module 18)
        f.seek(0)
        h = hashlib.sha256()
        remaining = footer_offset
        while remaining > 0:
            b = f.read(min(1 << 20, remaining))
            if not b:
                break
            h.update(b)
            remaining -= len(b)
        f.seek(footer_offset)
        f.write(h.digest())
        f.write(FOOTER_MAGIC)
        f.close()


class RDSReader:
    """mmap-based random-access reader. RAM me poori file load nahi hoti."""

    def __init__(self, path: str):
        self.path = path
        self._fd = open(path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()
        self._load_index()

    def _parse_header(self):
        # magic ke baad version padho, phir us version ka parser chuno.
        magic = bytes(self._mm[:4])
        if magic[:3] != MAGIC[:3]:                 # "RDS" prefix common rahega
            raise ValueError(f"not an RDS file (bad magic {magic!r})")
        (version,) = struct.unpack_from("<H", self._mm, 4)
        self.version = version
        parser = _PARSERS.get(version)
        if parser is None:
            raise ValueError(
                f"RDS format version {version} is newer than this reader supports "
                f"(known: {sorted(_PARSERS)}). Reader upgrade karo.")
        parser(self)

    def _parse_v1(self):
        """v1 header parser. Naye versions ke liye _parse_v2/_parse_v3 add karke
        _PARSERS me register karo — backward compatibility bani rahegi."""
        (magic, self.version, self.tok_version, self.vocab_size, self.seq_len,
         self.dtype_flag, self.flags, self.n_chunks, self.data_offset,
         self.index_offset, self.meta_offset, self.footer_offset) = \
            struct.unpack(HEADER_FMT, self._mm[:struct.calcsize(HEADER_FMT)])
        self.typecode = "H" if self.dtype_flag == 0 else "I"
        self.itemsize = 2 if self.dtype_flag == 0 else 4

    def _load_index(self):
        self._index = []
        base = self.index_offset
        for i in range(self.n_chunks):
            off, length = struct.unpack_from(INDEX_FMT, self._mm, base + i * INDEX_SIZE)
            self._index.append((off, length))

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i: int) -> array:
        """Module 16 — kisi bhi chunk ko O(1) me load karo (sequential scan nahi)."""
        if i < 0:
            i += self.n_chunks
        off, length = self._index[i]
        nbytes = length * self.itemsize
        arr = array(self.typecode)
        arr.frombytes(self._mm[off:off + nbytes])
        return arr

    def metadata(self, i: int | None = None):
        """Per-chunk metadata (Module 10). i=None => saari list."""
        (mlen,) = struct.unpack_from("<Q", self._mm, self.meta_offset)
        raw = self._mm[self.meta_offset + 8: self.meta_offset + 8 + mlen]
        metas = json.loads(raw.decode("utf-8"))
        return metas if i is None else metas[i]

    def verify_checksum(self) -> bool:
        """Module 18 — shard corruption detect karo."""
        h = hashlib.sha256(self._mm[:self.footer_offset]).digest()
        stored = self._mm[self.footer_offset:self.footer_offset + 32]
        magic = self._mm[self.footer_offset + 32:self.footer_offset + 36]
        return h == stored and magic == FOOTER_MAGIC

    def close(self):
        self._mm.close()
        self._fd.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# Version -> header-parser registry. Naya format add karne par yahan register karo.
# (v2/v3 parsers future me; abhi sirf v1 supported.)
_PARSERS = {
    1: RDSReader._parse_v1,
}
SUPPORTED_VERSIONS = tuple(sorted(_PARSERS))
