"""Source definitions — *where* corpus data comes from.

Har source ek declarative record hai (kaunsa downloader use hoga, license hint,
languages). Actual download `corpus/download/` karta hai. Isse sources ko config/
JSON me list kiya ja sakta hai bina code badle.

Source kinds:
  * huggingface — HF dataset (needs `datasets`, optional)
  * github      — GitHub repo (public zip via urllib)
  * http        — docs / markdown / tutorials over HTTP(S)
  * local       — a folder already on disk (always available, offline)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

KINDS = ("huggingface", "github", "http", "local")

# Categories from the spec (docs, unit tests, tutorials, example projects, …).
CATEGORIES = ("code", "docs", "tutorial", "tests", "examples", "markdown")


@dataclass
class Source:
    """A single corpus source."""
    id: str                               # unique id, e.g. "gh:pallets/click"
    kind: str                             # one of KINDS
    location: str                         # repo "owner/name" | HF name | URL | path
    license_hint: str = "UNKNOWN"         # SPDX hint; verified after download
    languages: tuple = ()                 # expected languages (informational)
    category: str = "code"                # one of CATEGORIES
    ref: str = "HEAD"                     # git ref / revision (github)
    subpath: str = ""                     # only ingest this subdir
    enabled: bool = True

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown source kind {self.kind!r} (want {KINDS})")
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        known = {f for f in cls.__dataclass_fields__}          # noqa: E1133
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SourceList:
    sources: list = field(default_factory=list)

    def add(self, source: Source) -> "SourceList":
        self.sources.append(source)
        return self

    def enabled(self) -> list:
        return [s for s in self.sources if s.enabled]

    def by_kind(self, kind: str) -> list:
        return [s for s in self.sources if s.kind == kind and s.enabled]

    def to_list(self) -> list:
        return [s.to_dict() for s in self.sources]

    @classmethod
    def from_list(cls, items) -> "SourceList":
        return cls([Source.from_dict(d) for d in items])


# A small curated set of well-known *permissive* projects, spanning the priority
# languages + docs/tests/tutorials. These are pointers only — nothing is fetched
# until a downloader runs, and the license is re-verified from the downloaded
# LICENSE file (the hint is never trusted blindly).
def default_sources() -> SourceList:
    S = Source
    return SourceList([
        S("gh:pallets/click", "github", "pallets/click", "BSD-3-Clause",
          ("python",), "code"),
        S("gh:psf/requests", "github", "psf/requests", "Apache-2.0",
          ("python",), "code"),
        S("gh:expressjs/express", "github", "expressjs/express", "MIT",
          ("javascript",), "code"),
        S("gh:microsoft/TypeScript-Node-Starter", "github",
          "microsoft/TypeScript-Node-Starter", "MIT", ("typescript",), "examples"),
        S("gh:BurntSushi/ripgrep", "github", "BurntSushi/ripgrep", "MIT",
          ("rust",), "code"),
        S("gh:spf13/cobra", "github", "spf13/cobra", "Apache-2.0",
          ("go",), "code"),
        S("gh:google/googletest", "github", "google/googletest", "BSD-3-Clause",
          ("cpp",), "tests"),
        S("gh:google/gson", "github", "google/gson", "Apache-2.0",
          ("java",), "code"),
        S("hf:code-search-net-python", "huggingface", "code_search_net", "MIT",
          ("python",), "code", enabled=False),   # opt-in (needs `datasets`)
        S("docs:python-tutorial", "http", "https://docs.python.org/3/tutorial/",
          "Python-2.0", ("markdown",), "tutorial", enabled=False),
    ])
