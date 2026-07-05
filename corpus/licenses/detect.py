"""License detection — text/heuristics -> SPDX id.

Do entry points:
  * `detect_license_text(text)` — ek LICENSE/COPYING file ka text -> SPDX id
  * `detect_repo_license(files)` — {path: text} se repo ka license nikaalta hai
    (LICENSE files + SPDX headers dono dekhta hai)

Pure standard library (regex + keyword matching). Match na ho to `UNKNOWN`.
"""

from __future__ import annotations

import re

from .spdx import UNKNOWN

# SPDX-License-Identifier: <id>  (source-file header)
_SPDX_HEADER = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)")

_LICENSE_FILENAMES = (
    "license", "license.txt", "license.md", "license-mit", "copying",
    "copying.txt", "unlicense", "licence", "licence.txt",
)

# Ordered: more specific patterns first.
_RULES = [
    ("Apache-2.0", (r"apache license", r"version 2\.0")),
    ("MPL-2.0", (r"mozilla public license", r"version 2\.0")),
    ("AGPL-3.0-or-later", (r"gnu affero general public license", r"version 3")),
    ("LGPL-3.0-or-later", (r"gnu lesser general public license", r"version 3")),
    ("LGPL-2.1-or-later", (r"gnu lesser general public license", r"version 2\.1")),
    ("GPL-3.0-or-later", (r"gnu general public license", r"version 3")),
    ("GPL-2.0-or-later", (r"gnu general public license", r"version 2")),
    ("BSD-3-Clause", (r"redistribution and use", r"neither the name")),
    ("BSD-2-Clause", (r"redistribution and use in source and binary forms",)),
    ("ISC", (r"permission to use, copy, modify, and(/or)? distribute",)),
    ("Unlicense", (r"this is free and unencumbered software released into the public domain",)),
    ("0BSD", (r"permission to use, copy, modify, and/or distribute this software for any",)),
    ("Zlib", (r"this software is provided ['\"]as-is['\"]", r"altered source versions")),
    ("MIT", (r"permission is hereby granted, free of charge",)),
]


def detect_license_text(text: str) -> str:
    """Classify a license blob into an SPDX id (best-effort)."""
    if not text:
        return UNKNOWN
    low = " ".join(text.lower().split())        # collapse whitespace
    m = _SPDX_HEADER.search(text)
    if m:
        return m.group(1)
    for spdx, patterns in _RULES:
        if all(re.search(p, low) for p in patterns):
            return spdx
    return UNKNOWN


def detect_spdx_headers(text: str) -> str | None:
    """Return the SPDX id from a source-file header, if present."""
    m = _SPDX_HEADER.search(text or "")
    return m.group(1) if m else None


def detect_repo_license(files: dict) -> str:
    """`files`: {relative_path: text}. Prefer LICENSE/COPYING files, then any
    SPDX header found in source files. Returns an SPDX id or UNKNOWN."""
    # 1) dedicated license files
    for path, text in files.items():
        base = path.rsplit("/", 1)[-1].lower()
        if base in _LICENSE_FILENAMES or base.startswith("license"):
            spdx = detect_license_text(text)
            if spdx != UNKNOWN:
                return spdx
    # 2) SPDX headers in any file
    for text in files.values():
        spdx = detect_spdx_headers(text)
        if spdx:
            return spdx
    return UNKNOWN
