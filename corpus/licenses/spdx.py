"""SPDX license classification tables.

Permissive vs copyleft vs unknown ka classification. Ye policy decide karta hai
ki koi repo corpus me aayega ya nahi (spec: MIT/Apache-2.0/BSD/ISC/MPL-2.0 allow;
GPL / unknown reject unless explicitly enabled).
"""

from __future__ import annotations

# Permissive (default-allowed with the standard policy).
PERMISSIVE = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear",
    "ISC", "0BSD", "Unlicense", "Zlib", "MPL-2.0", "BSL-1.0", "Python-2.0",
    "Apache-1.1", "PostgreSQL", "NCSA", "MIT-0",
}

# Weak/strong copyleft — rejected unless allow_copyleft is set.
COPYLEFT = {
    "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later", "GPL-2.0", "GPL-3.0", "LGPL-3.0",
    "AGPL-3.0", "EUPL-1.2", "CC-BY-SA-4.0",
}

UNKNOWN = "UNKNOWN"


def classify(spdx: str) -> str:
    """Return 'permissive' | 'copyleft' | 'unknown'."""
    if spdx in PERMISSIVE:
        return "permissive"
    if spdx in COPYLEFT:
        return "copyleft"
    return "unknown"


def is_allowed(spdx: str, *, allowed=None, allow_unknown=False,
               allow_copyleft=False) -> bool:
    """Policy check for a single SPDX id."""
    if allowed is not None and spdx in set(allowed):
        return True
    kind = classify(spdx)
    if kind == "permissive":
        # if an explicit allow-list is given, restrict to it
        return allowed is None or spdx in set(allowed)
    if kind == "copyleft":
        return allow_copyleft
    return allow_unknown
