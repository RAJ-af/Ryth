"""Export kept records to raw folders on disk.

Layout: `<out>/<split>/<repository>/<path>`. Ye woh input layout hai jo Ryth Data
Engine (RDE) + tokenizer training seedhe consume kar sakte hain.
"""

from __future__ import annotations

import os


def _safe(rel: str) -> str:
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts)


def export_raw(records: list, out_dir: str, by_split: bool = True) -> int:
    """Write every kept record's content to a file. Returns files written."""
    n = 0
    for r in records:
        if r.drop_reason or r.content is None:
            continue
        rel = f"{r.split}/{r.repository}/{r.path}" if by_split else f"{r.repository}/{r.path}"
        dest = os.path.join(out_dir, _safe(rel))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(r.content)
        n += 1
    return n
