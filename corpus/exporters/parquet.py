"""Export records to Parquet (optional — needs `pyarrow`).

Agar pyarrow installed nahi hai to clear ImportError milega. Metadata columns +
optional content.
"""

from __future__ import annotations

import os


def pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def export_parquet(records: list, path: str, include_content: bool = True) -> int:
    """Write kept records to a Parquet file. Returns row count."""
    if not pyarrow_available():
        raise ImportError(
            "parquet export needs pyarrow: pip install 'ryth[corpus-parquet]'")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r.to_dict(include_content=include_content)
            for r in records if not r.drop_reason]
    if not rows:
        rows = []
    cols = {}
    keys = sorted({k for row in rows for k in row}) if rows else []
    for k in keys:
        cols[k] = [row.get(k) for row in rows]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    table = pa.table(cols) if cols else pa.table({"repository": []})
    pq.write_table(table, path)
    return len(rows)
