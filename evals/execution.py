"""Generated code ko alag subprocess me chalao (timeout ke saath).

⚠️ SECURITY NOTE: ye model-output ko LOCAL machine pe execute karta hai —
timeout ke alawa koi heavy sandboxing nahi hai (no network jail). Sirf apne
trusted box pe use karo; CI/cloud me container ke andar chalana behtar hai.
Pure standard library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Ek program-run ka poora result (`ok` = exit 0 aur timeout nahi)."""

    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_program(code: str, timeout_s: float = 10.0) -> ExecResult:
    """Python program likho, fresh interpreter me chalao, result wapas lao."""
    with tempfile.TemporaryDirectory(prefix="ryth_exec_") as td:
        path = os.path.join(td, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout_s)
            return ExecResult(ok=(p.returncode == 0), exit_code=p.returncode,
                              stdout=p.stdout[-4000:], stderr=p.stderr[-4000:],
                              timed_out=False)
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else ""
            return ExecResult(ok=False, exit_code=None,
                              stdout=(out or "")[-4000:],
                              stderr=f"TIMEOUT after {timeout_s}s",
                              timed_out=True)
        except OSError as e:
            # fork failure / resource limit — ek sample ka fail, sweep abort nahi
            return ExecResult(ok=False, exit_code=None, stdout="",
                              stderr=f"spawn failed: {e}", timed_out=False)
