"""Cleaner — apply every file-level rule + secret handling in one pass.

`Cleaner.inspect(path, data)` ek decision deta hai: keep karein ya drop (reason
ke saath), aur agar keep to (possibly transformed) text kya hai — notebook outputs
stripped, secrets redacted.

Config-driven (size caps, secret policy, notebook policy). Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import rules
from .secrets import has_secret, redact_secrets
from ..config import CorpusConfig


@dataclass
class CleanResult:
    keep: bool
    reason: str = ""              # non-empty when dropped
    text: str | None = None      # cleaned text when kept
    secrets_redacted: int = 0
    transformed: bool = False    # notebook-stripped / redacted


class Cleaner:
    def __init__(self, config: CorpusConfig | None = None):
        self.cfg = config or CorpusConfig()

    def inspect(self, path: str, data: bytes) -> CleanResult:
        cfg = self.cfg

        # --- structural drops (cheap, no decode) ---
        if rules.is_vendor_path(path):
            return CleanResult(False, "vendor")
        if rules.is_lockfile(path):
            return CleanResult(False, "lockfile")
        if rules.is_too_large(len(data), cfg.max_file_bytes):
            return CleanResult(False, "too_large")
        if rules.is_binary(data):
            return CleanResult(False, "binary")
        if rules.has_corrupted_encoding(data):
            return CleanResult(False, "encoding")

        text = data.decode("utf-8")
        transformed = False

        # --- notebook: strip outputs (keep code) ---
        if rules.is_notebook(path) and cfg.strip_notebook_outputs:
            stripped = rules.strip_notebook_outputs(text)
            transformed = stripped != text
            text = stripped

        # --- size floor (after any transform) ---
        if rules.is_too_small(len(text.encode("utf-8")), cfg.min_file_bytes):
            return CleanResult(False, "too_small")

        # --- minified / generated ---
        if rules.is_minified(path, text, cfg.max_line_length):
            return CleanResult(False, "minified")
        if rules.is_generated(text):
            return CleanResult(False, "generated")

        # --- secrets ---
        redacted = 0
        if has_secret(text):
            if cfg.drop_secret_files:
                return CleanResult(False, "secret")
            if cfg.strip_secrets:
                text, redacted = redact_secrets(text)
                transformed = transformed or redacted > 0

        return CleanResult(True, "", text=text, secrets_redacted=redacted,
                           transformed=transformed)
