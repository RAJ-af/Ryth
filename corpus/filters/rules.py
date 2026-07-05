"""Record-level filters: license policy, size, language.

Cleaner file-content dekhta hai; ye filters metadata-level policy lagate hain
(license allowed hai? language priority me hai? size range me hai?). Har filter
`(keep: bool, reason: str)` deta hai taaki drop reasons report ho sakein.
"""

from __future__ import annotations

from ..config import CorpusConfig
from ..licenses import is_allowed
from .language import keep_language


class LicenseFilter:
    def __init__(self, config: CorpusConfig | None = None):
        self.cfg = config or CorpusConfig()

    def allows(self, spdx: str) -> bool:
        return is_allowed(spdx, allowed=self.cfg.allowed_licenses,
                          allow_unknown=self.cfg.allow_unknown_licenses,
                          allow_copyleft=self.cfg.allow_copyleft)

    def check(self, record):
        return (True, "") if self.allows(record.license) else (False, "license")


class LanguageFilter:
    def __init__(self, allowed=None):
        self.allowed = allowed

    def check(self, record):
        return (True, "") if keep_language(record, self.allowed) else (False, "language")


class SizeFilter:
    def __init__(self, config: CorpusConfig | None = None):
        self.cfg = config or CorpusConfig()

    def check(self, record):
        if record.size > self.cfg.max_file_bytes:
            return (False, "too_large")
        if record.size < self.cfg.min_file_bytes:
            return (False, "too_small")
        return (True, "")
