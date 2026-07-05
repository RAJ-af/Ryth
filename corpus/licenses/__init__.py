"""License detection + SPDX policy for the Ryth corpus."""

from .detect import (detect_license_text, detect_repo_license,
                     detect_spdx_headers)
from .spdx import COPYLEFT, PERMISSIVE, UNKNOWN, classify, is_allowed

__all__ = [
    "detect_license_text", "detect_repo_license", "detect_spdx_headers",
    "classify", "is_allowed", "PERMISSIVE", "COPYLEFT", "UNKNOWN",
]
