"""Compatibility shim: canonical definitions live in ``common.paths``.

Kept because notebooks import this module both as ``drift_output_paths``
(with sys.path pointing at drift_analise/) and as ``drift_analise.drift_output_paths``.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import common.paths  # noqa: F401
except ImportError:
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

from common.paths import *  # noqa: F401,F403
from common.paths import (  # noqa: F401
    project_root,
    DRIFT_RESULTS,
    CH01,
    CH02,
    CH03,
    CH04,
    COVARIATE_SHIFT_OUT,
    DRIFT_DECOMPOSITION_CSV,
    prepend_matching_venv_site_packages,
    ensure_chapter_dirs,
    ensure_drift_decomposition_csv,
)

__all__ = [
    "project_root",
    "DRIFT_RESULTS",
    "CH01",
    "CH02",
    "CH03",
    "CH04",
    "COVARIATE_SHIFT_OUT",
    "DRIFT_DECOMPOSITION_CSV",
    "prepend_matching_venv_site_packages",
    "ensure_chapter_dirs",
    "ensure_drift_decomposition_csv",
]
