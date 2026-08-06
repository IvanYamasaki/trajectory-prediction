"""Shared drift-detection helpers (robust-z smoothing, alarm metrics, ADWIN variant).

Canonical sources: drift_analise/phase2_grid_search.py (smooth_robz,
alarms_per_year, compute_snr) and the ADWIN_VARIANT env-var switch shared by
phase2/phase3/phase4_consolidate. No dependency on drift_analise modules.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common.constants import PERFECT_SNR


def smooth_robz(values: np.ndarray, w: int) -> np.ndarray:
    """Robust z-score local: (x - rolling_median) / rolling_MAD."""
    s = pd.Series(values)
    min_p = max(1, w // 2)
    med = s.rolling(w, min_periods=min_p).median()
    mad = (s - med).abs().rolling(w, min_periods=min_p).median()
    mad = mad.fillna(1.0).replace(0.0, 1.0)
    return ((s - med) / mad).fillna(0.0).values


def alarms_per_year(alarms: list[int], years: np.ndarray,
                    year_list: list[int]) -> dict[str, int]:
    if not alarms:
        return {f"n_{y}": 0 for y in year_list}
    arr = np.asarray(alarms, dtype=int)
    yr  = years[arr]
    return {f"n_{y}": int((yr == y).sum()) for y in year_list}


def compute_snr(n_post: int, n_2019_alarms: int,
                n_2019_total: int, n_post_total: int) -> float:
    if n_2019_alarms == 0:
        return PERFECT_SNR if n_post > 0 else 0.0
    if n_post_total == 0:
        return 0.0
    return float((n_post * n_2019_total) / (n_2019_alarms * n_post_total))


def adwin_variant_config() -> tuple[str, str]:
    """Le a env var ``ADWIN_VARIANT`` e devolve ``(variant, out_suffix)``.

    Reproduz o bloco repetido em phase2_grid_search.py:60-65,
    phase3_ensemble.py:48-53 e phase4_consolidate.py:57-62: variant e
    "lite" (default) ou "exact"; out_suffix e "" para lite e
    "_adwin_exact" para exact (ex.: f"mes9_phase2{out_suffix}").
    """
    variant = os.environ.get("ADWIN_VARIANT", "lite").lower()
    if variant not in ("lite", "exact"):
        raise ValueError(f"ADWIN_VARIANT must be 'lite' or 'exact', got {variant!r}")
    out_suffix = "" if variant == "lite" else "_adwin_exact"
    return variant, out_suffix
