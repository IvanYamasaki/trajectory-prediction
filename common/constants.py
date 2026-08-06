"""Shared experiment constants (proc_set/year mapping, detector configs, horizons).

Canonical sources: model_analise/ewc_finetune.py, drift_analise/phase1_smoothing.py,
drift_analise/phase3_ensemble.py, drift_analise/paper_figs.py.
"""
from __future__ import annotations

import os

# proc_set → year mapping (from retrain_at_breakpoints.py)
PROC_YEAR = {
    3: 2019, 4: 2019, 5: 2019, 6: 2019, 7: 2019, 8: 2019,
    9: 2021,10: 2021,11: 2021,12: 2021,13: 2021,14: 2021,
    15:2023,16:2023,17:2023,18:2023,19:2023,20:2023,
    21:2025,22:2025,23:2025,24:2025,25:2025,26:2025,
    27:2022,28:2022,29:2022,30:2022,31:2022,32:2022,
    33:2024,34:2024,35:2024,36:2024,37:2024,38:2024,
}
YEAR_ORDER   = [2019, 2021, 2022, 2023, 2024, 2025]
YEARS_ALL    = [2019, 2021, 2022, 2023, 2024, 2025]
YEARS_POST: list[int] = [2021, 2022, 2023, 2024, 2025]
BP_YEAR      = 2022  # breakpoint from Pelt / BOCPD

UNITS = 128

ROBZ_W = 200
RETRAIN_DEBOUNCE      = 5000
W_COINCIDENCE_DEFAULT = 200
PERFECT_SNR = 9999.0

HORIZON_30_15 = "30→15"
HORIZONS = ["30→15", "60→30"]

# Configs LITE incondicionais (independentes da env var ADWIN_VARIANT):
# usadas por scripts mes10/mes11/adwin_exact que hardcodam a variante lite.
ADWIN_S2S_LITE    = dict(delta=1e-7, min_window=600, cooldown=200,  max_window=2000)
ADWIN_KALMAN_LITE = dict(delta=1e-7, min_window=200, cooldown=1000, max_window=5000)

# ── detector configs (phase3_ensemble.py, controlled by ADWIN_VARIANT) ────────
ADWIN_VARIANT = os.environ.get("ADWIN_VARIANT", "lite").lower()
if ADWIN_VARIANT not in ("lite", "exact"):
    raise ValueError(f"ADWIN_VARIANT must be 'lite' or 'exact', got {ADWIN_VARIANT!r}")

# ADWIN S2S multi-year: espinha dorsal — cobre 2021-2025
#   * lite : delta=1e-7, mw=600, cd=200, xw=2000 → FAR=0.042/1k (paper)
#   * exact: delta=1e-7, mw=200, cd=1000, xw=2000 → FAR=0.125/1k (re-grid)
if ADWIN_VARIANT == "exact":
    ADWIN_S2S_MULTIYEAR = dict(delta=1e-7, min_window=200, cooldown=1000, max_window=2000)
else:
    ADWIN_S2S_MULTIYEAR = dict(delta=1e-7, min_window=600, cooldown=200,  max_window=2000)
# Kalman corroborator: OR-only — cobre os 5 anos
#   * lite : delta=1e-7, mw=200, cd=1000, xw=5000 → FAR=0.104/1k (paper)
#   * exact: delta=1e-7, mw=1000, cd=200, xw=2000 → FAR=0.1875/1k (re-grid)
if ADWIN_VARIANT == "exact":
    KALMAN_CORR = dict(delta=1e-7, min_window=1000, cooldown=200,  max_window=2000)
else:
    KALMAN_CORR = dict(delta=1e-7, min_window=200,  cooldown=1000, max_window=5000)
