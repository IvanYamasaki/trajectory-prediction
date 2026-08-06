"""Targeted comparison: ADWINLite vs ADWINExact (river) on the two
configurations that the original grid selected as finalists.

Goal: produce the canonical numbers that the checkpoint cites
(FAR_2019, year_coverage, n_alarms per year) under both detectors so the
methodological revision has actionable data even if the full 174-config
re-grid is still running.

Finalist configs (from Mes 9 Fase 2 PHASE2_REPORT.md):
  * Seq2Seq best: delta=1e-7, min_window=600, cooldown=200, max_window=2000
  * Kalman  best: delta=1e-7, min_window=200, cooldown=1000, max_window=5000
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.constants import YEARS_ALL, ADWIN_S2S_LITE, ADWIN_KALMAN_LITE
from common.detection import smooth_robz
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, ADWINExact, run_detector, load_errors, build_stream,
)

FINALISTS = {
    "Seq2Seq": dict(ADWIN_S2S_LITE),
    "Kalman":  dict(ADWIN_KALMAN_LITE),
}
OUT = PROJECT_ROOT / "Relas" / "results" / "drift" / "phase2_adwin_exact_finalists.csv"


def alarms_by_year(alarms: list[int], yrs: np.ndarray) -> dict[int, int]:
    if not alarms:
        return {y: 0 for y in YEARS_ALL}
    yr = yrs[np.asarray(alarms)]
    return {y: int((yr == y).sum()) for y in YEARS_ALL}


def main() -> None:
    df = load_errors()
    rows: list[dict] = []
    for model, cfg in FINALISTS.items():
        stream = build_stream(df, model=model)
        raw_vals  = stream["ade_traj"].astype(float).values
        year_vals = stream["year"].fillna(0).astype(int).values
        n_2019 = int((year_vals == 2019).sum())
        n_post = int((year_vals >= 2021).sum())
        smoothed = smooth_robz(raw_vals, 200)

        for variant, cls in [("lite", ADWINLite), ("exact", ADWINExact)]:
            t0 = time.time()
            det = cls(**cfg)
            al = run_detector(det, smoothed)
            dur = time.time() - t0
            by_yr = alarms_by_year(al, year_vals)
            n_a_2019 = by_yr[2019]
            n_a_post = sum(by_yr[y] for y in [2021,2022,2023,2024,2025])
            far19 = 1000.0 * n_a_2019 / max(1, n_2019)
            ycov  = sum(1 for y in [2021,2022,2023,2024,2025] if by_yr[y] > 0)
            row = dict(model=model, variant=variant, **cfg,
                       n_alarms=len(al), n_alarms_post=n_a_post,
                       FAR_2019_per1k=round(far19, 4), year_coverage=ycov,
                       runtime_s=round(dur, 2),
                       **{f"n_{y}": by_yr[y] for y in YEARS_ALL})
            rows.append(row)
            print(f"[{model:<7} | {variant:<5}] alarms={len(al):4d}  "
                  f"FAR2019={far19:6.3f}/1k  y_cov={ycov}/5  "
                  f"n_post={n_a_post:4d}  ({dur:.1f}s)  "
                  f"per_year={by_yr}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False, float_format="%.4f")
    print(f"\n[ok] {OUT}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
