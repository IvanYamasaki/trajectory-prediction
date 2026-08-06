"""
sample_size_sensitivity.py  —  Frente 1 do Mês 3
=================================================

Compara estatísticas de ADE (média + CI95 bootstrap) entre o run n=3
(backup parquet) e o run n=6 (parquet principal após compute_trajectory_errors
--n_per_year 6 --seed 42).

Saída: Relas/results/drift/04_robustez_e_validacao/sample_size_sensitivity.csv
Colunas: n_per_year, year, model, horizon, ade_mean, ci_lo, ci_hi, n_trajs

Uso:
    python model_analise/sample_size_sensitivity.py
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bootstrap import init_project

PROJECT_ROOT = init_project()

N_BOOT = 2000
SEED = 42
RNG = np.random.default_rng(SEED)

from common.paths import CH04

COV_DIR = Path("covariate_shift_out")
OUT_DIR = CH04
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_N3 = COV_DIR / "trajectory_errors_sample_n3_backup.parquet"
PARQUET_N6 = COV_DIR / "trajectory_errors_sample.parquet"
CSV_N6     = COV_DIR / "trajectory_errors_sample.csv"


def load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[warn] cannot read {path}: {e}")
    return None


def bootstrap_ci(values: np.ndarray, n_boot=N_BOOT, seed=SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(values.mean()), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame, n_per_year: int) -> pd.DataFrame:
    rows = []
    for (year, model, horizon), grp in df.groupby(["year", "model", "horizon"]):
        ades = grp["ade_traj"].astype(float).to_numpy()
        ci_lo, mean, ci_hi = bootstrap_ci(ades)
        rows.append({
            "n_per_year": n_per_year,
            "year": int(year),
            "model": model,
            "horizon": horizon,
            "ade_mean": round(mean, 4),
            "ci_lo":    round(ci_lo, 4),
            "ci_hi":    round(ci_hi, 4),
            "n_trajs":  len(ades),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parts = []

    df3 = load_parquet(PARQUET_N3)
    if df3 is not None:
        print(f"[info] n=3 backup parquet: {len(df3):,} rows")
        parts.append(summarize(df3, n_per_year=3))
    else:
        print("[warn] n=3 backup não encontrado; pulando.")

    df6 = load_parquet(PARQUET_N6)
    if df6 is None and CSV_N6.exists():
        df6 = pd.read_csv(CSV_N6)
        print(f"[info] n=6 CSV fallback: {len(df6):,} rows")
    elif df6 is not None:
        print(f"[info] n=6 parquet: {len(df6):,} rows")
    else:
        print("[err] n=6 parquet/CSV não encontrado. Execute primeiro:")
        print("  python model_analise/compute_trajectory_errors.py --n_per_year 6 --seed 42")
        sys.exit(1)

    # infer n_per_year from the data
    n6_actual = df6.groupby(["year", "model", "horizon"])["proc_set_file"].nunique().min()
    parts.append(summarize(df6, n_per_year=int(n6_actual)))

    out = pd.concat(parts, ignore_index=True)

    # flag instability: years where ADE mean shifts > 1 mm between n3 and n6
    if len(parts) == 2:
        pivot = out.pivot_table(index=["year","model","horizon"],
                                columns="n_per_year", values="ade_mean")
        if 3 in pivot.columns and int(n6_actual) in pivot.columns:
            diff = (pivot[int(n6_actual)] - pivot[3]).abs()
            flag = diff[diff > 1.0]
            if not flag.empty:
                print("\n[AVISO] Anos com ADE variando >1 mm entre n=3 e n=6:")
                print(flag.to_string())
            else:
                print("\n[OK] ADE estável entre n=3 e n=6 (delta < 1 mm em todos os casos).")

    out_path = OUT_DIR / "sample_size_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[ok] {len(out)} linhas -> {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
