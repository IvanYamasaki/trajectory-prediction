"""
drift_analise/tail_hypothesis_addendum.py
=========================================
Adendo ao cap-sweep (tail_hypothesis_check.py), que REFUTOU a hipotese de
que o cap 1/alpha=10 do RuLSIF descarta a cauda da correcao (LSIF satura
em cap~3-5; cap=10 nao remove nada).

Teste decisivo para o mecanismo real da discrepancia LSIF vs RuLSIF:
aplicar aos pesos LSIF a transformacao analitica do ratio relativo

    w_alpha(x) = w(x) / (alpha * w(x) + 1 - alpha),   alpha = 0.1

(plug-in do r estimado pelo LSIF no alvo do RuLSIF) e recomputar a recovery.

  - Se recovery(w_alpha) ~ recovery(LSIF)  -> o ALVO relativo nao explica a
    queda para 9-23%; a discrepancia vem do AJUSTE do RuLSIF (objetivo
    alpha-misturado + regularizacao => pesos quase uniformes, ESS ~ 0.99).
  - Se recovery(w_alpha) ~ recovery(RuLSIF) -> o alvo relativo em si ja
    elimina a correcao (shrinkage intrinseco, nao artefato de ajuste).

Saida: covariate_shift_out/tail_hypothesis_addendum.csv
Uso:  python drift_analise/tail_hypothesis_addendum.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from drift_analise.chapter03_visuals import (   # noqa: E402
    load_errors, build_loader, extract_features,
    lsif_weights, rulisf_weights, ess_ratio, HORIZON, BASELINE_YEAR,
)

ALPHA = 0.1
TARGET_YEARS = [2021, 2022, 2023, 2025]
OUT_CSV = Path("covariate_shift_out/tail_hypothesis_addendum.csv")


def recovery_of(w: np.ndarray, ade_y_arr: np.ndarray,
                ade_y: float, ade_2019: float) -> float:
    ade_iw = float((w @ ade_y_arr) / w.sum())
    return (ade_y - ade_iw) / (ade_y - ade_2019) * 100.0


def main() -> None:
    df_all = load_errors()
    df_all = df_all[(df_all["model"] == "Seq2Seq") &
                    (df_all["horizon"] == HORIZON)].copy()
    loader = build_loader()

    src_raw = df_all[df_all["year"] == BASELINE_YEAR].reset_index(drop=True)
    x_src_raw, src_rows = extract_features(src_raw, loader)
    ade_2019 = float(src_rows["ade_traj"].astype(float).mean())

    rows = []
    for year in TARGET_YEARS:
        tgt_raw = df_all[df_all["year"] == year].reset_index(drop=True)
        x_tgt_raw, tgt_rows = extract_features(tgt_raw, loader)
        ade_y_arr = tgt_rows["ade_traj"].astype(float).to_numpy()
        ade_y = float(ade_y_arr.mean())
        if ade_y - ade_2019 <= 1e-6:
            continue

        scaler = StandardScaler().fit(np.vstack([x_src_raw, x_tgt_raw]))
        xs = scaler.transform(x_tgt_raw)
        xt = scaler.transform(x_src_raw)

        w_lsif = lsif_weights(xs, xt, n_centers=300, reg=1e-3, seed=42)
        # plug-in: alvo do RuLSIF avaliado no ratio estimado pelo LSIF
        w_alpha = w_lsif / (ALPHA * w_lsif + 1.0 - ALPHA)
        # RuLSIF ajustado (mesma config do pipeline)
        w_rul = rulisf_weights(xs, xt, n_centers=300, reg=1e-3,
                               alpha=ALPHA, seed=42)

        rec_l = recovery_of(w_lsif, ade_y_arr, ade_y, ade_2019)
        rec_a = recovery_of(w_alpha, ade_y_arr, ade_y, ade_2019)
        rec_r = recovery_of(w_rul, ade_y_arr, ade_y, ade_2019)

        rows.append({
            "year": year,
            "rec_lsif": round(rec_l, 2),
            "rec_alpha_plugin": round(rec_a, 2),
            "rec_rulisf_fit": round(rec_r, 2),
            "ess_lsif": round(ess_ratio(w_lsif), 4),
            "ess_alpha_plugin": round(ess_ratio(w_alpha), 4),
            "ess_rulisf_fit": round(ess_ratio(w_rul), 4),
            "w_lsif_max": round(float(w_lsif.max()), 3),
            "w_lsif_p05": round(float(np.percentile(w_lsif, 5)), 4),
            "frac_w_below_0p1": round(float(np.mean(w_lsif < 0.1)), 4),
        })
        print(f"[{year}] rec: LSIF={rec_l:.1f}%  alpha-plugin={rec_a:.1f}%  "
              f"RuLSIF-fit={rec_r:.1f}%   ESS: {ess_ratio(w_lsif):.3f} / "
              f"{ess_ratio(w_alpha):.3f} / {ess_ratio(w_rul):.3f}   "
              f"w_max={w_lsif.max():.2f}")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")


if __name__ == "__main__":
    main()
