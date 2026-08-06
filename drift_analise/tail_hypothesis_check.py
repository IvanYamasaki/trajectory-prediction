"""
drift_analise/tail_hypothesis_check.py
======================================
Verificacao direta da "hipotese da cauda": a discrepancia LSIF (59-72%) vs
RuLSIF (9-23%) e atribuida a concentracao da correcao covariate na regiao
de razao de densidades alta (w > 1/alpha = 10), que o RuLSIF capa.

Teste: cap-sweep sobre os pesos LSIF. Para cada ano com excesso, ajusta o
LSIF uma vez e recalcula a recovery com pesos clipados em c crescente:

    recovery(c) = (ADE_y - ADE_IW(clip(w, 0, c))) / excess * 100

Leituras esperadas se a hipotese for verdadeira:
  1. recovery(c) cresce monotonicamente com c (a correcao vem da cauda);
     recovery(10) ~ nivel RuLSIF e recovery(inf) = nivel LSIF.
  2. A massa de peso e concentrada: top-1% das trajetorias carrega fracao
     desproporcional de sum(w).
  3. w correlaciona negativamente com accel_p99 (pesos altos = trajetorias
     estilo-2019, pouco agressivas; o novo regime recebe w ~ 0).

Saida: covariate_shift_out/tail_hypothesis_check.csv (+ resumo no stdout)
Uso:  python drift_analise/tail_hypothesis_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from drift_analise.chapter03_visuals import (   # noqa: E402
    load_errors, build_loader, extract_features,
    lsif_weights, ess_ratio, HORIZON, BASELINE_YEAR,
)

CAPS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, np.inf]
TARGET_YEARS = [2021, 2022, 2023, 2025]
ACCEL_P99_IDX = 5   # FEAT_NAMES = [speed_mean, speed_p90, speed_p99,
                    #               accel_mean, accel_p90, accel_p99,
                    #               turn_mean,  turn_p90]
OUT_CSV = Path("covariate_shift_out/tail_hypothesis_check.csv")


def main() -> None:
    df_all = load_errors()
    df_all = df_all[(df_all["model"] == "Seq2Seq") &
                    (df_all["horizon"] == HORIZON)].copy()
    loader = build_loader()

    src_raw = df_all[df_all["year"] == BASELINE_YEAR].reset_index(drop=True)
    x_src_raw, src_rows = extract_features(src_raw, loader)
    ade_2019 = float(src_rows["ade_traj"].astype(float).mean())
    print(f"[info] ADE 2019 = {ade_2019:.4f} mm  (n={len(src_rows):,})")

    rows = []
    for year in TARGET_YEARS:
        tgt_raw = df_all[df_all["year"] == year].reset_index(drop=True)
        x_tgt_raw, tgt_rows = extract_features(tgt_raw, loader)
        ade_y_arr = tgt_rows["ade_traj"].astype(float).to_numpy()
        ade_y = float(ade_y_arr.mean())
        excess = ade_y - ade_2019
        if excess <= 1e-6:
            print(f"[warn] {year}: sem excesso — pulando.")
            continue

        scaler = StandardScaler().fit(np.vstack([x_src_raw, x_tgt_raw]))
        xs = scaler.transform(x_tgt_raw)
        xt = scaler.transform(x_src_raw)
        w = lsif_weights(xs, xt, n_centers=300, reg=1e-3, seed=42)

        # (2) concentracao de massa de peso
        order = np.argsort(w)[::-1]
        wsum = w.sum()
        top1 = float(w[order[: max(1, len(w) // 100)]].sum() / wsum * 100)
        top5 = float(w[order[: max(1, len(w) // 20)]].sum() / wsum * 100)
        # (3) onde a cauda mora no espaco de features
        rho, _ = spearmanr(w, x_tgt_raw[:, ACCEL_P99_IDX])

        print(f"\n[info] === {year}:  excess={excess:.4f} mm  "
              f"ESS={ess_ratio(w):.3f}  top1%={top1:.1f}%  top5%={top5:.1f}%  "
              f"rho(w, accel_p99)={rho:+.3f} ===")

        for cap in CAPS:
            wc = np.clip(w, 0.0, cap)
            ade_iw_c = float((wc @ ade_y_arr) / wc.sum())
            rec_c = (ade_y - ade_iw_c) / excess * 100.0
            rows.append({
                "year": year, "cap": cap, "recovery_pct": round(rec_c, 2),
                "ess": round(ess_ratio(wc), 4),
                "w_max": round(float(w.max()), 2),
                "top1_share_pct": round(top1, 2),
                "top5_share_pct": round(top5, 2),
                "rho_w_accel_p99": round(float(rho), 4),
            })
            cap_s = "inf" if np.isinf(cap) else f"{cap:g}"
            print(f"    cap={cap_s:>4}: recovery={rec_c:6.2f}%  "
                  f"ESS={ess_ratio(wc):.3f}")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")


if __name__ == "__main__":
    main()
