"""
drift_analise/mes10_latency.py
==============================
Mes 10 — Pendencia 1.1: latencia de detecao do detetor primario no stream real.

Para cada ano pos-2019, mede o atraso entre a fronteira do ano (primeira
trajetoria daquele ano no stream global) e o primeiro alarme do detector
dentro do ano. Completa a triade padrao de avaliacao de detectores
(FAR + cobertura + latencia; Gama et al. 2014, Lu et al. 2019).

Convencao: como o drift de 2022-2025 e gradual (sem changepoint formal ao
nivel per-game, Pelt mBIC K*=0), a fronteira do ano e usada como proxy do
onset — a latencia reportada e portanto um LIMITE SUPERIOR da latencia real.

Detectores avaliados (configs finais da Fase 3/4, variante lite):
  ADWIN_S2S    : robz_w200 + ADWIN(delta=1e-7, mw=600, cd=200,  xw=2000)
  ADWIN_Kalman : robz_w200 + ADWIN(delta=1e-7, mw=200, cd=1000, xw=5000)

Saidas em Relas/results/drift/mes10_pendencias/latency/:
  latency_results.csv
  latency_bars.png
  LATENCY_REPORT.md

Uso:
    python drift_analise/mes10_latency.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.constants import ROBZ_W, YEARS_POST, ADWIN_S2S_LITE, ADWIN_KALMAN_LITE
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes10_pendencias" / "latency"

# Configs finais (phase3_ensemble.py, ADWIN_VARIANT=lite — as do paper)
CONFIGS = {
    "Seq2Seq": dict(ADWIN_S2S_LITE),
    "Kalman":  dict(ADWIN_KALMAN_LITE),
}

# ~8000 trajetorias por jogo no stream amostrado (48k/ano / 6 jogos)
TRAJS_PER_GAME = 8000.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_errors()

    rows = []
    alarm_map: dict[str, list[int]] = {}
    stream_map: dict[str, pd.DataFrame] = {}

    for model, cfg in CONFIGS.items():
        stream = build_stream(df, model=model)
        if stream.empty:
            print(f"[warn] sem dados para {model}")
            continue
        raw = stream["ade_traj"].astype(float).values
        yrs = stream["year"].fillna(0).astype(int).values
        sig = smooth_robz(raw, ROBZ_W)

        det = ADWINLite(**cfg)
        alarms = run_detector(det, sig)
        alarm_map[model] = alarms
        stream_map[model] = stream
        a = np.asarray(alarms, dtype=int)

        print(f"[{model}] n_stream={len(stream):,}  n_alarms={len(alarms)}  "
              f"(2019: {(yrs[a] == 2019).sum() if len(a) else 0})")

        for y in YEARS_POST:
            mask = yrs == y
            if not mask.any():
                continue
            boundary = int(np.argmax(mask))                 # primeiro idx do ano
            n_year   = int(mask.sum())
            in_year  = a[(a >= boundary) & (a < boundary + n_year)]
            if len(in_year):
                first  = int(in_year[0])
                delay  = first - boundary
                rows.append(dict(
                    model=model, year=y,
                    boundary_idx=boundary, first_alarm_idx=first,
                    delay_trajs=delay,
                    delay_games=round(delay / TRAJS_PER_GAME, 2),
                    delay_pct_year=round(100.0 * delay / n_year, 1),
                    n_alarms_year=len(in_year), n_trajs_year=n_year,
                ))
            else:
                rows.append(dict(
                    model=model, year=y,
                    boundary_idx=boundary, first_alarm_idx=-1,
                    delay_trajs=-1, delay_games=np.nan,
                    delay_pct_year=np.nan,
                    n_alarms_year=0, n_trajs_year=n_year,
                ))

    res = pd.DataFrame(rows)
    csv_path = OUT_DIR / "latency_results.csv"
    res.to_csv(csv_path, index=False)
    print(f"[ok] {csv_path}")
    print(res.to_string(index=False))

    # ── figura: barras de latencia por ano ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    xs = np.arange(len(YEARS_POST))
    colors = {"Seq2Seq": "#2A6FDB", "Kalman": "#D1495B"}
    for i, model in enumerate(CONFIGS):
        sub = res[res["model"] == model].set_index("year")
        vals = [sub.loc[y, "delay_trajs"] if y in sub.index else np.nan
                for y in YEARS_POST]
        vals = [np.nan if (v is None or v < 0) else v for v in vals]
        ax.bar(xs + (i - 0.5) * width, vals, width,
               label=f"ADWIN_{model}", color=colors[model], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in YEARS_POST])
    ax.set_ylabel("Latencia (trajetorias apos fronteira do ano)")
    ax.set_title("Latencia de detecao — primeiro alarme por ano (robz_w200 + ADWIN)")
    ax2 = ax.secondary_yaxis(
        "right", functions=(lambda v: v / TRAJS_PER_GAME,
                            lambda g: g * TRAJS_PER_GAME))
    ax2.set_ylabel("~jogos")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_bars.png", dpi=150)
    plt.close(fig)
    print(f"[ok] {OUT_DIR / 'latency_bars.png'}")

    # ── relatorio MD ──────────────────────────────────────────────────────────
    L = [
        "# Mes 10 — Latencia de detecao no stream real",
        "",
        "Convencao: fronteira do ano como proxy do onset (drift gradual sem",
        "changepoint formal) — latencia reportada e limite superior.",
        "",
        f"Escala: ~{int(TRAJS_PER_GAME)} trajetorias/jogo; 48k trajetorias/ano.",
        "",
    ]
    for model in CONFIGS:
        sub = res[res["model"] == model]
        L += [f"## ADWIN_{model} ({CONFIGS[model]})", ""]
        L.append("| Ano | Latencia (trajs) | ~jogos | % do bloco do ano | alarmes no ano |")
        L.append("|-----|-----------------:|-------:|------------------:|---------------:|")
        for _, r in sub.iterrows():
            if r["delay_trajs"] >= 0:
                L.append(f"| {r['year']} | {r['delay_trajs']:,} | "
                         f"{r['delay_games']:.2f} | {r['delay_pct_year']:.1f}% | "
                         f"{r['n_alarms_year']} |")
            else:
                L.append(f"| {r['year']} | sem alarme | — | — | 0 |")
        L.append("")

    (OUT_DIR / "LATENCY_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'LATENCY_REPORT.md'}")


if __name__ == "__main__":
    main()
