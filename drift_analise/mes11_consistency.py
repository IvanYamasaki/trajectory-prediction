"""
drift_analise/mes11_consistency.py
==================================
Mes 11 — Parte A3 do plano (PLANO_SECAO_DETECCAO.md): correcoes de consistencia.

(1) FAR da Frente B em unidades comparaveis.
    phase2_grid_search.py reportava FAR_2019_per1k da Frente B por 1000
    JANELAS (n_2019_total // N), enquanto Frentes A/C usam 1000 trajetorias.
    Recalcula a Frente B inteira reportando as DUAS unidades lado a lado.

(2) Gate AND unificado.
    A Fase 3 usa _and_gate (deque, limpa buffers apos CONFIRMED); o Mes 10
    usa and_confirmed (vizinho mais proximo). Recalcula o ensemble de mesma
    escala com AMBOS os gates para verificar se as conclusoes (FAR, cobertura,
    Delta_min) dependem da implementacao.

Saidas em Relas/results/drift/mes11_consistency/:
  frenteB_far_units.csv        — grid Frente B com FAR por janela E por trajetoria
  same_scale_gate_unified.csv  — sweep W do AND mesma-escala sob os dois gates
  MES11_CONSISTENCY_REPORT.md

Uso:
    python drift_analise/mes11_consistency.py
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, PageHinkley, run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import (
    smooth_robz, aggregate_stream, GRID_B,
)
from drift_analise.phase3_ensemble import _and_gate
from drift_analise.mes10_same_scale_ensemble import (
    build_accel_features, and_confirmed, delta_min, eval_alarms,
    ADWIN_ADE_CFG, ROBZ_W, W_SWEEP, DEBOUNCE, YEARS_ALL, YEARS_POST,
)

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes11_consistency"
ACCEL_GRID_CSV = (PROJECT_ROOT / "Relas" / "results" / "drift"
                  / "mes10_pendencias" / "same_scale_ensemble" / "accel_grid.csv")


# ── (1) Frente B com as duas unidades ─────────────────────────────────────────

def rerun_frente_b_units() -> pd.DataFrame:
    df = load_errors()
    rows: list[dict] = []
    for model in ("Seq2Seq", "Kalman"):
        stream = build_stream(df, model=model)
        year_vals = stream["year"].fillna(0).astype(int).values
        n_2019_traj = int((year_vals == 2019).sum())
        n_post_traj = int((year_vals >= 2021).sum())

        for N in GRID_B["N_window"]:
            ade_agg, year_agg, _ = aggregate_stream(stream, N)
            if len(ade_agg) == 0:
                continue
            b2019 = ade_agg[year_agg == 2019]
            b2019 = b2019[np.isfinite(b2019)]
            if len(b2019) < 5:
                b2019 = ade_agg[np.isfinite(ade_agg)]
            n_2019_win = max(1, n_2019_traj // N)

            for K_thr, K_delta in itertools.product(GRID_B["K_thr"],
                                                    GRID_B["K_delta"]):
                ph = PageHinkley.from_baseline(
                    b2019, K_thr=K_thr, K_delta=K_delta,
                    min_instances=max(3, int((year_agg == 2019).sum()) // 5),
                    cooldown=max(2, int((year_agg == 2019).sum()) // 20),
                )
                alarms_win = run_detector(ph, ade_agg)
                yr_at = year_agg[np.asarray(alarms_win, dtype=int)] \
                    if alarms_win else np.array([], dtype=int)
                n19 = int((yr_at == 2019).sum())
                n_post = int((yr_at >= 2021).sum())
                cov = sum(1 for y in YEARS_POST if (yr_at == y).sum() > 0)
                rows.append(dict(
                    model=model, N_window=N, K_thr=K_thr, K_delta=K_delta,
                    n_alarms=len(alarms_win), n_2019_alarms=n19,
                    n_post_alarms=n_post, year_coverage=cov,
                    FAR_2019_per1k_win=round(1000.0 * n19 / n_2019_win, 4),
                    FAR_2019_per1k_traj=round(1000.0 * n19 / n_2019_traj, 4),
                ))
    return pd.DataFrame(rows)


# ── (2) gate AND unificado no ensemble de mesma escala ────────────────────────

def rerun_same_scale_gates() -> tuple[pd.DataFrame, int]:
    feat = build_accel_features()
    df = load_errors()
    stream = build_stream(df, model="Seq2Seq")
    stream = stream.merge(feat, on=["proc_set_file", "traj_id"],
                          how="left", validate="one_to_one")
    year_vals = stream["year"].fillna(0).astype(int).values
    n_2019_total = int((year_vals == 2019).sum())
    n_post_total = int((year_vals >= 2021).sum())

    ade_sig = smooth_robz(stream["ade_traj"].astype(float).values, ROBZ_W)
    acc_sig = smooth_robz(stream["accel_p95"].astype(float).values, ROBZ_W)

    ade_alarms = run_detector(ADWINLite(**ADWIN_ADE_CFG), ade_sig)

    # melhor config de accel ja selecionada pelo mes10 (accel_grid.csv)
    grid = pd.read_csv(ACCEL_GRID_CSV)
    adm = grid[grid["FAR_2019_per1k"] <= 0.20]
    pool = adm if not adm.empty else grid
    best = pool.sort_values(
        ["year_coverage", "FAR_2019_per1k", "SNR_smooth"],
        ascending=[False, True, False]).iloc[0]
    acc_cfg = dict(delta=float(best["delta"]),
                   min_window=int(best["min_window"]),
                   cooldown=int(best["cooldown"]),
                   max_window=int(best["max_window"]))
    print(f"  [ACC cfg] {acc_cfg} (FAR={best['FAR_2019_per1k']}, "
          f"cov={best['year_coverage']})")
    acc_alarms = run_detector(ADWINLite(**acc_cfg), acc_sig)

    dmin = delta_min(np.asarray(ade_alarms), np.asarray(acc_alarms))
    rows = []
    for W in W_SWEEP:
        for gate_name, conf in [
            ("mes10_nearest", and_confirmed(np.asarray(ade_alarms),
                                            np.asarray(acc_alarms), W)),
            ("phase3_deque",  _and_gate(ade_alarms, acc_alarms, W, DEBOUNCE)),
        ]:
            m = eval_alarms(conf, year_vals, n_2019_total, n_post_total)
            rows.append(dict(gate=gate_name, W=W, n_confirmed=len(conf),
                             delta_min=dmin, **m))
    return pd.DataFrame(rows), dmin


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] Frente B — FAR nas duas unidades ...")
    fb = rerun_frente_b_units()
    fb_path = OUT_DIR / "frenteB_far_units.csv"
    fb.to_csv(fb_path, index=False)
    print(f"  [ok] {fb_path}")
    best_b = fb[(fb.model == "Seq2Seq") & (fb.N_window == 50) &
                (fb.K_thr == 50) & (fb.K_delta == 2.0)]
    print(best_b.to_string(index=False))

    print("[2/2] Ensemble mesma escala — dois gates AND ...")
    gates, dmin = rerun_same_scale_gates()
    g_path = OUT_DIR / "same_scale_gate_unified.csv"
    gates.to_csv(g_path, index=False)
    print(f"  [ok] {g_path}  (Delta_min={dmin})")
    print(gates[["gate", "W", "n_confirmed", "FAR_2019_per1k",
                 "year_coverage"]].to_string(index=False))

    # relatorio
    L = [
        "# Mes 11 — A3: correcoes de consistencia",
        "",
        "## (1) FAR da Frente B nas duas unidades",
        "",
        "A FAR publicada da Frente B estava por 1000 *janelas*"
        " (`n_2019_total // N`); Frentes A/C usam 1000 *trajetorias*.",
        "Config citada no paper (Seq2Seq, N=50, K=50, delta=2.0):",
        "",
        best_b.to_markdown(index=False) if not best_b.empty else "(vazio)",
        "",
        "Como o melhor PH_agg tem 0 alarmes em 2019, o headline (FAR=0) nao",
        "muda — mas qualquer comparacao cross-frente com FAR>0 deve usar a",
        "coluna `FAR_2019_per1k_traj`.",
        "",
        "## (2) Gate AND unificado (mesma escala)",
        "",
        f"Delta_min(ADE, ACC) = {dmin} trajetorias (identico sob ambos os gates,",
        "pois depende so das listas de alarmes).",
        "",
        gates.to_markdown(index=False),
        "",
        "Leitura: se `n_confirmed`/FAR/cobertura coincidem entre `mes10_nearest`",
        "e `phase3_deque`, a conclusao do Mes 10 nao depende da implementacao",
        "do gate; divergencias devem ser reportadas na secao com o gate da",
        "Fase 3 como canonico.",
    ]
    (OUT_DIR / "MES11_CONSISTENCY_REPORT.md").write_text(
        "\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'MES11_CONSISTENCY_REPORT.md'}")


if __name__ == "__main__":
    main()
