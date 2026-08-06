"""
drift_analise/mes10_same_scale_ensemble.py
==========================================
Mes 10 — Pendencia 1.4: ensemble AND de MESMA escala temporal.

A Fase 3 mostrou que AND(ADWIN_S2S, PH_agg) e estruturalmente inviavel
porque os detectores operam em escalas diferentes (trajetoria vs janela
de 50). O teste natural dessa explicacao e alinhar as escalas: dois ADWIN
por trajetoria — um sobre o ADE do Seq2Seq, outro sobre uma feature
cinematica de aceleracao da PROPRIA trajetoria (accel_p95 da janela de
entrada), ambos com pre-processamento robz_w200.

Pipeline:
  1. Extrai accel_p95 por trajetoria de todos os proc_sets (sem inferencia
     do modelo — apenas vx,vy desnormalizados da janela de entrada 30 passos;
     cache em accel_features.parquet).
  2. Alinha com o stream Seq2Seq 30->15 via (proc_set_file, traj_id).
  3. Mini-grid ADWIN sobre o sinal de accel (robz_w200) para escolher um
     detector admissivel (FAR_2019 <= 0.20/1k, cobertura max).
  4. AND/OR(ADWIN_ADE, ADWIN_ACC) com sweep de W; reporta Delta_min, FAR,
     year_coverage e SNR_smooth — comparavel a tabela da Fase 4.

Saidas em Relas/results/drift/mes10_pendencias/same_scale_ensemble/:
  accel_features.parquet   (cache da feature por trajetoria)
  accel_grid.csv           (mini-grid do detector de accel)
  same_scale_ensemble.csv  (sweep W do AND/OR)
  ensemble_stream.png
  SAME_SCALE_REPORT.md

Uso:
    python drift_analise/mes10_same_scale_ensemble.py
"""
from __future__ import annotations

import itertools
import os
import pickle
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

from common.constants import (
    ROBZ_W, YEARS_ALL, YEARS_POST,
    RETRAIN_DEBOUNCE as DEBOUNCE, ADWIN_S2S_LITE,
)
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes10_pendencias" / "same_scale_ensemble"
FEAT_CACHE = OUT_DIR / "accel_features.parquet"

# Detector primario de ADE (Fase 3/4, variante lite — a do paper)
ADWIN_ADE_CFG = dict(ADWIN_S2S_LITE)

# Mini-grid para o detector de accel (mesma familia de valores da Fase 2)
GRID_ACC = dict(
    delta      = [1e-6, 1e-7],
    min_window = [200, 600],
    cooldown   = [200, 1000],
    max_window = [2000],
)

W_SWEEP  = [50, 100, 200, 500, 1000, 2000]

STATS_PATH = Path("model") / "normalization_stats_30_15.pkl"
LOOK_BACK, LOOK_FORTH = 30, 15

PROC_YEAR = {
    3: 2019, 4: 2019, 5: 2019, 6: 2019, 7: 2019, 8: 2019,
    9: 2021, 10: 2021, 11: 2021, 12: 2021, 13: 2021, 14: 2021,
    15: 2023, 16: 2023, 17: 2023, 18: 2023, 19: 2023, 20: 2023,
    21: 2025, 22: 2025, 23: 2025, 24: 2025, 25: 2025, 26: 2025,
    27: 2022, 28: 2022, 29: 2022, 30: 2022, 31: 2022, 32: 2022,
    33: 2024, 34: 2024, 35: 2024, 36: 2024, 37: 2024, 38: 2024,
}


# ── extracao de feature ───────────────────────────────────────────────────────

def build_accel_features() -> pd.DataFrame:
    """accel_p95 por janela de entrada, para todos os proc_sets 3..38.

    Nao requer o modelo: usa apenas vx,vy desnormalizados dos 30 passos de
    entrada; accel proxy = ||delta v|| por passo (unidade: mm/s por frame,
    proporcional a aceleracao — a escala e irrelevante apos robz).
    """
    if FEAT_CACHE.exists():
        print(f"[cache] {FEAT_CACHE}")
        return pd.read_parquet(FEAT_CACHE)

    from dataset.load_dataset import LoadDataSet
    with open(STATS_PATH, "rb") as f:
        s = pickle.load(f)
    avg = np.asarray(s["avg"], dtype=np.float64)
    std = np.asarray(s["std"], dtype=np.float64)

    rows = []
    for n in sorted(PROC_YEAR):
        fpath = f"dataset/proc_set_{n}"
        if not Path(fpath + ".pkl").exists():
            print(f"[warn] sem {fpath}.pkl")
            continue
        loader = LoadDataSet(LOOK_BACK, LOOK_FORTH, target="dv")
        loader.robots_avg = avg
        loader.robots_std = std
        loader.ball_avg   = np.asarray(s["ball_avg"], dtype=np.float64)
        loader.ball_std   = np.asarray(s["ball_std"], dtype=np.float64)
        try:
            x_all, _, _, _ = loader.load_data([fpath], for_test=True)
        except Exception as e:
            print(f"[warn] {fpath}: {e}")
            continue
        if len(x_all) == 0:
            continue
        # desnormaliza vx,vy e calcula ||delta v|| por passo
        v = x_all[:, :, 2:4] * std[2:4] + avg[2:4]          # (N, 30, 2)
        dv = np.linalg.norm(np.diff(v, axis=1), axis=2)     # (N, 29)
        accel_p95 = np.percentile(dv, 95, axis=1)           # (N,)
        for tid, a in enumerate(accel_p95):
            rows.append((f"proc_set_{n}.pkl", int(tid), float(a)))
        print(f"  proc_set_{n} (year={PROC_YEAR[n]}): {len(x_all):,} janelas")

    feat = pd.DataFrame(rows, columns=["proc_set_file", "traj_id", "accel_p95"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(FEAT_CACHE, index=False)
    print(f"[ok] cache -> {FEAT_CACHE} ({len(feat):,} linhas)")
    return feat


# ── metricas ──────────────────────────────────────────────────────────────────

def eval_alarms(alarms: list[int] | np.ndarray, year_vals: np.ndarray,
                n_2019_total: int, n_post_total: int) -> dict:
    a = np.asarray(sorted(alarms), dtype=int)
    by = {y: int((year_vals[a] == y).sum()) if len(a) else 0 for y in YEARS_ALL}
    n19    = by[2019]
    n_post = sum(by[y] for y in YEARS_POST)
    far    = 1000.0 * n19 / max(1, n_2019_total)
    cov    = sum(1 for y in YEARS_POST if by[y] > 0)
    snr_sm = ((n_post + 1.0) * n_2019_total) / ((n19 + 1.0) * n_post_total)
    return dict(FAR_2019_per1k=round(far, 4), year_coverage=cov,
                SNR_smooth=round(snr_sm, 3), n_post=n_post,
                **{f"n_{y}": by[y] for y in YEARS_ALL})


def and_confirmed(a1: np.ndarray, a2: np.ndarray, W: int,
                  debounce: int = DEBOUNCE) -> list[int]:
    """Alarmes confirmados: pares (a1, a2) com |diff| <= W, debounced."""
    confirmed = []
    last = -10 ** 9
    j = 0
    a2 = np.asarray(sorted(a2), dtype=int)
    for x in sorted(a1):
        # avanca j ate o alarme de a2 mais proximo de x
        while j + 1 < len(a2) and abs(a2[j + 1] - x) <= abs(a2[j] - x):
            j += 1
        if len(a2) and abs(a2[j] - x) <= W:
            t = max(x, int(a2[j]))
            if t - last >= debounce:
                confirmed.append(t)
                last = t
    return confirmed


def delta_min(a1: np.ndarray, a2: np.ndarray) -> int:
    if not len(a1) or not len(a2):
        return -1
    a1 = np.asarray(sorted(a1)); a2 = np.asarray(sorted(a2))
    d = np.min(np.abs(a1[:, None] - a2[None, :]))
    return int(d)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Extraindo/carregando accel_p95 por trajetoria ...")
    feat = build_accel_features()

    print("[2/4] Alinhando com o stream Seq2Seq 30->15 ...")
    df = load_errors()
    stream = build_stream(df, model="Seq2Seq")
    stream = stream.merge(feat, on=["proc_set_file", "traj_id"],
                          how="left", validate="one_to_one")
    n_missing = int(stream["accel_p95"].isna().sum())
    print(f"  stream={len(stream):,} | accel ausente={n_missing}")
    if n_missing > 0.01 * len(stream):
        raise RuntimeError(
            f"{n_missing} trajetorias sem accel — verifique o alinhamento "
            "(proc_set_file, traj_id).")

    year_vals = stream["year"].fillna(0).astype(int).values
    n_2019_total = int((year_vals == 2019).sum())
    n_post_total = int((year_vals >= 2021).sum())

    ade_sig = smooth_robz(stream["ade_traj"].astype(float).values, ROBZ_W)
    acc_sig = smooth_robz(stream["accel_p95"].astype(float).values, ROBZ_W)

    # detector de ADE (config final)
    print("[3/4] ADWIN_ADE (config final) + mini-grid ADWIN_ACC ...")
    ade_alarms = run_detector(ADWINLite(**ADWIN_ADE_CFG), ade_sig)
    m_ade = eval_alarms(ade_alarms, year_vals, n_2019_total, n_post_total)
    print(f"  ADWIN_ADE: {m_ade}")

    grid_rows = []
    acc_alarms_by_cfg = {}
    for delta, mw, cd, xw in itertools.product(
            GRID_ACC["delta"], GRID_ACC["min_window"],
            GRID_ACC["cooldown"], GRID_ACC["max_window"]):
        det = ADWINLite(delta=delta, min_window=mw, cooldown=cd, max_window=xw)
        alarms = run_detector(det, acc_sig)
        m = eval_alarms(alarms, year_vals, n_2019_total, n_post_total)
        key = f"d{delta:.0e}_mw{mw}_cd{cd}_xw{xw}"
        acc_alarms_by_cfg[key] = alarms
        grid_rows.append(dict(cfg=key, delta=delta, min_window=mw,
                              cooldown=cd, max_window=xw,
                              n_alarms=len(alarms), **m))
        print(f"  ACC {key}: FAR={m['FAR_2019_per1k']} cov={m['year_coverage']} "
              f"n_post={m['n_post']}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT_DIR / "accel_grid.csv", index=False)

    # escolhe detector de accel: admissivel, cov DESC, FAR ASC, SNR DESC
    adm = grid[grid["FAR_2019_per1k"] <= 0.20]
    pool = adm if not adm.empty else grid
    best = pool.sort_values(
        ["year_coverage", "FAR_2019_per1k", "SNR_smooth"],
        ascending=[False, True, False]).iloc[0]
    acc_alarms = acc_alarms_by_cfg[best["cfg"]]
    print(f"  [best ACC] {best['cfg']}: FAR={best['FAR_2019_per1k']} "
          f"cov={best['year_coverage']} (admissivel={not adm.empty})")

    # ── ensemble ─────────────────────────────────────────────────────────────
    print("[4/4] Ensemble AND/OR mesma escala ...")
    dmin = delta_min(np.asarray(ade_alarms), np.asarray(acc_alarms))
    print(f"  Delta_min(ADE, ACC) = {dmin} trajetorias")

    ens_rows = []
    for W in W_SWEEP:
        conf = and_confirmed(np.asarray(ade_alarms), np.asarray(acc_alarms), W)
        m = eval_alarms(conf, year_vals, n_2019_total, n_post_total)
        ens_rows.append(dict(ensemble="AND", W=W, n_confirmed=len(conf),
                             delta_min=dmin, **m))
    or_alarms = sorted(set(ade_alarms) | set(acc_alarms))
    m_or = eval_alarms(or_alarms, year_vals, n_2019_total, n_post_total)
    ens_rows.append(dict(ensemble="OR", W=0, n_confirmed=len(or_alarms),
                         delta_min=dmin, **m_or))
    # linhas standalone p/ referencia
    ens_rows.append(dict(ensemble="ADE_standalone", W=0,
                         n_confirmed=len(ade_alarms), delta_min=dmin, **m_ade))
    m_acc = eval_alarms(acc_alarms, year_vals, n_2019_total, n_post_total)
    ens_rows.append(dict(ensemble="ACC_standalone", W=0,
                         n_confirmed=len(acc_alarms), delta_min=dmin, **m_acc))

    ens = pd.DataFrame(ens_rows)
    ens.to_csv(OUT_DIR / "same_scale_ensemble.csv", index=False)
    print(ens[["ensemble", "W", "n_confirmed", "FAR_2019_per1k",
               "year_coverage", "SNR_smooth"]].to_string(index=False))

    # ── figura ───────────────────────────────────────────────────────────────
    x = stream["global_idx"].values
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 1.4],
                                              hspace=0.08))
    for yr in YEARS_ALL:
        mask = year_vals == yr
        if mask.any():
            x0, x1 = int(x[mask].min()), int(x[mask].max())
            for ax in axes:
                ax.axvspan(x0, x1 + 1, alpha=0.06, color="C0" if yr == 2019 else "C1")
    axes[0].plot(x, ade_sig, lw=0.7, color="#1c1c8c")
    axes[0].set_ylabel("ADE robz")
    axes[0].set_title("Ensemble de mesma escala — ADWIN(ADE) x ADWIN(accel_p95), robz_w200")
    axes[1].plot(x, acc_sig, lw=0.7, color="#8c1c1c")
    axes[1].set_ylabel("accel_p95 robz")
    if len(ade_alarms):
        axes[2].vlines(np.asarray(ade_alarms), 0.66, 0.98, color="#2A6FDB", lw=1)
    if len(acc_alarms):
        axes[2].vlines(np.asarray(acc_alarms), 0.34, 0.64, color="#D1495B", lw=1)
    conf200 = and_confirmed(np.asarray(ade_alarms), np.asarray(acc_alarms), 200)
    if conf200:
        axes[2].vlines(np.asarray(conf200), 0.02, 0.32, color="#111", lw=1.5)
    axes[2].set_yticks([0.17, 0.49, 0.82])
    axes[2].set_yticklabels([f"AND W=200 ({len(conf200)})",
                             f"ACC ({len(acc_alarms)})",
                             f"ADE ({len(ade_alarms)})"], fontsize=8)
    axes[2].set_xlabel("Trajetoria (ordem global)")
    fig.savefig(OUT_DIR / "ensemble_stream.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── relatorio ────────────────────────────────────────────────────────────
    L = [
        "# Mes 10 — Ensemble AND de mesma escala temporal",
        "",
        f"Sinais: ADE Seq2Seq e accel_p95 por trajetoria (janela de entrada),",
        f"ambos robz_w{ROBZ_W}, por trajetoria (n={len(stream):,}).",
        "",
        f"- ADWIN_ADE (config final): FAR={m_ade['FAR_2019_per1k']}/1k, "
        f"cov={m_ade['year_coverage']}/5, n_post={m_ade['n_post']}",
        f"- ADWIN_ACC (melhor do grid, `{best['cfg']}`): "
        f"FAR={best['FAR_2019_per1k']}/1k, cov={best['year_coverage']}/5, "
        f"n_post={best['n_post']}",
        f"- **Delta_min = {dmin} trajetorias** "
        "(Fase 3, escalas distintas: Delta_min=214)",
        "",
        "## Sweep W (AND) + OR",
        "",
        "| Ensemble | W | n_conf | FAR/1k | year_cov | SNR_smooth |",
        "|----------|--:|-------:|-------:|---------:|-----------:|",
    ]
    for _, r in ens.iterrows():
        L.append(f"| {r['ensemble']} | {r['W']} | {r['n_confirmed']} | "
                 f"{r['FAR_2019_per1k']:.4f} | {r['year_coverage']}/5 | "
                 f"{r['SNR_smooth']:.3f} |")
    (OUT_DIR / "SAME_SCALE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'SAME_SCALE_REPORT.md'}")


if __name__ == "__main__":
    main()
