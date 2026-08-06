"""
drift_analise/fig_deteccao_narrativa.py
======================================
Figura narrativa da secao de deteccao online (substitui fig_deteccao_pt).

A figura antiga (mes11_accel_battery.make_summary_png) tinha 4 paineis de
auditoria: forest plot de FAR, scatter calib-vs-teste, heatmap de contagens e
latencia. Bom para conferir o experimento, ruim para o leitor: nada mostra o
que o detector ve, e 7 detectores competem por atencao quando so 3 entram na
arquitetura final.

Esta versao segue a linha de raciocinio da secao em tres passos:

  (a) O STREAM E OS ALARMES — sinal z-robusto do ADE ao longo do split de
      teste (fold A), faixas por ano, e rug dos tres detectores da arquitetura
      (erro / accel_p95 / AND) nos dois folds. Le-se FAR (2019 vazio),
      cobertura (todo ano pos-2019 marcado) e latencia (marca logo apos a
      fronteira) numa unica imagem.
  (b) O CUSTO DE CADA CAMADA — plano FAR x cobertura no teste, com o teto de
      admissibilidade. Camadas escolhidas destacadas; detectores dominados em
      cinza (mantem o ranking accel > speed visivel sem poluir).
  (c) QUEM AVISA PRIMEIRO — dumbbell de latencia por ano-fold, accel_p95 vs
      detector de erro.

Reusa os artefatos ja calculados:
  Relas/results/drift/mes11_accel_battery/battery_results.csv   (configs + metricas)
  Relas/results/drift/mes11_accel_battery/battery_latency.csv   (latencias)
  Relas/results/drift/mes11_accel_battery/features_ext.parquet  (features)
  covariate_shift_out/trajectory_errors_sample.parquet          (stream de ADE)

Os alarmes sao reexecutados a partir das configs vencedoras registradas no
battery_results.csv (mesmas classes/detectores do experimento) e cacheados em
alarms_stream.csv, para que a figura seja reproduzivel sem refazer a bateria.

Uso:
    python drift_analise/fig_deteccao_narrativa.py            # gera cache + PNG
    python drift_analise/fig_deteccao_narrativa.py --paper    # + PDF em main_pt/figs
    python drift_analise/fig_deteccao_narrativa.py --plot     # so replota do cache
"""
from __future__ import annotations

import argparse
import os
import re
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from drift_analise.chapter02_deteccao_pipeline import (
    run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz
from drift_analise.phase3_ensemble import _and_gate
from drift_analise.mes11_holdout_validation import (
    ADWINLiteFast, split_by_games, FAR_ADMISSIBILITY, YEARS_POST, ROBZ_W,
)
from drift_analise.mes11_accel_battery import FEATURES, FEAT_CACHE, W_AND, DEBOUNCE

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes11_accel_battery"
RES_CSV = OUT_DIR / "battery_results.csv"
LAT_CSV = OUT_DIR / "battery_latency.csv"
ALARM_CACHE = OUT_DIR / "alarms_stream.csv"
SIGNAL_CACHE = OUT_DIR / "signal_foldA.npz"

DET_ADE = "ADE (erro do modelo)"
DET_ACC = "accel_p95"
DET_AND = "AND(ADE, accel_p95)"
LAYERS = [DET_ADE, DET_ACC, DET_AND]

# rotulos curtos (PT) usados na figura
LABEL = {
    DET_ADE: "erro do modelo (ADE)",
    DET_ACC: "aceleração (accel_p95)",
    DET_AND: "conjunção erro∧accel",
}


def parse_cfg(cfg_str: str) -> dict:
    """'delta=1e-04, mw=1000, cd=500, xw=5000' -> dict de kwargs."""
    g = dict(re.findall(r"(\w+)=([0-9.e+-]+)", cfg_str))
    return dict(delta=float(g["delta"]), min_window=int(g["mw"]),
                cooldown=int(g["cd"]), max_window=int(g["xw"]))


# ── recomputa alarmes dos 3 detectores da arquitetura, por fold ──────────────

def compute_alarms() -> pd.DataFrame:
    res = pd.read_csv(RES_CSV)
    feat = pd.read_parquet(FEAT_CACHE, columns=["proc_set_file", "traj_id", DET_ACC])
    stream = build_stream(load_errors(), model="Seq2Seq")
    stream = stream.merge(feat, on=["proc_set_file", "traj_id"],
                          how="left", validate="one_to_one")

    rows = []
    for fold in ("A", "B"):
        _, test, _ = split_by_games(stream, fold)
        years = test["year"].fillna(0).astype(int).values

        cfg_ade = parse_cfg(res.query("fold == @fold and detector == @DET_ADE")
                            ["config"].iloc[0])
        cfg_acc = parse_cfg(res.query("fold == @fold and detector == @DET_ACC")
                            ["config"].iloc[0])

        raw_ade = test["ade_traj"].astype(float).values
        sig_ade = smooth_robz(raw_ade, ROBZ_W)
        sig_acc = smooth_robz(test[DET_ACC].astype(float).values, ROBZ_W)

        a_ade = run_detector(ADWINLiteFast(**cfg_ade), sig_ade)
        a_acc = run_detector(ADWINLiteFast(**cfg_acc), sig_acc)
        a_and = _and_gate(a_ade, a_acc, W_AND, DEBOUNCE)

        for name, alarms in ((DET_ADE, a_ade), (DET_ACC, a_acc), (DET_AND, a_and)):
            for idx in alarms:
                rows.append(dict(fold=fold, detector=name, idx=int(idx),
                                 year=int(years[idx])))
        print(f"[fold {fold}] n_teste={len(test):,} | "
              f"ADE={len(a_ade)} accel={len(a_acc)} AND={len(a_and)}")

        if fold == "A":
            np.savez_compressed(SIGNAL_CACHE,
                                raw=raw_ade.astype(np.float32),
                                sig=sig_ade.astype(np.float32), years=years)
            print(f"[ok] {SIGNAL_CACHE}")

    out = pd.DataFrame(rows)
    out.to_csv(ALARM_CACHE, index=False)
    print(f"[ok] {ALARM_CACHE} ({len(out)} alarmes)")
    return out


# ── paleta (mesma da bateria, dataviz reference modo claro) ─────────────────

from common.plotting import INK, INK_2, MUTED, GRID, BASELINE, SURFACE, FAINT

C_ADE = "#2a78d6"
C_ACC = "#1baf7a"
C_AND = "#e34948"
COLOR = {DET_ADE: C_ADE, DET_ACC: C_ACC, DET_AND: C_AND}


def make_figure(paper: bool = False) -> None:
    res = pd.read_csv(RES_CSV)
    lat = pd.read_csv(LAT_CSV)
    al  = pd.read_csv(ALARM_CACHE)
    z   = np.load(SIGNAL_CACHE)
    raw, sig, years = z["raw"], z["sig"], z["years"]
    al_a = al[al["fold"] == "A"]

    # No modo paper a figura entra em \linewidth do LNCS (~4,8 in): a escala e
    # de ~0,63, entao a figura precisa nascer pequena e com fonte grande para
    # que o texto impresso fique em ~7,5 pt. Manter a razao H/W <= 0,72 para
    # nao empurrar paginas.
    base_fs = 11.5 if paper else 10
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": base_fs,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.grid": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": base_fs - 1, "ytick.labelsize": base_fs - 1,
    })
    d = 0.0 if paper else 0

    fig = plt.figure(figsize=(7.6, 5.45) if paper else (12.0, 9.6))
    L_, R_ = (0.115, 0.985) if paper else (0.095, 0.975)
    gs_top = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.85, 0.75],
                              hspace=0.18, left=L_, right=R_,
                              top=0.925, bottom=0.435)
    gs_bot = fig.add_gridspec(1, 2, wspace=0.30, left=L_, right=R_,
                              top=0.315, bottom=0.085)
    ax_raw = fig.add_subplot(gs_top[0])
    ax_sig = fig.add_subplot(gs_top[1], sharex=ax_raw)
    ax_rug = fig.add_subplot(gs_top[2], sharex=ax_raw)
    ax_far = fig.add_subplot(gs_bot[0])
    ax_lat = fig.add_subplot(gs_bot[1])

    # ── (a) stream: bruto -> normalizado -> alarmes ─────────────────────────
    yrs = [2019] + YEARS_POST
    bounds = {}
    for y in yrs:
        m = np.flatnonzero(years == y)
        if len(m):
            bounds[y] = (int(m[0]), int(m[-1]) + 1)

    x = np.arange(len(sig))
    for i, y in enumerate(yrs):
        if y not in bounds:
            continue
        x0, x1 = bounds[y]
        if i % 2 == 1:
            for a in (ax_raw, ax_sig, ax_rug):
                a.axvspan(x0, x1, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax_raw.text((x0 + x1) / 2, 1.04, str(y),
                    transform=ax_raw.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=base_fs - 1 + d,
                    color=INK if y != 2019 else INK_2,
                    fontweight="bold" if y != 2019 else "normal")
        if y != 2019:
            for a in (ax_raw, ax_sig, ax_rug):
                a.axvline(x0, color=BASELINE, lw=0.9, ls=(0, (3, 3)), zorder=1)

    # (a.1) sinal bruto em mm — o erro que cresce entre temporadas
    roll_raw = pd.Series(raw.astype(float)).rolling(4000, min_periods=1000).mean()
    base19 = float(np.nanmean(raw[years == 2019]))
    ax_raw.axhline(base19, color=INK_2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax_raw.plot(x, roll_raw.values, color=C_ADE, lw=1.5, zorder=3)
    ax_raw.fill_between(x, base19, roll_raw.values,
                        where=(roll_raw.values > base19),
                        color=C_ADE, alpha=0.16, lw=0, zorder=2)
    ax_raw.text(len(sig) * 0.30, base19 - 0.12, "média de 2019",
                fontsize=base_fs - 3 + d, color=INK_2, va="top", ha="center")
    ax_raw.set_ylabel("ADE bruto (mm)", fontsize=base_fs - 2 + d)
    ax_raw.set_title("(a) O stream, o sinal que o ADWIN lê e os alarmes "
                     "(fold A, teste)",
                     fontsize=base_fs - 0.5 + d, loc="left", color=INK, pad=16)
    ax_raw.grid(axis="y", color=GRID, lw=0.7)
    ax_raw.set_axisbelow(True)
    ax_raw.tick_params(labelbottom=False)

    # (a.2) sinal z-robusto — estacionario por construcao; o ADWIN age aqui
    roll = pd.Series(sig.astype(float)).rolling(2000, min_periods=500).mean()
    ax_sig.axhline(0, color=BASELINE, lw=0.9, zorder=1)
    ax_sig.plot(x, roll.values, color=INK_2, lw=1.2, zorder=3)
    ax_sig.set_ylabel("z-robusto", fontsize=base_fs - 2 + d)
    ax_sig.set_yticks([0.0, 0.4, 0.8])
    ax_sig.grid(axis="y", color=GRID, lw=0.7)
    ax_sig.set_axisbelow(True)
    ax_sig.tick_params(labelbottom=False)

    for a in (ax_raw, ax_sig):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    # (a.3) rug de alarmes das 3 camadas
    rug_y = {DET_ADE: 2, DET_ACC: 1, DET_AND: 0}
    for _, r in al_a.iterrows():
        base = rug_y[r["detector"]]
        ax_rug.plot([r["idx"], r["idx"]], [base - 0.30, base + 0.30],
                    color=COLOR[r["detector"]], lw=1.1, alpha=0.9, zorder=3)
    ax_rug.set_ylim(-0.75, 2.55)
    ax_rug.set_yticks([2, 1, 0])
    ax_rug.set_yticklabels(["erro (ADE)", "accel_p95", "erro∧accel"],
                           fontsize=base_fs - 2.5 + d)
    for tick, dn in zip(ax_rug.get_yticklabels(), [DET_ADE, DET_ACC, DET_AND]):
        tick.set_color(COLOR[dn])
    ax_rug.set_xlabel(f"{len(sig)//1000} mil trajetórias do stream de teste, "
                      f"em ordem temporal",
                      fontsize=base_fs - 2 + d, labelpad=2)
    ax_rug.set_xlim(0, len(sig))
    ax_rug.set_xticks([])
    for s in ("top", "right", "left"):
        ax_rug.spines[s].set_visible(False)
    ax_rug.tick_params(axis="y", length=0)
    ax_rug.grid(axis="x", color=GRID, lw=0.6)
    ax_rug.set_axisbelow(True)

    # (a nota sobre 2019 = zona sem drift fica na legenda da figura)

    # ── (b) plano FAR x cobertura ───────────────────────────────────────────
    others = [c for c in res["detector"].unique()
              if c not in LAYERS and c.startswith(("accel", "speed"))]
    ax_far.axvspan(-0.02, FAR_ADMISSIBILITY, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax_far.axvline(FAR_ADMISSIBILITY, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    rng = np.random.default_rng(7)
    for _, r in res.iterrows():
        dn = r["detector"]
        if dn not in LAYERS and dn not in others:
            continue
        jit = rng.uniform(-0.07, 0.07)
        is_layer = dn in LAYERS
        ax_far.scatter([r["FAR_2019_per1k"]], [r["year_coverage"] + jit],
                       s=70 if is_layer else 34,
                       color=COLOR[dn] if is_layer else FAINT,
                       marker="o" if r["fold"] == "A" else "s",
                       edgecolors=SURFACE, linewidths=1.1,
                       zorder=4 if is_layer else 3)
    ax_far.annotate("demais features\n(accel_mean, speed_*)", (0.42, 4.40),
                    fontsize=base_fs - 3 + d, color=MUTED, ha="left", va="center")
    for dn, txt, xy, ha in ((DET_ADE, "erro do modelo", (0.0, 5.72), "left"),
                            (DET_ACC, "aceleração", (0.34, 5.72), "left"),
                            (DET_AND, "conjunção\nerro∧accel", (0.05, 2.55), "left")):
        ax_far.annotate(txt, xy, fontsize=base_fs - 2.5 + d, color=COLOR[dn],
                        ha=ha, va="center", linespacing=1.15)
    ax_far.text(FAR_ADMISSIBILITY / 2, 1.60, "admissível",
                fontsize=base_fs - 3 + d, color=INK_2, ha="center", va="center")
    ax_far.set_xlim(-0.03, 0.88)
    ax_far.set_ylim(1.4, 6.05)
    ax_far.set_yticks([2, 3, 4, 5])
    ax_far.set_xlabel("falsos alarmes em 2019 (/1k trajetórias)",
                      fontsize=base_fs - 2 + d, labelpad=2)
    ax_far.set_ylabel("cobertura (anos de 5)", fontsize=base_fs - 2 + d)
    ax_far.set_title("(b) O custo de cada camada", fontsize=base_fs - 0.5 + d,
                     loc="left", color=INK, pad=6)
    ax_far.grid(color=GRID, lw=0.7)
    ax_far.set_axisbelow(True)
    ax_far.scatter([], [], marker="o", color=INK_2, s=30, label="fold A")
    ax_far.scatter([], [], marker="s", color=INK_2, s=30, label="fold B")
    ax_far.legend(loc="lower right", fontsize=base_fs - 3 + d, frameon=False)
    for s in ("top", "right"):
        ax_far.spines[s].set_visible(False)

    # ── (c) dumbbell de latencia accel vs erro ──────────────────────────────
    MAXLAT = 1.75          # posicao das barras "não dispara"
    rowsl = []
    for fold in ("A", "B"):
        for y in YEARS_POST:
            def _get(dn):
                s = lat[(lat["fold"] == fold) & (lat["detector"] == dn)
                        & (lat["year"] == y)]
                if s.empty or float(s["delay_trajs"].iloc[0]) < 0:
                    return np.nan
                return float(s["delay_games"].iloc[0])
            rowsl.append(dict(fold=fold, year=y,
                              ade=_get(DET_ADE), acc=_get(DET_ACC)))
    L = pd.DataFrame(rowsl)
    L = L.sort_values(["year", "fold"], ascending=[False, True]).reset_index(drop=True)
    ypos = np.arange(len(L))
    n_first = 0
    for i, r in L.iterrows():
        a, b = r["acc"], r["ade"]
        a_p = MAXLAT if np.isnan(a) else a
        b_p = MAXLAT if np.isnan(b) else b
        acc_first = (not np.isnan(a)) and (np.isnan(b) or a < b)
        n_first += int(acc_first)
        ax_lat.plot([a_p, b_p], [i, i], color=BASELINE if not acc_first else FAINT,
                    lw=2.2, solid_capstyle="round", zorder=2)
        ax_lat.scatter([b_p], [i], s=46, color=C_ADE, zorder=4,
                       marker="o" if not np.isnan(b) else "X",
                       edgecolors=SURFACE, linewidths=1.0)
        ax_lat.scatter([a_p], [i], s=46, color=C_ACC, zorder=5,
                       marker="o" if not np.isnan(a) else "X",
                       edgecolors=SURFACE, linewidths=1.0)
    ax_lat.set_yticks(ypos)
    ax_lat.set_yticklabels([f"{int(r['year'])}{r['fold']}" for _, r in L.iterrows()],
                           fontsize=base_fs - 3 + d, color=INK_2)
    ax_lat.set_ylim(-0.7, len(L) - 0.3)
    ax_lat.set_xlim(-0.10, MAXLAT + 0.12)
    ax_lat.axvline(MAXLAT, color=BASELINE, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax_lat.text(MAXLAT - 0.05, len(L) / 2, "sem alarme",
                fontsize=base_fs - 3.5 + d, color=MUTED, ha="center",
                va="center", rotation=90)
    ax_lat.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax_lat.set_xlabel("latência do 1º alarme no ano (jogos)",
                      fontsize=base_fs - 2 + d, labelpad=2)
    ax_lat.set_title(f"(c) Quem avisa primeiro ({n_first}/{len(L)} ano-folds)",
                     fontsize=base_fs - 0.5 + d, loc="left", color=INK, pad=6)
    ax_lat.grid(axis="x", color=GRID, lw=0.7)
    ax_lat.set_axisbelow(True)
    # rotulos diretos na linha do topo, em vez de legenda
    top = len(L) - 1
    rt = L.iloc[top]
    ax_lat.annotate("accel_p95", (rt["acc"], top), textcoords="offset points",
                    xytext=(-5, 9), ha="right", fontsize=base_fs - 3 + d,
                    color=C_ACC, annotation_clip=False)
    ax_lat.annotate("erro (ADE)", (rt["ade"], top), textcoords="offset points",
                    xytext=(5, 9), ha="left", fontsize=base_fs - 3 + d,
                    color=C_ADE, annotation_clip=False)
    ax_lat.set_ylim(-0.7, len(L) + 0.45)
    for s in ("top", "right", "left"):
        ax_lat.spines[s].set_visible(False)
    ax_lat.tick_params(axis="y", length=0)

    out_png = OUT_DIR / "fig_deteccao_narrativa.png"
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"[ok] {out_png}")
    if paper:
        for lang_dir, stem in ((PROJECT_ROOT / "Relas" / "main_pt" / "figs",
                                "fig_deteccao_pt"),):
            lang_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(lang_dir / f"{stem}.pdf", facecolor=SURFACE,
                        bbox_inches="tight")
            fig.savefig(lang_dir / f"_{stem}.png", dpi=130, facecolor=SURFACE,
                        bbox_inches="tight")
            print(f"[ok] {lang_dir / (stem + '.pdf')} (+ preview png)")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="so replota a partir do cache de alarmes")
    ap.add_argument("--paper", action="store_true",
                    help="grava tambem fig_deteccao_pt.pdf em main_pt/figs/")
    args = ap.parse_args()
    if not args.plot or not ALARM_CACHE.exists():
        compute_alarms()
    make_figure(paper=args.paper)


if __name__ == "__main__":
    main()
