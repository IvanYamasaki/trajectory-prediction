"""
drift_analise/mes11_accel_battery.py
====================================
Mes 11 — Bateria de detectores dirigidos pelo diagnostico do covariate shift.

O capitulo de quantificacao (LSIF por feature, Tabela tab:iw_feat) mostra que
o drift e dominado pela cauda de ACELERACAO (accel_p99/p90 lideram a recovery;
velocidade e intermediaria; giro e residual). A secao de adaptacao conclui que
esse diagnostico nao ajuda a ADAPTAR, mas deve ajudar a MONITORAR. Esta bateria
testa exatamente isso, sob o protocolo de holdout do Mes 11 (folds A e B):

  Sinais por trajetoria (janela de entrada de 30 passos, SEM inferencia do
  modelo — vx,vy desnormalizados):
    accel_p95, accel_mean  (p95 e nao p99: 29 amostras/janela)
    speed_p95, speed_mean
  Detector: ADWINLiteFast + robz_w200 (variante que generalizou melhor no
  holdout), mini-grid calibrada nos jogos de calibracao, avaliada UMA vez
  nos jogos de teste.

  Referencia: detector de erro (ADE robz) — vencedor lite do fold.
  Ensembles de mesma escala (OOS): AND(ADE, accel_p95) e OR(ADE, accel_p95),
  gate da Fase 3 (deque, debounce 5000), W=200.

Saidas em Relas/results/drift/mes11_accel_battery/:
  features_ext.parquet     — cache: 4 features x 288k janelas
  battery_results.csv      — calib + teste OOS por detector x fold
  battery_latency.csv      — latencia por ano no teste
  battery_summary.png      — figura de sintese (4 paineis)
  MES11_BATTERY_REPORT.md

Uso:
    python drift_analise/mes11_accel_battery.py            # tudo
    python drift_analise/mes11_accel_battery.py --plot     # so re-gera o PNG
"""
from __future__ import annotations

import argparse
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.constants import (
    W_COINCIDENCE_DEFAULT as W_AND, RETRAIN_DEBOUNCE as DEBOUNCE,
)
from common.plotting import INK, INK_2, MUTED, GRID, BASELINE, SURFACE
from drift_analise.chapter02_deteccao_pipeline import (
    run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz
from drift_analise.phase3_ensemble import _and_gate
from drift_analise.phase4_metrics import wilson_ci_per1k
from drift_analise.mes11_holdout_validation import (
    ADWINLiteFast, split_by_games, eval_alarms, latency_by_year,
    select_winner, FAR_ADMISSIBILITY, YEARS_ALL, YEARS_POST, ROBZ_W,
)
from drift_analise.mes10_same_scale_ensemble import (
    PROC_YEAR, STATS_PATH, LOOK_BACK, LOOK_FORTH,
)

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes11_accel_battery"
FEAT_CACHE = OUT_DIR / "features_ext.parquet"
HOLDOUT_PARTIAL = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes11_holdout" / "partial"

FEATURES = ["accel_p95", "accel_mean", "speed_p95", "speed_mean"]

MINI_GRID = dict(
    delta      = [1e-6, 1e-7],
    min_window = [200, 600],
    cooldown   = [200, 1000],
    max_window = [2000],
)

# ── extracao de features estendidas ───────────────────────────────────────────

def build_ext_features() -> pd.DataFrame:
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
        v = x_all[:, :, 2:4] * std[2:4] + avg[2:4]          # (N, 30, 2)
        speed = np.linalg.norm(v, axis=2)                    # (N, 30)
        dv = np.linalg.norm(np.diff(v, axis=1), axis=2)      # (N, 29)
        feat = dict(
            accel_p95 =np.percentile(dv, 95, axis=1),
            accel_mean=dv.mean(axis=1),
            speed_p95 =np.percentile(speed, 95, axis=1),
            speed_mean=speed.mean(axis=1),
        )
        for tid in range(len(x_all)):
            rows.append((f"proc_set_{n}.pkl", int(tid),
                         *(float(feat[k][tid]) for k in FEATURES)))
        print(f"  proc_set_{n} (year={PROC_YEAR[n]}): {len(x_all):,} janelas",
              flush=True)

    out = pd.DataFrame(rows, columns=["proc_set_file", "traj_id"] + FEATURES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEAT_CACHE, index=False)
    print(f"[ok] cache -> {FEAT_CACHE} ({len(out):,} linhas)")
    return out


# ── ADE de referencia do fold (vencedor lite do holdout) ──────────────────────

def ade_winner_cfg(fold: str) -> dict:
    tag = "" if fold == "A" else f"_fold{fold}"
    parts = sorted(HOLDOUT_PARTIAL.glob(f"grid_lite_Seq2Seq_robz*{tag}.csv"))
    parts = [p for p in parts
             if (fold == "A" and "fold" not in p.name) or
                (fold != "A" and f"fold{fold}" in p.name)]
    if not parts:
        raise RuntimeError(f"grids lite S2S do fold {fold} ausentes.")
    grid = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    w = select_winner(grid)
    return dict(delta=float(w["delta"]), min_window=int(w["min_window"]),
                cooldown=int(w["cooldown"]), max_window=int(w["max_window"]))


# ── bateria ───────────────────────────────────────────────────────────────────

def run_battery() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat = build_ext_features()

    df = load_errors()
    stream = build_stream(df, model="Seq2Seq")
    stream = stream.merge(feat, on=["proc_set_file", "traj_id"],
                          how="left", validate="one_to_one")
    n_missing = int(stream[FEATURES[0]].isna().sum())
    print(f"[merge] stream={len(stream):,} | feature ausente={n_missing}")
    if n_missing > 0.01 * len(stream):
        raise RuntimeError("alinhamento (proc_set_file, traj_id) falhou")

    res_rows, lat_rows = [], []

    for fold in ("A", "B"):
        calib, test, _ = split_by_games(stream, fold)
        yv_c = calib["year"].fillna(0).astype(int).values
        yv_t = test["year"].fillna(0).astype(int).values
        n19_t = int((yv_t == 2019).sum())
        print(f"\n=== fold {fold}: calib={len(calib):,} | teste={len(test):,} "
              f"(2019 teste: {n19_t:,}) ===")

        def _record(name: str, family: str, cfg_str: str,
                    alarms_test: list[int],
                    far_calib: float | None, cov_calib: int | None) -> dict:
            m = eval_alarms(alarms_test, yv_t)
            lo, hi = wilson_ci_per1k(m["n_2019_alarms"], n19_t)
            row = dict(fold=fold, detector=name, family=family, config=cfg_str,
                       FAR_calib=far_calib, cov_calib=cov_calib,
                       wilson_lo_per1k=lo, wilson_hi_per1k=hi, **m)
            res_rows.append(row)
            for r in latency_by_year(alarms_test, test):
                lat_rows.append(dict(fold=fold, detector=name, **r))
            print(f"  [{name:<22s}] FAR_t={m['FAR_2019_per1k']:.4f} "
                  f"cov_t={m['year_coverage']} n_post={m['n_post_alarms']}")
            return row

        # referencia: detector de erro (ADE) — vencedor lite do fold
        ade_cfg = ade_winner_cfg(fold)
        ade_str = (f"delta={ade_cfg['delta']:.0e}, mw={ade_cfg['min_window']}, "
                   f"cd={ade_cfg['cooldown']}, xw={ade_cfg['max_window']}")
        sig_ade_c = smooth_robz(calib["ade_traj"].astype(float).values, ROBZ_W)
        sig_ade_t = smooth_robz(test["ade_traj"].astype(float).values, ROBZ_W)
        m_ade_c = eval_alarms(run_detector(ADWINLiteFast(**ade_cfg), sig_ade_c), yv_c)
        ade_alarms_t = run_detector(ADWINLiteFast(**ade_cfg), sig_ade_t)
        _record("ADE (erro do modelo)", "erro", ade_str, ade_alarms_t,
                m_ade_c["FAR_2019_per1k"], m_ade_c["year_coverage"])

        # detectores por feature cinematica (model-free)
        best_feat_alarms: dict[str, list[int]] = {}
        for f in FEATURES:
            sig_c = smooth_robz(calib[f].astype(float).values, ROBZ_W)
            sig_t = smooth_robz(test[f].astype(float).values, ROBZ_W)
            grid_rows = []
            for delta, mw, cd, xw in itertools.product(
                    MINI_GRID["delta"], MINI_GRID["min_window"],
                    MINI_GRID["cooldown"], MINI_GRID["max_window"]):
                det = ADWINLiteFast(delta=delta, min_window=mw,
                                    cooldown=cd, max_window=xw)
                m = eval_alarms(run_detector(det, sig_c), yv_c)
                grid_rows.append(dict(delta=delta, min_window=mw, cooldown=cd,
                                      max_window=xw, **m))
            gdf = pd.DataFrame(grid_rows)
            w = select_winner(gdf)
            cfg = dict(delta=float(w["delta"]), min_window=int(w["min_window"]),
                       cooldown=int(w["cooldown"]), max_window=int(w["max_window"]))
            cfg_str = (f"delta={cfg['delta']:.0e}, mw={cfg['min_window']}, "
                       f"cd={cfg['cooldown']}, xw={cfg['max_window']}")
            alarms_t = run_detector(ADWINLiteFast(**cfg), sig_t)
            best_feat_alarms[f] = alarms_t
            _record(f, "feature", cfg_str, alarms_t,
                    float(w["FAR_2019_per1k"]), int(w["year_coverage"]))

        # ensembles de mesma escala com accel_p95 (OOS)
        acc_t = best_feat_alarms["accel_p95"]
        and_t = _and_gate(ade_alarms_t, acc_t, W_AND, DEBOUNCE)
        or_t  = sorted(set(ade_alarms_t) | set(acc_t))
        _record("AND(ADE, accel_p95)", "ensemble",
                f"gate fase3, W={W_AND}, deb={DEBOUNCE}", and_t, None, None)
        _record("OR(ADE, accel_p95)", "ensemble", "uniao", or_t, None, None)

    res = pd.DataFrame(res_rows)
    res.to_csv(OUT_DIR / "battery_results.csv", index=False)
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(OUT_DIR / "battery_latency.csv", index=False)
    print(f"\n[ok] {OUT_DIR / 'battery_results.csv'}")
    print(f"[ok] {OUT_DIR / 'battery_latency.csv'}")

    L = [
        "# Mes 11 — Bateria de detectores dirigidos pelo diagnostico (accel)",
        "",
        "Motivacao: tab:iw_feat mostra accel_p99/p90 como vetor dominante do",
        "covariate shift. Aqui: detectores ADWIN+robz sobre features da janela",
        "de entrada (model-free), holdout 2-fold, vs o detector de erro (ADE).",
        "",
        "## Resultados no teste (out-of-sample)",
        "",
        res.drop(columns=[c for c in res.columns if c.startswith("n_2")
                          and c != "n_2019_alarms"]).to_markdown(index=False),
        "",
        "## Alarmes por ano (teste)",
        "",
        res[["fold", "detector"] + [f"n_{y}" for y in YEARS_ALL]]
            .to_markdown(index=False),
        "",
        "## Latencia (teste)",
        "",
        lat.to_markdown(index=False),
    ]
    (OUT_DIR / "MES11_BATTERY_REPORT.md").write_text("\n".join(L),
                                                     encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'MES11_BATTERY_REPORT.md'}")


# ── figura de sintese ─────────────────────────────────────────────────────────

# paleta categorica validada (dataviz reference, modo claro) — em common.plotting
SERIES = {                      # identidade fixa por detector
    "ADE (erro do modelo)": "#2a78d6",   # slot 1 blue
    "accel_p95":            "#1baf7a",   # slot 2 aqua
    "accel_mean":           "#eda100",   # slot 3 yellow
    "speed_p95":            "#4a3aa7",   # slot 5 violet
    "speed_mean":           "#e87ba4",   # slot 7 magenta
    "AND(ADE, accel_p95)":  "#e34948",   # slot 6 red
    "OR(ADE, accel_p95)":   "#eb6834",   # slot 8 orange
}
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

ORDER = ["ADE (erro do modelo)", "accel_p95", "accel_mean",
         "speed_p95", "speed_mean", "AND(ADE, accel_p95)", "OR(ADE, accel_p95)"]


def make_summary_png(paper: bool = False) -> None:
    res = pd.read_csv(OUT_DIR / "battery_results.csv")
    lat = pd.read_csv(OUT_DIR / "battery_latency.csv")

    base_fs = 13 if paper else 10
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": base_fs,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.grid": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": base_fs - 1, "ytick.labelsize": base_fs - 1,
    })

    d = 3 if paper else 0          # incremento de fonte no modo paper
    fig = plt.figure(figsize=(12.5, 9.0) if paper else (13.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.12 if paper else 0.10, right=0.97,
                          top=0.96 if paper else 0.90, bottom=0.07)
    ax_far  = fig.add_subplot(gs[0, 0])
    ax_gen  = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_lat  = fig.add_subplot(gs[1, 1])

    dets = [d for d in ORDER if d in set(res["detector"])]

    # ── (a) FAR teste com IC Wilson, por detector x fold ─────────────────────
    ypos = {d: len(dets) - 1 - i for i, d in enumerate(dets)}
    for _, r in res.iterrows():
        dn = r["detector"]
        y = ypos[dn] + (0.16 if r["fold"] == "A" else -0.16)
        c = SERIES[dn]
        ax_far.plot([r["wilson_lo_per1k"], r["wilson_hi_per1k"]], [y, y],
                    color=c, lw=2, solid_capstyle="round", alpha=0.55, zorder=2)
        ax_far.scatter([r["FAR_2019_per1k"]], [y], s=52, color=c, zorder=3,
                       marker="o" if r["fold"] == "A" else "s",
                       edgecolors=SURFACE, linewidths=1.2)
        cov = int(r["year_coverage"])
        ax_far.annotate(f"{cov}/5", (r["wilson_hi_per1k"], y),
                        textcoords="offset points", xytext=(6, -3),
                        fontsize=8+d, color=INK_2)
    ax_far.axvline(FAR_ADMISSIBILITY, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax_far.text(FAR_ADMISSIBILITY, -0.62, " teto 0,20/1k",
                fontsize=8+d, color=INK_2, va="top", ha="left")
    ax_far.set_ylim(-0.75, len(dets) - 0.4)
    ax_far.set_yticks([ypos[d] for d in dets])
    ax_far.set_yticklabels(dets, fontsize=9+d, color=INK)
    ax_far.set_xlabel("FAR 2019 no teste (/1k trajs) — IC Wilson 95%")
    ax_far.set_title("(a) Falsos alarmes fora da amostra — anotação = cobertura",
                     fontsize=10.5+d, loc="left", color=INK, pad=8)
    ax_far.grid(axis="x", color=GRID, lw=0.7)
    ax_far.set_axisbelow(True)
    ax_far.scatter([], [], marker="o", color=INK_2, label="fold A")
    ax_far.scatter([], [], marker="s", color=INK_2, label="fold B")
    ax_far.legend(loc="lower right", fontsize=8+d, frameon=False)

    # ── (b) generalizacao: FAR calib vs FAR teste ────────────────────────────
    sel = res.dropna(subset=["FAR_calib"])
    lim = max(0.42, float(sel[["FAR_calib", "FAR_2019_per1k"]].max().max()) * 1.15)
    ax_gen.plot([0, lim], [0, lim], color=BASELINE, lw=1, zorder=1)
    ax_gen.axhline(FAR_ADMISSIBILITY, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax_gen.axvline(FAR_ADMISSIBILITY, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for _, r in sel.iterrows():
        dn = r["detector"]
        ax_gen.scatter([r["FAR_calib"]], [r["FAR_2019_per1k"]], s=52,
                       color=SERIES[dn], zorder=3,
                       marker="o" if r["fold"] == "A" else "s",
                       edgecolors=SURFACE, linewidths=1.2)
    ax_gen.set_xlabel("FAR 2019 na calibração (/1k)")
    ax_gen.set_ylabel("FAR 2019 no teste (/1k)")
    ax_gen.set_title("(b) Generalização da seleção — diagonal = replica perfeita",
                     fontsize=10.5+d, loc="left", color=INK, pad=8)
    ax_gen.set_xlim(-0.01, lim)
    ax_gen.set_ylim(-0.01, lim)
    ax_gen.grid(color=GRID, lw=0.7)
    ax_gen.set_axisbelow(True)

    # ── (c) alarmes por ano no teste (heatmap sequencial) ────────────────────
    hm_dets = dets
    counts = np.zeros((len(hm_dets), len(YEARS_POST)))
    for i, dn in enumerate(hm_dets):
        sub = res[res["detector"] == dn]
        for j, y in enumerate(YEARS_POST):
            counts[i, j] = sub[f"n_{y}"].sum()      # soma dos 2 folds
    vmax = max(1.0, counts.max())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#ffffff"] + SEQ_BLUES)
    im = ax_heat.imshow(counts, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    for i in range(len(hm_dets)):
        for j in range(len(YEARS_POST)):
            v = int(counts[i, j])
            ax_heat.text(j, i, str(v), ha="center", va="center", fontsize=9+d,
                         color=SURFACE if counts[i, j] > 0.6 * vmax else INK)
    ax_heat.set_xticks(range(len(YEARS_POST)))
    ax_heat.set_xticklabels([str(y) for y in YEARS_POST])
    ax_heat.set_yticks(range(len(hm_dets)))
    ax_heat.set_yticklabels(hm_dets, fontsize=9+d, color=INK)
    ax_heat.set_title("(c) Alarmes por ano no teste (soma dos folds A+B)",
                      fontsize=10.5+d, loc="left", color=INK, pad=8)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # ── (d) latencia por ano (jogos), media dos folds ────────────────────────
    lat_ok = lat[lat["delay_trajs"] >= 0]
    show = ["ADE (erro do modelo)", "accel_p95", "OR(ADE, accel_p95)"]
    label_dy = {"ADE (erro do modelo)": 0, "accel_p95": 9,
                "OR(ADE, accel_p95)": -11}
    label_txt = {"ADE (erro do modelo)": "ADE", "accel_p95": "accel_p95",
                 "OR(ADE, accel_p95)": "OR(ADE, accel)"}
    for dn in show:
        sub = (lat_ok[lat_ok["detector"] == dn]
               .groupby("year")["delay_games"].mean())
        yrs = [y for y in YEARS_POST if y in sub.index]
        ax_lat.plot(yrs, [sub[y] for y in yrs], "-", color=SERIES[dn], lw=2,
                    marker="o", ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.0)
        if yrs:
            ax_lat.annotate(label_txt[dn], (yrs[-1], sub[yrs[-1]]),
                            textcoords="offset points",
                            xytext=(8, label_dy[dn]),
                            fontsize=8.5+d, color=SERIES[dn], va="center")
    miss = lat[lat["delay_trajs"] < 0]
    for dn in show:
        for _, r in miss[miss["detector"] == dn].iterrows():
            ax_lat.scatter([r["year"]], [0], marker="x", s=40,
                           color=SERIES[dn], zorder=3)
    ax_lat.set_xticks(YEARS_POST)
    ax_lat.set_xlabel("ano")
    ax_lat.set_ylabel("latência do 1º alarme (jogos, média dos folds)")
    ax_lat.set_title("(d) Latência de detecção no teste — × = ano sem alarme",
                     fontsize=10.5+d, loc="left", color=INK, pad=8)
    ax_lat.grid(color=GRID, lw=0.7)
    ax_lat.set_axisbelow(True)
    ax_lat.set_xlim(2020.6, 2026.2)

    if paper:
        figs_dir = PROJECT_ROOT / "Relas" / "main_pt" / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs_dir / "fig_deteccao_pt.pdf", facecolor=SURFACE,
                    bbox_inches="tight")
        fig.savefig(figs_dir / "_fig_deteccao_pt.png", dpi=130,
                    facecolor=SURFACE, bbox_inches="tight")
        print(f"[ok] {figs_dir / 'fig_deteccao_pt.pdf'} (+ preview png)")
    else:
        fig.suptitle("Bateria Mês 11 — detectores de erro (ADE) vs. features "
                     "cinemáticas (model-free), holdout 2-fold out-of-sample",
                     fontsize=12, color=INK, x=0.10, ha="left")
        out = OUT_DIR / "battery_summary.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        print(f"[ok] {out}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="so re-gera as figuras a partir dos CSVs")
    ap.add_argument("--paper", action="store_true",
                    help="gera tambem fig_deteccao_pt.pdf em main_pt/figs/")
    args = ap.parse_args()
    if not args.plot:
        run_battery()
    make_summary_png(paper=False)
    if args.paper:
        make_summary_png(paper=True)


if __name__ == "__main__":
    main()
