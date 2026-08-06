"""
drift_analise/paper_figs.py
===========================
Figuras consolidadas para o artigo final (Relas/main_pt), bilingues
(sufixos _pt/_en, convencao do repo), substituindo as Figs. 1-7 antigas:

  fig_degradacao_{pt,en}.pdf       (a) ADE medio por ano (S2S vs Kalman, 2 horizontes)
                                   (b) aumento % do ADE vs 2019
                                   (c) razao p99/p50 do ADE por trajetoria

  fig_drift_cinematico_{pt,en}.pdf (a) KS por ano vs baseline 2019 (speed/accel/turn)
                                   (b) Wasserstein relativa a 2019 (indexada, 2019=1)
                                   (c) ADE por jogo vs accel_p99 (colorido por ano)

  fig_iw_{pt,en}.pdf               ADE observado vs ponderado (LSIF barras, RuLSIF
                                   marcador) por ano + baseline 2019 + recovery

Convencoes de dataviz: paleta Okabe-Ito validada (CVD-safe); modelo -> cor,
horizonte -> estilo de linha (encoding secundario); um eixo por painel;
WD indexada a base comum; legendas fora da area de dados (abaixo da figura).

Saida: Relas/main_pt/figs/
Uso:  python drift_analise/paper_figs.py
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
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.constants import YEARS_ALL as YEARS, HORIZONS
from common.plotting import T, make_plot_style, YEAR_COLORS_OKABE_ITO as YEAR_COLORS

OUT = PROJECT_ROOT / "Relas" / "main_pt" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["pt", "en"]


# ── estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update(make_plot_style(**{
    "axes.titlesize": 13.5,
    "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 1.8,
    "axes.axisbelow": True,
}))

# Paleta Okabe-Ito (validada: CVD deutan/tritan >= 17)
C_S2S, C_KAL = "#0072B2", "#D55E00"
C_SPEED, C_ACCEL, C_TURN = "#0072B2", "#E69F00", "#009E73"
LS = {"30→15": "-", "60→30": "--"}


def load_traj() -> pd.DataFrame:
    df = pd.read_parquet("covariate_shift_out/trajectory_errors_sample.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df[df["year"].isin(YEARS)]


def _year_axis(ax, lang: str) -> None:
    ax.set_xticks([int(y) for y in YEARS])
    ax.set_xticklabels([str(y)[2:] for y in YEARS])
    ax.set_xlabel(T("ano (20xx)", "year (20xx)", lang))


def _save(fig: plt.Figure, stem: str, lang: str) -> None:
    fig.savefig(OUT / f"{stem}_{lang}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"_{stem}_{lang}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT / f'{stem}_{lang}.pdf'}")


# ── Figura 1: degradacao consolidada ─────────────────────────────────────────

def fig_degradacao(df: pd.DataFrame, lang: str) -> None:
    agg = (df.groupby(["model", "horizon", "year"])["ade_traj"]
             .agg(mean="mean",
                  p50=lambda s: np.percentile(s, 50),
                  p99=lambda s: np.percentile(s, 99))
             .reset_index())
    agg["p99_p50"] = agg["p99"] / agg["p50"]

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1))

    def _fmt(v: float, spec: str) -> str:
        txt = spec.format(v)
        return txt.replace(".", ",") if lang == "pt" else txt

    def _stacked_labels(ax, pts, spec):
        # rotulos por ponto com separacao minima vertical por ano: cada
        # rotulo fica proximo do seu ponto, mas quando as series quase
        # coincidem os rotulos sao empurrados para baixo em pilha
        all_y = [y for _, y, _ in pts]
        yr_ = (max(all_y) - min(all_y)) or 1.0
        up, sep = 0.035 * yr_, 0.085 * yr_
        cols: dict = {}
        for x, y, c in pts:
            cols.setdefault(x, []).append((y, c))
        for x, grp in cols.items():
            grp.sort(key=lambda t: -t[0])
            prev = None
            for y, c in grp:
                yl = y + up
                if prev is not None and yl > prev - sep:
                    yl = prev - sep
                ax.text(x, yl, _fmt(y, spec), ha="center", va="bottom",
                        fontsize=5.2, color=c, zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.3,
                                                    foreground="white")])
                prev = yl

    pts_a, pts_b, pts_c = [], [], []
    for model, color in [("Seq2Seq", C_S2S), ("Kalman", C_KAL)]:
        for hz in HORIZONS:
            sub = agg[(agg["model"] == model) & (agg["horizon"] == hz)]
            sub = sub.sort_values("year")
            yrs = sub["year"].astype(int).values

            axes[0].plot(yrs, sub["mean"], LS[hz], color=color, marker="o",
                         ms=3.5, label=f"{model} {hz}")
            base = float(sub.loc[sub["year"] == 2019, "mean"].iloc[0])
            pct = 100.0 * (sub["mean"] - base) / base
            axes[1].plot(yrs, pct, LS[hz], color=color, marker="o", ms=3.5)
            axes[2].plot(yrs, sub["p99_p50"], LS[hz], color=color,
                         marker="o", ms=3.5)

            pts_a += [(x, y, color) for x, y in zip(yrs, sub["mean"])]
            pts_b += [(x, y, color) for x, y in zip(yrs, pct) if x != 2019]
            pts_c += [(x, y, color) for x, y in zip(yrs, sub["p99_p50"])]

    _stacked_labels(axes[0], pts_a, "{:.1f}")
    _stacked_labels(axes[1], pts_b, "{:.0f}")
    _stacked_labels(axes[2], pts_c, "{:.2f}")

    axes[0].set_ylabel(T("ADE médio (mm)", "Mean ADE (mm)", lang))
    axes[0].set_title(T("(a) ADE por ano", "(a) ADE per year", lang))
    # margem superior para o rotulo do ponto mais alto nao ser cortado
    axes[0].set_ylim(0, float(agg["mean"].max()) * 1.12)

    axes[1].axhline(0, color="#666", lw=0.8)
    axes[1].set_ylabel(T("aumento do ADE (%)", "ADE increase (%)", lang))
    axes[1].set_title(T("(b) aumento vs. 2019", "(b) increase vs. 2019", lang))
    axes[1].margins(y=0.14)

    axes[2].set_ylabel(T("razão p99/p50 do ADE", "ADE p99/p50 ratio", lang))
    axes[2].set_title(T("(c) cauda (p99/p50)", "(c) tail (p99/p50)", lang))
    axes[2].margins(y=0.16)

    for ax in axes:
        _year_axis(ax, lang)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=False,
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout()
    _save(fig, "fig_degradacao", lang)


# ── Figura 2: drift cinematico consolidado ───────────────────────────────────

def fig_drift_cinematico(df: pd.DataFrame, lang: str) -> None:
    kw = pd.read_csv("covariate_shift_out/ks_wd_per_dataset.csv")
    ctx = pd.read_csv("covariate_shift_out/per_game_context.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1))

    feat_style = [
        ("speed", C_SPEED, T("velocidade", "speed", lang)),
        ("accel", C_ACCEL, T("aceleração", "acceleration", lang)),
        ("turn",  C_TURN,  T("giro", "turn", lang)),
    ]

    for feat, color, label in feat_style:
        sub = kw[kw["feature"] == feat]
        g = sub.groupby("year").agg(ks=("ks", "mean"),
                                    wd=("wasserstein", "mean"))
        g = g.reindex(YEARS)
        axes[0].plot(YEARS, g["ks"], "-", color=color, marker="o",
                     ms=3.5, label=label)
        wd_base = float(g.loc[2019, "wd"])
        axes[1].plot(YEARS, g["wd"] / wd_base, "-", color=color,
                     marker="o", ms=3.5, label=label)
        axes[0].scatter(sub["year"], sub["ks"], s=8, color=color, alpha=0.30,
                        edgecolors="none")

    axes[0].set_ylabel(T("KS vs. baseline 2019", "KS vs. 2019 baseline", lang))
    axes[0].set_title(T("(a) KS por ano", "(a) KS per year", lang))

    axes[1].axhline(1.0, color="#666", lw=0.8)
    axes[1].set_ylabel(T("WD relativa", "relative WD", lang))
    axes[1].set_title(T("(b) WD relativa (2019 = 1)",
                        "(b) relative WD (2019 = 1)", lang))

    for ax in axes[:2]:
        _year_axis(ax, lang)

    # (c) ADE por jogo vs accel_p99
    s2s = df[(df["model"] == "Seq2Seq") & (df["horizon"] == "30→15")]
    ade_game = (s2s.groupby("proc_set_file")["ade_traj"].mean()
                    .rename("ade").reset_index())
    m = ade_game.merge(ctx[["proc_set_file", "year", "accel_p99"]],
                       on="proc_set_file", validate="one_to_one")

    for y in YEARS:
        sy = m[m["year"] == y]
        axes[2].scatter(sy["accel_p99"], sy["ade"], s=26,
                        color=YEAR_COLORS[y], label=str(y),
                        edgecolors="white", linewidths=0.6)
    x, yv = m["accel_p99"].values, m["ade"].values
    b1, b0 = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    rng = np.random.default_rng(42)
    boots = np.empty((2000, len(xs)))
    for i in range(2000):
        idx = rng.integers(0, len(x), len(x))
        bb1, bb0 = np.polyfit(x[idx], yv[idx], 1)
        boots[i] = bb0 + bb1 * xs
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    axes[2].plot(xs, b0 + b1 * xs, color="#333", lw=1.4)
    axes[2].fill_between(xs, lo, hi, color="#333", alpha=0.12,
                         edgecolor="none")
    rho = pd.Series(x).corr(pd.Series(yv), method="spearman")
    axes[2].text(0.03, 0.96, f"Spearman $\\rho$ = {rho:.2f}",
                 transform=axes[2].transAxes, va="top", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[2].set_xlabel("accel_p99")
    axes[2].set_ylabel(T("ADE do jogo (mm)", "per-game ADE (mm)", lang))
    axes[2].set_title(T("(c) ADE vs. accel_p99", "(c) ADE vs. accel_p99", lang))

    h_feat, l_feat = axes[0].get_legend_handles_labels()
    fig.legend(h_feat, l_feat, loc="lower center", ncol=3,
               bbox_to_anchor=(0.24, -0.06), frameon=False,
               columnspacing=1.4)
    h_yr, l_yr = axes[2].get_legend_handles_labels()
    fig.legend(h_yr, l_yr, loc="lower center", ncol=6,
               bbox_to_anchor=(0.76, -0.06), frameon=False,
               columnspacing=0.9, handletextpad=0.3)

    fig.tight_layout()
    _save(fig, "fig_drift_cinematico", lang)
    if lang == "pt":
        print(f"  (rho={rho:.3f})")


# ── Figura 3: IW em painel unico ─────────────────────────────────────────────

def fig_iw(lang: str) -> None:
    base_dir = Path("Relas/results/drift/03_covariate_shift_e_explicacao")
    lsif = pd.read_csv(base_dir / "iw_decomposition_lsif.csv")

    years = lsif["year"].astype(int).tolist()
    x = np.arange(len(years))
    w = 0.34
    ade_2019 = float(lsif["ade_2019"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.6, 3.4))

    ax.bar(x - w / 2, lsif["ade_y"], w, color="#9aa0a6",
           label=T("ADE observado", "Observed ADE", lang), zorder=2)
    ax.errorbar(x - w / 2, lsif["ade_y"],
                yerr=[lsif["ade_y"] - lsif["ade_y_ci_lo"],
                      lsif["ade_y_ci_hi"] - lsif["ade_y"]],
                fmt="none", ecolor="#333", lw=1.1, capsize=2.5, zorder=4)
    ax.bar(x + w / 2, lsif["ade_iw"], w, color=C_S2S,
           label=T("ADE ponderado (LSIF)", "Weighted ADE (LSIF)", lang),
           zorder=2)
    ax.errorbar(x + w / 2, lsif["ade_iw"],
                yerr=[lsif["ade_iw"] - lsif["ade_iw_ci_lo"],
                      lsif["ade_iw_ci_hi"] - lsif["ade_iw"]],
                fmt="none", ecolor="#333", lw=1.1, capsize=2.5, zorder=4)

    base_val = f"{ade_2019:.2f}"
    if lang == "pt":
        base_val = base_val.replace(".", ",")
    ax.axhline(ade_2019, color="#111", lw=1.2, ls="--", zorder=3,
               label=T(f"baseline 2019 ({base_val} mm)",
                       f"2019 baseline ({base_val} mm)", lang))

    for i, (_, r) in enumerate(lsif.iterrows()):
        top = max(r["ade_y_ci_hi"], r["ade_iw_ci_hi"]) + 0.12
        if np.isfinite(r["recovery_pct"]):
            ax.text(x[i], top, f"rec. {r['recovery_pct']:.0f}%",
                    ha="center", va="bottom", fontsize=10)
        else:
            ax.text(x[i], top, T("sem excesso", "no excess", lang),
                    ha="center", va="bottom", fontsize=10, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel(T("ano", "year", lang))
    ax.set_ylabel("ADE (mm)")
    ax.set_ylim(0, float(lsif["ade_y_ci_hi"].max()) + 1.0)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07),
               frameon=False, columnspacing=1.4)
    fig.tight_layout()
    _save(fig, "fig_iw", lang)


def fig_finetune(lang: str) -> None:
    """Comparativo de estrategias de adaptacao (protocolo do Mes 7).

    (a) cat_ratio por estrategia (EWC como plato do sweep de lambda);
    (b) ADE apos adaptacao, regime antigo vs novos, com linhas de
        referencia do modelo original.
    So entram runs do MESMO protocolo (ade_before_base = 4.5843 mm), agora
    com regime antigo = 2019 e dados novos = 2022-2025 (2021 excluido como
    outlier): mes7_no2021/{replay,ewc,targeted}_results.csv.
    """
    mes7 = Path("Relas/results/mes7_no2021")
    rep  = pd.read_csv(mes7 / "replay_results.csv")
    ctrl = rep[rep["replay_ratio"] == 0.0].iloc[0]
    rep05 = rep[rep["replay_ratio"] == 0.5].iloc[0]
    ewc  = pd.read_csv(mes7 / "ewc_results.csv")
    ewc  = ewc[ewc["ewc_lambda"] > 0]          # plato da regularizacao
    tgt  = pd.read_csv(mes7 / "targeted_results.csv").set_index("variant")

    def row(tag, label, r):
        return dict(tag=tag, label=label,
                    cat=float(r["cat_ratio"]),
                    base=float(r["ade_after_base"]),
                    test=float(r["ade_after_test"]))

    methods = [
        row("ewc", T("EWC ($\\lambda$ 50–$10^{7}$)",
                     "EWC ($\\lambda$ 50–$10^{7}$)", lang),
            ewc.median(numeric_only=True)),
        row("ctrl", T("FT conservador (r=0)",
                      "Conservative FT (r=0)", lang), ctrl),
        row("rep", T("Replay uniforme", "Uniform replay", lang), rep05),
        row("rep_iw", T("Replay IW", "IW replay", lang),
            tgt.loc["replay_iw"]),
        row("tgt", T("Seletivo accel", "Accel-targeted", lang),
            tgt.loc["target_accel"]),
        row("tgt_iw", T("Seletivo + replay IW", "Targeted + IW replay", lang),
            tgt.loc["target_accel+replay_iw"]),
    ]
    dfm = pd.DataFrame(methods)
    new_tags = {"rep_iw", "tgt", "tgt_iw"}
    ade_before_base, ade_before_test = 4.5843, 4.9649

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    x = np.arange(len(dfm))

    # (a) cat_ratio
    colors = [C_S2S if t in new_tags else "#9aa0a6" for t in dfm["tag"]]
    axes[0].bar(x, dfm["cat"], 0.62, color=colors, zorder=2)
    # plato do EWC como faixa min-max
    axes[0].errorbar([0], [dfm.loc[0, "cat"]],
                     yerr=[[dfm.loc[0, "cat"] - ewc["cat_ratio"].min()],
                           [ewc["cat_ratio"].max() - dfm.loc[0, "cat"]]],
                     fmt="none", ecolor="#333", lw=1.1, capsize=3, zorder=4)
    for i, v in enumerate(dfm["cat"]):
        axes[0].text(i, v + 0.008, f"{v:.3f}".replace(".", "," if lang == "pt" else "."),
                     ha="center", va="bottom", fontsize=10)
    axes[0].axhline(float(ctrl["cat_ratio"]), color="#666", lw=0.9, ls=":")
    axes[0].set_ylabel("cat_ratio")
    axes[0].set_ylim(0.40, 0.48)
    axes[0].set_title(T("(a) esquecimento (menor = melhor)",
                        "(a) forgetting (lower is better)", lang))

    # (b) ADE apos, regime antigo vs novos
    w = 0.36
    axes[1].bar(x - w / 2, dfm["base"], w, color="#9aa0a6",
                label=T("regime antigo (2019)", "old regime (2019)", lang),
                zorder=2)
    axes[1].bar(x + w / 2, dfm["test"], w, color=C_S2S,
                label=T("dados novos (2022–2025)", "new data (2022–2025)", lang),
                zorder=2)
    axes[1].axhline(ade_before_base, color="#666", lw=1.1, ls="--", zorder=3,
                    label=T("original, antigo", "original, old", lang))
    axes[1].axhline(ade_before_test, color=C_KAL, lw=1.1, ls="--", zorder=3,
                    label=T("original, novos", "original, new", lang))
    # rotulos de valor sobre cada barra (2 casas, virgula no pt)
    def _fmt(v: float, d: int = 2) -> str:
        return f"{v:.{d}f}".replace(".", "," if lang == "pt" else ".")
    for i in range(len(dfm)):
        axes[1].text(i - w / 2, dfm.loc[i, "base"] + 0.04, _fmt(dfm.loc[i, "base"]),
                     ha="center", va="bottom", fontsize=7.5, color="#555", zorder=5)
        axes[1].text(i + w / 2, dfm.loc[i, "test"] + 0.04, _fmt(dfm.loc[i, "test"]),
                     ha="center", va="bottom", fontsize=7.5, color=C_S2S, zorder=5)
    axes[1].set_ylabel("ADE (mm)")
    axes[1].set_ylim(4.3, 5.35)
    axes[1].set_title(T("(b) ADE após adaptação",
                        "(b) ADE after adaptation", lang))

    labels = dfm["label"].tolist()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5)
    axes[1].legend(loc="upper right", framealpha=0.95, fontsize=9, ncol=2, columnspacing=1.0)

    fig.tight_layout()
    _save(fig, "fig_finetune", lang)


def main() -> None:
    df = load_traj()
    for lang in LANGS:
        fig_degradacao(df, lang)
        fig_drift_cinematico(df, lang)
        fig_iw(lang)
        if (Path("Relas/results/mes7") / "targeted_results.csv").exists():
            fig_finetune(lang)


if __name__ == "__main__":
    main()
