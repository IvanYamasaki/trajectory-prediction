"""
plot_iw_results.py  —  Visualizações do Importance Weighting (Frente 2)
========================================================================

Gera figuras para a Seção 13 do notebook:
  - iw_ade_comparison.pdf : ADE direto vs ADE-IW por ano, com IC95 bootstrap
  - iw_recovery.pdf       : barra de recovery_pct por ano com flag de ESS
  - iw_weights_dist.pdf   : distribuição dos pesos por ano

Uso:
    MPLBACKEND=Agg python drift_analise/plot_iw_results.py
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
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

OUT_DIR = Path("Relas") / "results" / "mes3"
COV_DIR = Path("covariate_shift_out")
IW_CSV  = OUT_DIR / "iw_decomposition.csv"

COLORS = {"ade_y": "#d62728", "ade_iw": "#2ca02c", "ade_2019": "#1f77b4"}
ESS_THRESH = 0.3


def load_iw() -> pd.DataFrame:
    if not IW_CSV.exists():
        raise FileNotFoundError(f"Execute compute_importance_weights.py primeiro: {IW_CSV}")
    return pd.read_csv(IW_CSV)


def plot_ade_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    years = df["year"].astype(int).tolist()
    x = np.arange(len(years))
    w = 0.30

    # Barras ADE direto
    ax.bar(x - w/2, df["ade_y"], width=w, color=COLORS["ade_y"],
           alpha=0.85, label="ADE direto (ano y)")
    ax.errorbar(x - w/2, df["ade_y"],
                yerr=[df["ade_y"] - df["ade_y_ci_lo"],
                      df["ade_y_ci_hi"] - df["ade_y"]],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    # Barras ADE-IW
    iw_vals = df["ade_iw"].where(df["ess_stable"], np.nan)
    iw_lo   = (df["ade_iw"] - df["ade_iw_ci_lo"]).where(df["ess_stable"], np.nan)
    iw_hi   = (df["ade_iw_ci_hi"] - df["ade_iw"]).where(df["ess_stable"], np.nan)
    ax.bar(x + w/2, iw_vals, width=w, color=COLORS["ade_iw"],
           alpha=0.85, label="ADE importance-weighted")
    ax.errorbar(x + w/2, iw_vals,
                yerr=[iw_lo.fillna(0), iw_hi.fillna(0)],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    # Linha baseline 2019
    b = df["ade_2019"].iloc[0]
    ax.axhline(b, color=COLORS["ade_2019"], linestyle="--", linewidth=1.5,
               label=f"ADE 2019 = {b:.2f} mm")

    # ESS instável
    for i, row in enumerate(df.itertuples()):
        if not row.ess_stable:
            ax.text(i + w/2, 0.2, f"ESS={row.ess_ratio:.2f}\n⚠",
                    ha="center", va="bottom", fontsize=8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Ano")
    ax.set_ylabel("ADE médio (mm)")
    ax.set_title("ADE direto vs ADE importance-weighted por ano\n(Seq2Seq 30→15, CI95 bootstrap)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    path = OUT_DIR / "iw_ade_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_recovery(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    years = df["year"].astype(int).tolist()
    x = np.arange(len(years))
    rec = df["recovery_pct"].fillna(0)
    colors = ["#2ca02c" if (r > 20 and ess) else "#d62728" if not ess else "#aec7e8"
              for r, ess in zip(rec, df["ess_stable"])]
    bars = ax.bar(x, rec, color=colors, alpha=0.85)

    for bar, ess, rv in zip(bars, df["ess_stable"], df["recovery_pct"]):
        label = f"{rv:.1f}%" if pd.notna(rv) else "n/a"
        if not ess:
            label += "\n⚠ESS"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha="center", va="bottom", fontsize=9)

    ax.axhline(0,   color="black", linewidth=0.8)
    ax.axhline(100, color="gray",  linewidth=0.8, linestyle="--", label="100% recovery")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Recovery (% excesso de ADE eliminado)")
    ax.set_title("Recovery por importance-weighting\n(verde = criterio aceite: >20%)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "iw_recovery.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_weights_dist(df: pd.DataFrame) -> None:
    targets = df["year"].tolist()
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, year in zip(axes, targets):
        wpath = COV_DIR / f"importance_weights_{year}.npy"
        if not wpath.exists():
            ax.set_title(f"{year}\n(sem pesos)")
            continue
        w = np.load(wpath)
        ax.hist(w, bins=50, color="#5598d3", edgecolor="white", linewidth=0.4)
        ess = ((w.sum())**2) / (len(w) * (w**2).sum())
        ax.set_title(f"{year}  ESS={ess:.2f}", fontsize=10)
        ax.set_xlabel("peso w")
        ax.set_ylabel("contagem" if year == targets[0] else "")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8)
    fig.suptitle("Distribuição dos pesos p_2019(x)/p_y(x) por ano", fontsize=11)
    plt.tight_layout()
    path = OUT_DIR / "iw_weights_dist.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_iw()
    print(f"[info] {len(df)} anos no iw_decomposition.csv")
    plot_ade_comparison(df)
    plot_recovery(df)
    plot_weights_dist(df)
    print("[ok] plots IW gerados.")


if __name__ == "__main__":
    main()
