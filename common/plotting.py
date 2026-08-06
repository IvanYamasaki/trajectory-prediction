"""Shared plotting style, palettes and figure/markdown helpers.

Canonical sources: drift_analise/chapter02_deteccao_pipeline.py,
phase1_smoothing.py, chapter03_visuals.py, mes11_accel_battery.py,
fig_deteccao_narrativa.py, phase2_grid_search.py, chapter01_descriptive_pipeline.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

BASE_PLOT_STYLE = {
    "font.family": "serif", "font.size": 13,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "mathtext.fontset": "cm", "lines.linewidth": 2,
    "axes.grid": True, "grid.alpha": 0.25,
    "figure.autolayout": False,
}


def make_plot_style(**overrides):
    return {**BASE_PLOT_STYLE, **overrides}


def T(pt: str, en: str, lang: str) -> str:
    return en if lang == "en" else pt


YEAR_COLORS = {
    2019: "#1a7abf", 2021: "#e67e22", 2022: "#27ae60",
    2023: "#8e44ad", 2024: "#c0392b", 2025: "#16a085",
}

YEAR_COLORS_TAB10 = {2019: "#1f77b4", 2021: "#ff7f0e", 2022: "#2ca02c",
                     2023: "#d62728", 2024: "#9467bd", 2025: "#8c564b"}

# Paleta Okabe-Ito (validada: CVD deutan/tritan >= 17) — de paper_figs.py
YEAR_COLORS_OKABE_ITO = {2019: "#0072B2", 2021: "#D55E00", 2022: "#009E73",
                         2023: "#CC79A7", 2024: "#E69F00", 2025: "#56B4E9"}

# paleta categorica validada (dataviz reference, modo claro) — ordem fixa
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE  = "#fcfcfb"
FAINT    = "#d8d7cf"


def df_to_md(df: pd.DataFrame, floatfmt: str | None = None) -> str:
    """Markdown table; avoids optional ``tabulate`` for ``DataFrame.to_markdown``."""
    float_fmt = ".3f" if floatfmt is None else floatfmt
    try:
        return df.to_markdown(index=False, floatfmt=float_fmt)
    except ImportError:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        def fv(v):
            return format(v, float_fmt) if isinstance(v, float) else str(v)
        rows = ["| " + " | ".join(fv(v) for v in row) + " |"
                for _, row in df.iterrows()]
        return "\n".join([header, sep] + rows)


def save_fig_bilingual(fig, name, out_dir):
    """Salva 1 figura em PT e EN (PDF) em out_dir.
    `name` deve ser sem extensão; sufixos `_pt.pdf` e `_en.pdf` são adicionados.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ["pt", "en"]:
        out = out_dir / f"{name}_{lang}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
