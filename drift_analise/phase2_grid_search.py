"""
drift_analise/phase2_grid_search.py
=====================================
Fase 2 do Plano de Ensaios — Grid Search de Hiperparametros.

3 frentes (redesenhadas a partir dos achados da Fase 1):

  Frente A — ADWIN + robz_w200 pre-processamento fixo
              Grid: delta x min_window x cooldown x max_window (96 configs)

  Frente B — PH com agregacao por janela de N trajetorias (pivot estrategico)
              Restaura suposicao i.i.d. via TLC. Grid: N x K_thr x K_delta (60 configs)

  Frente C — KSWIN (Kolmogorov-Smirnov Windowed), puro scipy, sem river
              Testa mudancas distribucionais; insensivel a autocorrelacao linear.
              Grid: alpha x window_size x stat_size (18 configs)

Total: 174 configs x 2 modelos (Seq2Seq, Kalman).

Saidas em Relas/results/drift/mes9_phase2/:
  phase2_results.csv   -- tabela completa de metricas
  PHASE2_REPORT.md     -- relatorio com ranking, heatmaps e recomendacao para Fase 3
  images/frente_A/     -- heatmaps + top-5 stream plots
  images/frente_B/
  images/frente_C/

Uso:
    python drift_analise/phase2_grid_search.py
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.constants import YEARS_ALL, PERFECT_SNR
from common.detection import (
    smooth_robz, alarms_per_year, compute_snr, adwin_variant_config,
)
from common.plotting import YEAR_COLORS, make_plot_style, df_to_md as _df_to_md
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, ADWINExact, PageHinkley, run_detector,
    load_errors, build_stream,
)

# ── paths ─────────────────────────────────────────────────────────────────────
# Variant of ADWIN used in Frente A.  Set ADWIN_VARIANT="exact" to use
# river.drift.ADWIN (Bifet & Gavalda 2007), or "lite" for the legacy
# midpoint-z ADWINLite.  Results land in different folders so legacy runs
# remain reproducible.
ADWIN_VARIANT, _OUT_SUFFIX = adwin_variant_config()
ADWIN_CLASS = ADWINExact if ADWIN_VARIANT == "exact" else ADWINLite

_OUT_NAME = f"mes9_phase2{_OUT_SUFFIX}"
OUT_ROOT = PROJECT_ROOT / "Relas" / "results" / "drift" / _OUT_NAME
IMG_A    = OUT_ROOT / "images" / "frente_A"
IMG_B    = OUT_ROOT / "images" / "frente_B"
IMG_C    = OUT_ROOT / "images" / "frente_C"

PLOT_STYLE = make_plot_style(**{
    "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8,  "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
    # o arquivo nao definia mathtext.fontset; preserva o default do matplotlib
    "mathtext.fontset": "dejavusans",
})
plt.rcParams.update(PLOT_STYLE)

# ── grids ─────────────────────────────────────────────────────────────────────
GRID_A = dict(
    delta      = [1e-4, 1e-5, 1e-6, 1e-7],
    min_window = [200,  400,  600,  1000],
    cooldown   = [200,  500,  1000],
    max_window = [2000, 5000],
)

GRID_B = dict(
    N_window = [50,  100, 200, 500],
    K_thr    = [5,   10,  20,  30, 50],
    K_delta  = [0.5, 1.0, 2.0],
)

GRID_C = dict(
    alpha       = [0.001, 1e-4, 1e-5],
    window_size = [200,   500,  1000],
    stat_size   = [50,    100],
)

# ── KSWIN (puro scipy, sem river) ─────────────────────────────────────────────
class KSWINDetector:
    """KSWIN drift detector baseado em scipy.stats.ks_2samp.

    Mantém um buffer deslizante de tamanho window_size + stat_size.
    A cada stat_size observacoes, compara as ultimas stat_size amostras
    (janela de teste) contra as primeiras window_size (janela de referencia)
    via KS de duas amostras. Se p-value < alpha: drift detectado e buffer
    reiniciado com a janela de teste como nova referencia.

    A checagem a cada stat_size passos (em vez de a cada passo) garante
    complexidade O(n * w_log_w / stat_size) em vez de O(n * w_log_w).
    """

    def __init__(self, alpha: float = 0.001,
                 window_size: int = 500, stat_size: int = 100):
        self.alpha       = alpha
        self.window_size = window_size
        self.stat_size   = stat_size
        self.buffer: list[float] = []
        self.n           = 0
        self.n_since_check = 0
        self.drift_detected = False

    def update(self, x: float) -> None:
        self.drift_detected = False
        self.n += 1
        self.buffer.append(float(x))
        self.n_since_check += 1

        # Trim ao tamanho maximo
        max_buf = self.window_size + self.stat_size
        if len(self.buffer) > max_buf:
            self.buffer = self.buffer[-max_buf:]

        # So testa a cada stat_size observacoes
        if self.n_since_check < self.stat_size:
            return
        if len(self.buffer) < self.window_size + self.stat_size:
            return

        self.n_since_check = 0
        ref  = np.asarray(self.buffer[:self.window_size])
        test = np.asarray(self.buffer[-self.stat_size:])
        _, p_val = ks_2samp(ref, test)

        if p_val < self.alpha:
            self.drift_detected = True
            self.buffer = list(test)    # reinicia com janela de teste

# ── pré-processamento e agregacao ─────────────────────────────────────────────
# smooth_robz vem de common.detection (re-exportado aqui para os consumidores
# que fazem `from phase2_grid_search import smooth_robz`)


def aggregate_stream(stream: pd.DataFrame, N: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agrega stream em janelas de N trajetorias.

    Retorna:
      ade_mean   : (n_windows,) array de medias ADE por janela
      year_win   : (n_windows,) array de anos majoritarios por janela
      global_mid : (n_windows,) indice global do centro de cada janela
    """
    n_windows = len(stream) // N
    if n_windows == 0:
        return np.array([]), np.array([]), np.array([])

    ade    = stream["ade_traj"].astype(float).values[:n_windows * N]
    years  = stream["year"].fillna(0).astype(int).values[:n_windows * N]
    glb    = stream["global_idx"].values[:n_windows * N]

    ade_r   = ade.reshape(n_windows, N)
    years_r = years.reshape(n_windows, N)
    glb_r   = glb.reshape(n_windows, N)

    ade_mean   = np.nanmean(ade_r, axis=1)
    # ano maioritario por janela
    year_win   = np.array([
        int(pd.Series(years_r[i]).mode().iloc[0]) for i in range(n_windows)
    ])
    global_mid = glb_r[:, N // 2]

    return ade_mean, year_win, global_mid

# ── metricas ──────────────────────────────────────────────────────────────────
# alarms_per_year / compute_snr vem de common.detection (re-exportados aqui)

# ── plots de stream (top configs) ─────────────────────────────────────────────

def make_stream_plot(stream: pd.DataFrame, signal: np.ndarray,
                     alarms: list[int], detector_name: str,
                     label: str, model_name: str,
                     frente: str, is_window: bool = False) -> plt.Figure:
    x    = stream["global_idx"].values if not is_window else np.arange(len(signal))
    yr_v = stream["year"].fillna(0).astype(int).values if not is_window else None

    fig = plt.figure(figsize=(14, 5.5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[5, 1.3], hspace=0.06,
                           left=0.07, right=0.985, top=0.91, bottom=0.09)
    ax  = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    # sombras por ano (so para streams originais)
    if yr_v is not None:
        for yr in sorted(set(int(y) for y in yr_v if y != 0)):
            mask = yr_v == yr
            x0, x1 = int(x[mask].min()), int(x[mask].max())
            c = YEAR_COLORS.get(yr, "#888")
            ax.axvspan(x0, x1+1, color=c, alpha=0.07, zorder=0)
            axr.axvspan(x0, x1+1, color=c, alpha=0.07, zorder=0)
            ypos = float(np.nanpercentile(signal[np.isfinite(signal)], 99)) * 0.97
            ax.text((x0+x1)/2, ypos, str(yr), color=c, fontsize=9,
                    ha="center", va="top", fontweight="bold", alpha=0.9)

    # sinal
    ax.plot(x, signal, color="#1c1c8c", lw=1.4, zorder=2, label=label)
    ymax = float(np.nanpercentile(signal[np.isfinite(signal)], 99.5)) * 1.08
    ax.set_ylim(bottom=min(signal[np.isfinite(signal)].min() * 0.95, 0),
                top=max(ymax, 1.0))
    ax.set_ylabel("ADE (mm)" if "robz" not in label.lower() else "ADE (z-score)")
    ax.set_title(f"{model_name} 30->15 | Frente {frente} | {detector_name} | {label}", pad=7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # raster de alarmes
    det_color = {"ADWIN": "#2A6FDB", "PH": "#D1495B", "KSWIN": "#27ae60"}.get(
        detector_name, "#333")
    if alarms:
        axr.vlines(x[alarms] if not is_window else np.array(alarms),
                   0.1, 0.9, color=det_color, lw=0.8, alpha=0.8)
    axr.set_ylim(0, 1)
    axr.set_yticks([0.5])
    axr.set_yticklabels([f"{detector_name} ({len(alarms)})"], fontsize=9)
    axr.tick_params(axis="y", length=0)
    axr.set_xlabel("Janela" if is_window else "Trajetoria (ordem global)")
    plt.setp(ax.get_xticklabels(), visible=False)
    return fig


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)

# ── heatmaps ──────────────────────────────────────────────────────────────────

def make_heatmap(df_sub: pd.DataFrame, row_param: str, col_param: str,
                 metric: str, title: str, fmt: str = ".3f") -> plt.Figure:
    """Heatmap de metric (e.g. FAR_2019_per1k) em funcao de dois parametros."""
    pivot = df_sub.groupby([row_param, col_param])[metric].min().unstack(col_param)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=float(np.nanpercentile(pivot.values, 95)))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=9)
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(title, pad=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8, color="white" if v > 1.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label=metric)
    plt.tight_layout()
    return fig


def make_scatter(df_sub: pd.DataFrame, color_param: str,
                 title: str) -> plt.Figure:
    """Scatter FAR_2019_per1k (x) vs n_alarms_post (y), colorido por color_param."""
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = sorted(df_sub[color_param].unique())
    cmap = plt.get_cmap("viridis", len(vals))
    for i, v in enumerate(vals):
        sub = df_sub[df_sub[color_param] == v]
        ax.scatter(sub["FAR_2019_per1k"], sub["n_alarms_post"],
                   label=str(v), color=cmap(i), s=40, alpha=0.8, edgecolors="none")
    ax.set_xlabel("FAR_2019 / 1k trajs (< 0.5 = bom)")
    ax.set_ylabel("n_alarms_post (2021-2025)")
    ax.set_title(title, pad=8)
    ax.axvline(0.5, color="red", lw=1, ls="--", alpha=0.7, label="limiar 0.5/1k")
    ax.legend(title=color_param, fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig

# ── runner generico ───────────────────────────────────────────────────────────

def run_detector_metrics(detector, signal: np.ndarray,
                         year_vals: np.ndarray,
                         n_2019_total: int,
                         n_post_total: int) -> dict:
    alarms = run_detector(detector, signal)
    by_yr  = alarms_per_year(alarms, year_vals, YEARS_ALL)
    n_2019_alarms = by_yr["n_2019"]
    n_post        = sum(by_yr[f"n_{y}"] for y in [2021, 2022, 2023, 2024, 2025])
    far19  = 1000.0 * n_2019_alarms / max(1, n_2019_total)
    snr    = compute_snr(n_post, n_2019_alarms, n_2019_total, n_post_total)
    return dict(
        alarms=alarms,
        n_alarms_total=len(alarms),
        FAR_2019_per1k=round(far19, 4),
        n_alarms_post=n_post,
        SNR_op=round(min(snr, PERFECT_SNR), 2),
        **by_yr,
    )

# ── Frente A ──────────────────────────────────────────────────────────────────

def run_frente_a(stream: pd.DataFrame, raw_vals: np.ndarray,
                 year_vals: np.ndarray,
                 n_2019_total: int, n_post_total: int,
                 model: str) -> list[dict]:
    print(f"  [Frente A] pre-processando robz_w200 ...", end="", flush=True)
    smoothed = smooth_robz(raw_vals, 200)
    b2019    = smoothed[year_vals == 2019]
    b2019    = b2019[np.isfinite(b2019)]
    print(f" ok (n_2019_smooth={len(b2019)})")

    rows: list[dict] = []
    total = len(GRID_A["delta"]) * len(GRID_A["min_window"]) * \
            len(GRID_A["cooldown"]) * len(GRID_A["max_window"])
    done  = 0

    for delta, min_w, cooldown, max_w in itertools.product(
            GRID_A["delta"], GRID_A["min_window"],
            GRID_A["cooldown"], GRID_A["max_window"]):
        det = ADWIN_CLASS(delta=delta, min_window=min_w,
                          cooldown=cooldown, max_window=max_w)
        m   = run_detector_metrics(det, smoothed, year_vals,
                                   n_2019_total, n_post_total)
        done += 1
        if done % 24 == 0:
            print(f"    [{model}] Frente A: {done}/{total} configs ...", flush=True)

        name = f"A_d{delta:.0e}_mw{min_w}_cd{cooldown}_xw{max_w}"
        rows.append(dict(
            frente="A", model=model, detector="ADWIN",
            exp=name,
            delta=delta, min_window=min_w, cooldown=cooldown, max_window=max_w,
            pre_proc="robz_w200",
            n_2019_total=n_2019_total, n_post_total=n_post_total,
            **{k: v for k, v in m.items() if k != "alarms"},
        ))

    return rows


# ── Frente B ──────────────────────────────────────────────────────────────────

def run_frente_b(stream: pd.DataFrame, raw_vals: np.ndarray,
                 year_vals: np.ndarray,
                 n_2019_total: int, n_post_total: int,
                 model: str) -> list[dict]:
    rows: list[dict] = []
    total = (len(GRID_B["N_window"]) * len(GRID_B["K_thr"])
             * len(GRID_B["K_delta"]))
    done  = 0

    for N in GRID_B["N_window"]:
        ade_agg, year_agg, _ = aggregate_stream(stream, N)
        if len(ade_agg) == 0:
            continue

        # baseline 2019 do sinal agregado
        b2019_agg = ade_agg[year_agg == 2019]
        b2019_agg = b2019_agg[np.isfinite(b2019_agg)]
        if len(b2019_agg) < 5:
            b2019_agg = ade_agg[np.isfinite(ade_agg)]

        n_2019_win = int((year_agg == 2019).sum())
        n_post_win = int((year_agg >= 2021).sum())
        # FAR em unidades de janelas (e despois convertemos para trajetorias)
        # n_2019_total e em trajetorias; dividimos por N para obter janelas
        n_2019_win_equiv = max(1, n_2019_total // N)
        n_post_win_equiv = max(1, n_post_total // N)

        for K_thr, K_delta in itertools.product(GRID_B["K_thr"], GRID_B["K_delta"]):
            ph = PageHinkley.from_baseline(
                b2019_agg, K_thr=K_thr, K_delta=K_delta,
                min_instances=max(3, n_2019_win // 5),
                cooldown=max(2, n_2019_win // 20),
            )
            # rodar sobre stream agregado
            alarms_win = run_detector(ph, ade_agg)
            by_yr      = alarms_per_year(alarms_win, year_agg, YEARS_ALL)
            n_2019_a   = by_yr["n_2019"]
            n_post_a   = sum(by_yr[f"n_{y}"] for y in [2021,2022,2023,2024,2025])

            # FAR em unidades de trajetorias (para comparacao com Fase 1 e Frente A)
            far19_traj = 1000.0 * n_2019_a / max(1, n_2019_win_equiv)
            snr = compute_snr(n_post_a, n_2019_a, n_2019_win_equiv, n_post_win_equiv)

            done += 1
            name = f"B_N{N}_K{K_thr}_d{K_delta}"
            rows.append(dict(
                frente="B", model=model, detector="PH",
                exp=name,
                N_window=N, K_thr=K_thr, K_delta=K_delta,
                pre_proc="window_agg",
                n_2019_total=n_2019_win_equiv, n_post_total=n_post_win_equiv,
                n_alarms_total=len(alarms_win),
                FAR_2019_per1k=round(far19_traj, 4),
                n_alarms_post=n_post_a,
                SNR_op=round(min(snr, PERFECT_SNR), 2),
                **by_yr,
            ))

        if done % 15 == 0:
            print(f"    [{model}] Frente B: {done}/{total} configs (N={N}) ...",
                  flush=True)

    return rows


# ── Frente C ──────────────────────────────────────────────────────────────────

def run_frente_c(stream: pd.DataFrame, raw_vals: np.ndarray,
                 year_vals: np.ndarray,
                 n_2019_total: int, n_post_total: int,
                 model: str) -> list[dict]:
    rows: list[dict] = []
    total = (len(GRID_C["alpha"]) * len(GRID_C["window_size"])
             * len(GRID_C["stat_size"]))
    done  = 0

    for alpha, win_s, stat_s in itertools.product(
            GRID_C["alpha"], GRID_C["window_size"], GRID_C["stat_size"]):
        if stat_s >= win_s:
            continue  # invalido: janela de teste >= janela de referencia

        det = KSWINDetector(alpha=alpha, window_size=win_s, stat_size=stat_s)
        m   = run_detector_metrics(det, raw_vals, year_vals,
                                   n_2019_total, n_post_total)
        done += 1
        name = f"C_a{alpha:.0e}_ws{win_s}_ss{stat_s}"
        rows.append(dict(
            frente="C", model=model, detector="KSWIN",
            exp=name,
            alpha=alpha, window_size=win_s, stat_size=stat_s,
            pre_proc="none",
            n_2019_total=n_2019_total, n_post_total=n_post_total,
            **{k: v for k, v in m.items() if k != "alarms"},
        ))

    print(f"    [{model}] Frente C: {done}/{total} configs", flush=True)
    return rows

# ── geracao de imagens ─────────────────────────────────────────────────────────

def generate_frente_a_images(df_a_model: pd.DataFrame,
                              stream: pd.DataFrame,
                              raw_vals: np.ndarray,
                              year_vals: np.ndarray,
                              model: str, img_dir: Path) -> None:
    smoothed = smooth_robz(raw_vals, 200)
    # Top-5 por SNR_op com FAR <= 2.0
    top = df_a_model[df_a_model["FAR_2019_per1k"] <= 2.0].nlargest(5, "SNR_op")
    for _, r in top.iterrows():
        det = ADWIN_CLASS(delta=r["delta"], min_window=int(r["min_window"]),
                          cooldown=int(r["cooldown"]), max_window=int(r["max_window"]))
        alarms = run_detector(det, smoothed)
        label  = (f"robz+ADWIN d={r['delta']:.0e} mw={int(r['min_window'])}"
                  f" cd={int(r['cooldown'])} xw={int(r['max_window'])}")
        fig = make_stream_plot(stream, smoothed, alarms, "ADWIN", label, model, "A")
        save_fig(fig, img_dir / f"{r['exp']}_{model.lower()}_stream.png")

    # Heatmaps: delta x min_window (min FAR sobre cooldown e max_window)
    for metric, fmt in [("FAR_2019_per1k", ".2f"), ("SNR_op", ".1f")]:
        try:
            fig = make_heatmap(
                df_a_model, "delta", "min_window", metric,
                f"Frente A — {model} | {metric}\n(min sobre cooldown x max_window)"
            )
            save_fig(fig, img_dir / f"heatmap_{metric}_{model.lower()}.png")
        except Exception as e:
            print(f"  [warn] heatmap Frente A {metric}: {e}")
            plt.close("all")

    # Scatter
    try:
        fig = make_scatter(df_a_model, "delta",
                           f"Frente A — {model} | ADWIN+robz_w200\nFAR_2019 vs n_post")
        save_fig(fig, img_dir / f"scatter_{model.lower()}.png")
    except Exception as e:
        print(f"  [warn] scatter Frente A: {e}")
        plt.close("all")


def generate_frente_b_images(df_b_model: pd.DataFrame,
                              stream: pd.DataFrame,
                              raw_vals: np.ndarray,
                              model: str, img_dir: Path) -> None:
    # Top-5 por SNR_op
    top = df_b_model.nlargest(5, "SNR_op")
    for _, r in top.iterrows():
        N = int(r["N_window"])
        ade_agg, year_agg, _ = aggregate_stream(stream, N)
        if len(ade_agg) == 0:
            continue
        b2019 = ade_agg[year_agg == 2019]
        b2019 = b2019[np.isfinite(b2019)]
        if len(b2019) < 5:
            b2019 = ade_agg[np.isfinite(ade_agg)]
        ph = PageHinkley.from_baseline(
            b2019, K_thr=r["K_thr"], K_delta=r["K_delta"],
            min_instances=max(3, len(b2019) // 5),
            cooldown=max(2, len(b2019) // 20),
        )
        alarms = run_detector(ph, ade_agg)
        label  = f"PH N={N} K={r['K_thr']} Kd={r['K_delta']}"
        # Para plot de stream agregado: usamos ade_agg como sinal e year_agg
        stream_fake = pd.DataFrame({
            "global_idx": np.arange(len(ade_agg)),
            "ade_traj":   ade_agg,
            "year":       year_agg,
        })
        fig = make_stream_plot(stream_fake, ade_agg, alarms, "PH", label,
                               model, "B", is_window=False)
        save_fig(fig, img_dir / f"{r['exp']}_{model.lower()}_stream.png")

    # Heatmap N x K_thr
    for metric, fmt in [("FAR_2019_per1k", ".2f"), ("SNR_op", ".1f")]:
        try:
            fig = make_heatmap(
                df_b_model, "N_window", "K_thr", metric,
                f"Frente B — {model} | {metric}\n(min sobre K_delta)"
            )
            save_fig(fig, img_dir / f"heatmap_{metric}_{model.lower()}.png")
        except Exception as e:
            print(f"  [warn] heatmap Frente B {metric}: {e}")
            plt.close("all")

    try:
        fig = make_scatter(df_b_model, "N_window",
                           f"Frente B — {model} | PH+agregacao\nFAR_2019 vs n_post")
        save_fig(fig, img_dir / f"scatter_{model.lower()}.png")
    except Exception as e:
        print(f"  [warn] scatter Frente B: {e}")
        plt.close("all")


def generate_frente_c_images(df_c_model: pd.DataFrame,
                              stream: pd.DataFrame,
                              raw_vals: np.ndarray,
                              year_vals: np.ndarray,
                              model: str, img_dir: Path) -> None:
    top = df_c_model[df_c_model["FAR_2019_per1k"] <= 3.0].nlargest(5, "SNR_op")
    for _, r in top.iterrows():
        det = KSWINDetector(alpha=r["alpha"], window_size=int(r["window_size"]),
                            stat_size=int(r["stat_size"]))
        alarms = run_detector(det, raw_vals)
        label  = f"KSWIN a={r['alpha']:.0e} w={int(r['window_size'])} s={int(r['stat_size'])}"
        fig = make_stream_plot(stream, raw_vals, alarms, "KSWIN", label, model, "C")
        save_fig(fig, img_dir / f"{r['exp']}_{model.lower()}_stream.png")

    for metric in ["FAR_2019_per1k", "SNR_op"]:
        try:
            fig = make_heatmap(
                df_c_model, "alpha", "window_size", metric,
                f"Frente C — {model} | {metric}\n(min sobre stat_size)"
            )
            save_fig(fig, img_dir / f"heatmap_{metric}_{model.lower()}.png")
        except Exception as e:
            print(f"  [warn] heatmap Frente C {metric}: {e}")
            plt.close("all")

    try:
        fig = make_scatter(df_c_model, "alpha",
                           f"Frente C — {model} | KSWIN\nFAR_2019 vs n_post")
        save_fig(fig, img_dir / f"scatter_{model.lower()}.png")
    except Exception as e:
        print(f"  [warn] scatter Frente C: {e}")
        plt.close("all")

# ── relatorio MD ──────────────────────────────────────────────────────────────
# _df_to_md vem de common.plotting (df_to_md)

def write_report(df: pd.DataFrame, out_dir: Path) -> None:
    L: list[str] = []

    # Contexto dos achados da Fase 1
    L += [
        "# Fase 2 — Grid Search de Hiperparametros",
        "",
        "## Contexto (Fase 1)",
        "- **SMA/Mediana/EWMA** pioram ADWIN (autocorrelacao viola suposicao i.i.d.)",
        "- **robz_w200** e o unico pre-proc que ajuda ADWIN: FAR_2019 1.23 -> 0.60/1k",
        "- **Holt a=0.05 b=0.01** foi o melhor para PH: FAR_2019 4.98 -> 2.52/1k",
        "- Nenhum experimento da Fase 1 atingiu FAR_2019 = 0",
        "",
        "## Metricas",
        "| Coluna | Descricao | Direcao |",
        "|--------|-----------|---------|",
        "| `FAR_2019_per1k` | Alarmes / 1.000 trajs (ou janelas) em 2019 | **abaixo melhor** |",
        "| `n_alarms_post` | Total alarmes 2021-2025 | **acima melhor** |",
        "| `SNR_op` | (n_post x n_2019_total) / (n_2019_alarms x n_post_total) | **acima melhor** |",
        "",
        "## Grids testados",
        "",
        "**Frente A (ADWIN + robz_w200):** 96 configs — delta x min_window x cooldown x max_window",
        "**Frente B (PH + agregacao por janela):** 60 configs — N_window x K_thr x K_delta",
        "**Frente C (KSWIN, puro scipy):** 18 configs validas — alpha x window_size x stat_size",
        "",
    ]

    DISPLAY_BASE = ["exp", "FAR_2019_per1k",
                    "n_2019", "n_2021", "n_2022", "n_2023", "n_2024", "n_2025",
                    "n_alarms_post", "SNR_op", "n_alarms_total"]

    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        L += ["---", f"## Modelo: {model}", ""]

        # ── Frente A ────────────────────────────────────────────────────────
        da = mdf[mdf["frente"] == "A"].copy()
        if not da.empty:
            L += ["### Frente A — ADWIN + robz_w200", ""]
            top20 = da.sort_values("FAR_2019_per1k").head(20)
            extra = ["delta", "min_window", "cooldown", "max_window"]
            disp  = [c for c in extra + DISPLAY_BASE if c in da.columns]
            L.append(_df_to_md(top20[disp]))
            L.append("")

            best_zero = da[da["FAR_2019_per1k"] == 0].nlargest(3, "SNR_op")
            best_low  = da[da["FAR_2019_per1k"] <= 0.5].nlargest(3, "SNR_op")
            best_mid  = da[da["FAR_2019_per1k"] <= 2.0].nlargest(3, "SNR_op")

            if not best_zero.empty:
                L += ["**FAR_2019 = 0 (perfeito):**", ""]
                for _, r in best_zero.iterrows():
                    L.append(f"- `{r['exp']}`: SNR_op={r['SNR_op']:.2f}  "
                             f"n_post={r['n_alarms_post']}  n_2021={r.get('n_2021',0)}")
                L.append("")
            if not best_low.empty:
                L += ["**FAR_2019 <= 0.5/1k:**", ""]
                for _, r in best_low.iterrows():
                    L.append(f"- `{r['exp']}`: FAR={r['FAR_2019_per1k']:.3f}  "
                             f"SNR_op={r['SNR_op']:.2f}  n_post={r['n_alarms_post']}")
                L.append("")
            if not best_mid.empty:
                L += ["**FAR_2019 <= 2.0/1k (top SNR_op):**", ""]
                for _, r in best_mid.iterrows():
                    L.append(f"- `{r['exp']}`: FAR={r['FAR_2019_per1k']:.3f}  "
                             f"SNR_op={r['SNR_op']:.2f}  delta={r['delta']:.0e}  "
                             f"min_w={int(r['min_window'])}")
                L.append("")
            L += [
                f"**Minimo FAR_2019:** {da['FAR_2019_per1k'].min():.4f}/1k",
                f"**Maximo SNR_op (todos):** {da['SNR_op'].max():.2f}",
                "",
            ]

        # ── Frente B ────────────────────────────────────────────────────────
        db = mdf[mdf["frente"] == "B"].copy()
        if not db.empty:
            L += ["### Frente B — PH com agregacao por janela (N trajetorias)", ""]
            top20b = db.sort_values("FAR_2019_per1k").head(20)
            extra = ["N_window", "K_thr", "K_delta"]
            disp  = [c for c in extra + DISPLAY_BASE if c in db.columns]
            L.append(_df_to_md(top20b[disp]))
            L.append("")

            best_zero_b = db[db["FAR_2019_per1k"] == 0].nlargest(5, "SNR_op")
            best_low_b  = db[db["FAR_2019_per1k"] <= 1.0].nlargest(5, "SNR_op")
            if not best_zero_b.empty:
                L += ["**FAR_2019 = 0:**", ""]
                for _, r in best_zero_b.iterrows():
                    L.append(f"- `{r['exp']}`: N={int(r['N_window'])} K_thr={r['K_thr']}  "
                             f"SNR_op={r['SNR_op']:.2f}  n_post={r['n_alarms_post']}")
                L.append("")
            if not best_low_b.empty:
                L += ["**FAR_2019 <= 1.0/1k (janelas):**", ""]
                for _, r in best_low_b.iterrows():
                    L.append(f"- `{r['exp']}`: FAR={r['FAR_2019_per1k']:.3f}  "
                             f"SNR_op={r['SNR_op']:.2f}  N={int(r['N_window'])} K={r['K_thr']}")
                L.append("")
            L += [
                f"**Minimo FAR_2019:** {db['FAR_2019_per1k'].min():.4f}/1k",
                "",
            ]

        # ── Frente C ────────────────────────────────────────────────────────
        dc = mdf[mdf["frente"] == "C"].copy()
        if not dc.empty:
            L += ["### Frente C — KSWIN (Kolmogorov-Smirnov Windowed)", ""]
            top20c = dc.sort_values("FAR_2019_per1k").head(20)
            extra = ["alpha", "window_size", "stat_size"]
            disp  = [c for c in extra + DISPLAY_BASE if c in dc.columns]
            L.append(_df_to_md(top20c[disp]))
            L.append("")

            best_c = dc[dc["FAR_2019_per1k"] <= 2.0].nlargest(3, "SNR_op")
            if not best_c.empty:
                L += ["**Top KSWIN FAR_2019 <= 2.0/1k:**", ""]
                for _, r in best_c.iterrows():
                    L.append(f"- `{r['exp']}`: FAR={r['FAR_2019_per1k']:.3f}  "
                             f"alpha={r['alpha']:.0e}  win={int(r['window_size'])}  "
                             f"SNR_op={r['SNR_op']:.2f}")
                L.append("")
            L += [f"**Minimo FAR_2019:** {dc['FAR_2019_per1k'].min():.4f}/1k", ""]

    # ── Comparacao cross-frente (Seq2Seq) ──────────────────────────────────────
    seq = df[df["model"] == "Seq2Seq"].copy()
    if not seq.empty:
        L += [
            "---",
            "## Comparacao cross-frente — Seq2Seq",
            "",
            "Melhor config por frente (menor FAR_2019):",
            "",
        ]
        comp_rows = []
        for frente in ["A", "B", "C"]:
            sub = seq[seq["frente"] == frente]
            if sub.empty:
                continue
            best = sub.loc[sub["FAR_2019_per1k"].idxmin()]
            comp_rows.append({
                "Frente": frente,
                "Detector": best["detector"],
                "exp": best["exp"],
                "FAR_2019_per1k": best["FAR_2019_per1k"],
                "n_alarms_post":  best["n_alarms_post"],
                "SNR_op":         best["SNR_op"],
            })
        if comp_rows:
            L.append(_df_to_md(pd.DataFrame(comp_rows)))
            L.append("")

    # ── Recomendacao Fase 3 ─────────────────────────────────────────────────
    L += [
        "---",
        "## Recomendacao para Fase 3 (Ensemble)",
        "",
        "Selecionar as 2 melhores configs por frente (menor FAR_2019, maior SNR_op).",
        "Combinar com ensemble AND-temporal (janela de concordancia = 200 trajs).",
        "",
        "### Candidatos para Fase 3 (preencher apos analise visual)",
        "",
        "| Frente | Detector | Config | FAR_2019 | SNR_op |",
        "|--------|----------|--------|----------|--------|",
        "| A | ADWIN | `<exp>` | `<val>` | `<val>` |",
        "| B | PH    | `<exp>` | `<val>` | `<val>` |",
        "| C | KSWIN | `<exp>` | `<val>` | `<val>` |",
        "",
        "### Imagens geradas",
        "- `images/frente_A/` : heatmap FAR_2019, heatmap SNR_op, scatter, top-5 stream plots",
        "- `images/frente_B/` : idem para PH com agregacao",
        "- `images/frente_C/` : idem para KSWIN",
        "",
    ]

    (out_dir / "PHASE2_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  [ok] {out_dir / 'PHASE2_REPORT.md'}")

# ── main ──────────────────────────────────────────────────────────────────────

def run_phase2() -> None:
    for d in [OUT_ROOT, IMG_A, IMG_B, IMG_C]:
        d.mkdir(parents=True, exist_ok=True)

    print("[phase2] Carregando dados ...")
    df_raw = load_errors()
    print(f"  shape={df_raw.shape}")

    all_rows: list[dict] = []

    for model in ("Seq2Seq", "Kalman"):
        stream = build_stream(df_raw, model=model)
        if stream.empty:
            print(f"[warn] sem dados para {model}")
            continue

        raw_vals  = stream["ade_traj"].astype(float).values
        year_vals = stream["year"].fillna(0).astype(int).values
        n = len(stream)
        n_2019_total = int((year_vals == 2019).sum())
        n_post_total = int((year_vals >= 2021).sum())
        print(f"\n=== {model} ({n:,} trajetorias | 2019={n_2019_total} | post={n_post_total}) ===")

        # Frente A
        print(f"  -- Frente A (ADWIN + robz_w200, 96 configs) --")
        rows_a = run_frente_a(stream, raw_vals, year_vals,
                              n_2019_total, n_post_total, model)
        all_rows.extend(rows_a)
        fa_min = min(r["FAR_2019_per1k"] for r in rows_a)
        print(f"  Frente A concluida: {len(rows_a)} configs | FAR_2019 min={fa_min:.4f}/1k")

        # Frente B
        print(f"  -- Frente B (PH + agregacao, 60 configs) --")
        rows_b = run_frente_b(stream, raw_vals, year_vals,
                              n_2019_total, n_post_total, model)
        all_rows.extend(rows_b)
        if rows_b:
            fb_min = min(r["FAR_2019_per1k"] for r in rows_b)
            print(f"  Frente B concluida: {len(rows_b)} configs | FAR_2019 min={fb_min:.4f}/1k")

        # Frente C
        print(f"  -- Frente C (KSWIN, 18 configs) --")
        rows_c = run_frente_c(stream, raw_vals, year_vals,
                              n_2019_total, n_post_total, model)
        all_rows.extend(rows_c)
        if rows_c:
            fc_min = min(r["FAR_2019_per1k"] for r in rows_c)
            print(f"  Frente C concluida: {len(rows_c)} configs | FAR_2019 min={fc_min:.4f}/1k")

        # Imagens (so para Seq2Seq nas imagens de stream; heatmaps para ambos)
        print(f"  -- Gerando imagens ({model}) --")
        df_this = pd.DataFrame(all_rows)
        df_this = df_this[df_this["model"] == model]
        if not df_this.empty:
            da = df_this[df_this["frente"] == "A"]
            db = df_this[df_this["frente"] == "B"]
            dc = df_this[df_this["frente"] == "C"]
            if not da.empty:
                generate_frente_a_images(da, stream, raw_vals, year_vals, model, IMG_A)
            if not db.empty:
                generate_frente_b_images(db, stream, raw_vals, model, IMG_B)
            if not dc.empty:
                generate_frente_c_images(dc, stream, raw_vals, year_vals, model, IMG_C)

    # CSV
    results = pd.DataFrame(all_rows)
    csv_path = OUT_ROOT / "phase2_results.csv"
    results.to_csv(str(csv_path), index=False, float_format="%.6f")
    print(f"\n[ok] CSV -> {csv_path}")

    # Relatorio
    write_report(results, OUT_ROOT)
    print(f"[ok] Relatorio -> {OUT_ROOT / 'PHASE2_REPORT.md'}")
    print(f"[ok] Imagens   -> {OUT_ROOT / 'images/'}")


if __name__ == "__main__":
    run_phase2()
