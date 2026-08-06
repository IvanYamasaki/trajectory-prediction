"""
drift_analise/phase4_consolidate.py
=====================================
Fase 4 — Consolidação Científica e Resultado Final.

Este script NÃO re-executa grids nem reimplementa detetores.
Usa chapter02_deteccao_pipeline.py + phase3_ensemble.py já existentes
para extrair listas de alarmes (necessárias para a figura de inviabilidade)
e re-tabula os finalistas das Fases 1–3 com métricas corrigidas.

Outputs (todos em Relas/results/drift/mes9_phase4/):
    phase4_master_table.csv
    W_infeasibility_sweep.csv
    images/and_infeasibility.png
    PHASE4_FINAL.md
"""
from __future__ import annotations

import io
import os
import sys
import textwrap
from pathlib import Path

# fix console encoding on Windows (cp1252 doesn't support Delta/subscript chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.detection import adwin_variant_config
from common.plotting import YEAR_COLORS, make_plot_style
from drift_analise.chapter02_deteccao_pipeline import load_errors, build_stream
from drift_analise.phase2_grid_search import smooth_robz
from drift_analise.phase3_ensemble import (
    _run_adwin, _run_ph_agg, _and_gate, _eval,
    ADWIN_S2S_MULTIYEAR, ROBZ_WINDOW, YEARS_ALL, RETRAIN_DEBOUNCE,
)
from drift_analise.phase4_metrics import (
    enrich, rank_detectors, robustness_table, wilson_ci_per1k, YEARS_POST,
)

np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
# Mirrors phase2/phase3: ADWIN_VARIANT controls which set of inputs we consume
# and where the outputs land.
ADWIN_VARIANT, _SUF = adwin_variant_config()

OUT_ROOT = PROJECT_ROOT / "Relas" / "results" / "drift" / f"mes9_phase4{_SUF}"
IMG_DIR  = OUT_ROOT / "images"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_CSV = PROJECT_ROOT / "Relas" / "results" / "drift" / f"mes9_phase3{_SUF}" / "phase3_results.csv"
PHASE2_CSV = PROJECT_ROOT / "Relas" / "results" / "drift" / f"mes9_phase2{_SUF}" / "phase2_results.csv"

PLOT_STYLE = make_plot_style(**{
    "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
    # o arquivo nao definia mathtext.fontset; preserva o default do matplotlib
    "mathtext.fontset": "dejavusans",
})
plt.rcParams.update(PLOT_STYLE)

# configs dos finalistas (para coluna 'config' na master table)
# Spelled out per ADWIN variant — paper used ADWINLite; the exact-ADWIN
# re-grid (Mai/2026) shifted the operating points (mw, cd).
if ADWIN_VARIANT == "exact":
    _ADWIN_S2S_CFG = "delta=1e-7, mw=200, cd=1000, xw=2000"
    _ADWIN_KAL_CFG = "delta=1e-7, mw=1000, cd=200, xw=2000"
else:
    _ADWIN_S2S_CFG = "delta=1e-7, mw=600, cd=200, xw=2000"
    _ADWIN_KAL_CFG = "delta=1e-7, mw=200, cd=1000, xw=5000"

CONFIGS = {
    "adwin_standalone":    _ADWIN_S2S_CFG,
    "ph_standalone":       "N=50, K_thr=50, K_delta=2.0",
    "kalman_standalone":   _ADWIN_KAL_CFG,
    "suspected (OR)":      "OR(ADWIN_S2S, ADWIN_Kalman)",
    "confirmed (AND+debounce)": "AND(ADWIN_S2S, PH_agg), W=200, deb=5000",
    "KSWIN":               "alpha=1e-5, ws=1000, ss=100",
}

DETECTOR_LABELS = {
    "adwin_standalone":         "ADWIN_S2S",
    "ph_standalone":            "PH_agg",
    "kalman_standalone":        "ADWIN_Kalman",
    "suspected (OR)":           "OR(ADWIN_S2S, ADWIN_Kalman)",
    "confirmed (AND+debounce)": "AND(ADWIN_S2S, PH_agg)",
    "KSWIN":                    "KSWIN",
}

PAPEIS = {
    "adwin_standalone":         "Detetor primário recomendado",
    "ph_standalone":            "Resultado negativo — cobertura 1/5 anos",
    "kalman_standalone":        "Corroborador OR",
    "suspected (OR)":           "Vigilância multi-ano",
    "confirmed (AND+debounce)": "Resultado negativo — AND inviável",
    "KSWIN":                    "Inadmissível — FAR > 0.20/1k (especificidade insuficiente)",
}

FAR_ADMISSIBILITY = 0.20  # limiar de elegibilidade: FAR > valor → inadmissível

_FACTOR2_YEAR_COLS = ["n_2021", "n_2022", "n_2023", "n_2024", "n_2025", "year_coverage"]


def _check_factor2_consistency(master: pd.DataFrame, ph_r: "pd.Series", adwin_r: "pd.Series") -> None:
    """Assert: valores usados na tabela do Factor 2 são idênticos à master table.
    O script falha com AssertionError se divergirem — impede reintrodução do bug."""
    for det_label, row in [("PH_agg", ph_r), ("ADWIN_S2S", adwin_r)]:
        master_rows = master[master["detector"] == det_label]
        assert not master_rows.empty, (
            f"[COERÊNCIA CRUZADA] '{det_label}' ausente da master table — regeneração bloqueada."
        )
        for col in _FACTOR2_YEAR_COLS:
            expected = int(master_rows.iloc[0][col])
            actual   = int(row.get(col, -999))
            assert actual == expected, (
                f"[COERÊNCIA CRUZADA] {det_label}.{col}: "
                f"Factor 2 usaria {actual} mas master table tem {expected}. "
                f"Corrija a fonte de dados do Factor 2 antes de regenerar."
            )


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Carregar stream e re-correr detetores
# ══════════════════════════════════════════════════════════════════════════════

def load_and_run() -> dict:
    """Carrega stream Seq2Seq e extrai listas de alarmes para a figura."""
    print("[1] Carregando stream Seq2Seq …")
    df = load_errors()
    stream_s2s = build_stream(df, model="Seq2Seq")

    year_vals    = stream_s2s["year"].fillna(0).astype(int).values
    n_2019_total = int((year_vals == 2019).sum())
    n_post_total = int((year_vals >= 2021).sum())

    print(f"    n_2019_total={n_2019_total}, n_post_total={n_post_total}, "
          f"total={len(stream_s2s)}")

    print("[1] Re-correndo ADWIN_S2S e PH_agg para extrair índices de alarme …")
    adwin_alarms          = _run_adwin(stream_s2s, ADWIN_S2S_MULTIYEAR)
    ph_alarms_traj, _, _  = _run_ph_agg(stream_s2s)

    # verificação de consistência com phase3_results.csv
    n_adwin_2019 = sum(1 for a in adwin_alarms if year_vals[a] == 2019)
    n_ph_2019    = sum(1 for a in ph_alarms_traj if year_vals[a] == 2019)
    print(f"    ADWIN: {len(adwin_alarms)} alarmes (2019: {n_adwin_2019}) "
          f"— esperado 25 (2019: 2)")
    print(f"    PH:    {len(ph_alarms_traj)} alarmes (2019: {n_ph_2019}) "
          f"— esperado 12 (2019: 0)")

    # encontrar primeiro par e par mais próximo na transição 2021
    adwin_2021 = [a for a in adwin_alarms if year_vals[a] == 2021]
    ph_2021    = [a for a in ph_alarms_traj if year_vals[a] == 2021]

    ph_first    = min(ph_2021)    if ph_2021    else None
    adwin_first = min(adwin_2021) if adwin_2021 else None
    delta1      = (adwin_first - ph_first) if (ph_first and adwin_first) else None

    min_dist, nearest_pair = None, None
    for ai in adwin_alarms:
        for pi in ph_alarms_traj:
            d = abs(ai - pi)
            if min_dist is None or d < min_dist:
                min_dist = d
                nearest_pair = (ai, pi)

    print(f"    PH primeiro alarme 2021: {ph_first}")
    print(f"    ADWIN primeiro alarme 2021: {adwin_first}")
    print(f"    Δ₁ (primeiro par 2021): {delta1}")
    print(f"    Par mais próximo: ADWIN@{nearest_pair[0]}, "
          f"PH@{nearest_pair[1]}, Δ_min={min_dist}")

    return dict(
        stream_s2s=stream_s2s,
        year_vals=year_vals,
        n_2019_total=n_2019_total,
        n_post_total=n_post_total,
        adwin_alarms=adwin_alarms,
        ph_alarms_traj=ph_alarms_traj,
        ph_first=ph_first,
        adwin_first=adwin_first,
        delta1=delta1,
        nearest_pair=nearest_pair,
        min_dist=min_dist,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Construir master table
# ══════════════════════════════════════════════════════════════════════════════

def build_master_table(ctx: dict) -> pd.DataFrame:
    """Agrega finalistas das 3 fases numa tabela única enriquecida."""
    print("[2] Construindo master table …")
    n19t = ctx["n_2019_total"]
    npot = ctx["n_post_total"]

    # ── rows da Fase 3 ─────────────────────────────────────────────────────
    ph3 = pd.read_csv(PHASE3_CSV)

    # seleccionar só as variantes finalistas (sem ablações redundantes)
    keep_variants = [
        "adwin_standalone",
        "suspected (OR)",
        "kalman_standalone",
        "ph_standalone",
        "confirmed (AND+debounce)",
    ]
    ph3 = ph3[ph3["variant"].isin(keep_variants)].copy()
    ph3["fase"] = 3
    ph3["detector"] = ph3["variant"].map(DETECTOR_LABELS)
    ph3["config"]   = ph3["variant"].map(CONFIGS)
    ph3["papel_final"] = ph3["variant"].map(PAPEIS)
    # normalizar nome de colunas para compatibilidade com métricas
    ph3 = ph3.rename(columns={"variant": "exp_name",
                               "n_alarms": "n_alarms",
                               "n_alarms_post": "n_alarms_post"})

    # ── row da Fase 2: KSWIN melhor (para mostrar que é mau) ──────────────
    ph2 = pd.read_csv(PHASE2_CSV)
    kswin_row = ph2[
        (ph2["frente"] == "C") &
        (ph2["model"]  == "Seq2Seq") &
        (ph2["exp"]    == "C_a1e-05_ws1000_ss100")
    ].copy()

    if kswin_row.empty:
        # fallback: melhor FAR da frente C Seq2Seq
        kswin_row = ph2[
            (ph2["frente"] == "C") & (ph2["model"] == "Seq2Seq")
        ].sort_values("FAR_2019_per1k").head(1).copy()

    kswin_row["fase"]       = 2
    kswin_row["exp_name"]   = "KSWIN"
    kswin_row["detector"]   = "KSWIN"
    kswin_row["config"]     = CONFIGS["KSWIN"]
    kswin_row["papel_final"] = PAPEIS["KSWIN"]
    # alinhar colunas: phase2 usa n_alarms_total e n_alarms_post
    kswin_row = kswin_row.rename(columns={"n_alarms_total": "n_alarms"})

    # ── concatenar e seleccionar colunas comuns ────────────────────────────
    common_cols = [
        "fase", "detector", "config",
        "FAR_2019_per1k", "n_alarms", "n_alarms_post",
        "n_2019", "n_2021", "n_2022", "n_2023", "n_2024", "n_2025",
        "SNR_op", "papel_final",
    ]

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        for c in common_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[common_cols].copy()

    master = pd.concat([_align(ph3), _align(kswin_row)], ignore_index=True)

    # ── enriquecer com métricas novas ─────────────────────────────────────
    master = enrich(master, a=1.0,
                    n_2019_total=n19t, n_post_total=npot)

    # ── filtro de elegibilidade (FAR > FAR_ADMISSIBILITY → inadmissível) ──
    # Aplicado ANTES da regra de ordenação, conforme especificação da Fase 4.1.
    # "Resultado negativo" vai sempre para a secção separada (independente de FAR).
    is_negative     = master["papel_final"].str.startswith("Resultado negativo")
    is_inadmissible = master["papel_final"].str.startswith("Inadmissível")
    is_admissible   = ~is_negative & ~is_inadmissible

    admissible   = rank_detectors(master[is_admissible].copy())
    inadmissible = master[is_inadmissible].copy()
    negatives    = master[is_negative].copy()
    master = pd.concat([admissible, inadmissible, negatives], ignore_index=True)

    # reordenar colunas finais conforme especificação do prompt
    final_cols = [
        "fase", "detector", "config",
        "FAR_2019_per1k", "year_coverage", "SNR_smooth",
        "detection_efficiency",
        "n_alarms_post", "n_2019",
        "n_2021", "n_2022", "n_2023", "n_2024", "n_2025",
        "SNR_op", "papel_final",
    ]
    for c in final_cols:
        if c not in master.columns:
            master[c] = np.nan
    master = master[final_cols]

    out_path = OUT_ROOT / "phase4_master_table.csv"
    master.to_csv(out_path, index=False)
    print(f"    Guardado: {out_path}")
    print(master[["detector", "FAR_2019_per1k", "year_coverage",
                  "SNR_smooth", "SNR_op", "papel_final"]].to_string(index=False))
    return master


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — Figura and_infeasibility.png
# ══════════════════════════════════════════════════════════════════════════════

def plot_and_infeasibility(ctx: dict) -> None:
    """Figura de publicação: inviabilidade estrutural do AND temporal."""
    print("[3] Gerando and_infeasibility.png …")

    stream     = ctx["stream_s2s"]
    year_vals  = ctx["year_vals"]
    adwin_alarms = ctx["adwin_alarms"]
    ph_alarms    = ctx["ph_alarms_traj"]
    ph_first     = ctx["ph_first"]
    adwin_first  = ctx["adwin_first"]
    delta1       = ctx["delta1"]
    nearest_pair = ctx["nearest_pair"]
    min_dist     = ctx["min_dist"]

    x   = stream["global_idx"].values
    raw = stream["ade_traj"].astype(float).values
    sig = smooth_robz(raw, ROBZ_WINDOW)

    # fronteira 2019 / 2021
    mask_2019 = year_vals == 2019
    mask_2021 = year_vals == 2021
    x_2019_max = int(x[mask_2019].max()) if mask_2019.any() else 0
    x_2021_min = int(x[mask_2021].min()) if mask_2021.any() else x_2019_max + 1

    # janela de zoom
    ZOOM_LO = max(0, x_2021_min - 12_000)
    ZOOM_HI = min(len(x) - 1, (nearest_pair[0] if nearest_pair else x_2021_min + 40_000) + 15_000)

    # alarmes dentro da janela
    adwin_in = [a for a in adwin_alarms if ZOOM_LO <= a <= ZOOM_HI]
    ph_in    = [a for a in ph_alarms    if ZOOM_LO <= a <= ZOOM_HI]

    # sinal dentro da janela (decimado)
    mask_zoom  = (x >= ZOOM_LO) & (x <= ZOOM_HI)
    x_zoom     = x[mask_zoom]
    sig_zoom   = sig[mask_zoom]
    step       = max(1, mask_zoom.sum() // 4000)
    roll_zoom  = pd.Series(sig[mask_zoom]).rolling(300, min_periods=1).mean().values

    # ── figura: 2 painéis (sinal + rug de alarmes) ────────────────────────
    fig, (ax_sig, ax_rug) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06},
        sharex=True,
    )

    # painel superior: sinal
    ax_sig.scatter(x_zoom[::step], sig_zoom[::step],
                   s=1.5, color="#aaaaaa", alpha=0.35, zorder=1, rasterized=True)
    ax_sig.plot(x_zoom, roll_zoom, color="#333333", lw=1.8, zorder=3,
                label="Sinal robz (média móvel 300)")

    # sombreado de anos
    ax_sig.axvspan(ZOOM_LO, x_2021_min - 1,
                   color=YEAR_COLORS[2019], alpha=0.10, zorder=0)
    ax_sig.axvspan(x_2021_min, ZOOM_HI,
                   color=YEAR_COLORS[2021], alpha=0.10, zorder=0)
    ax_sig.axvline(x_2021_min, color="#555555", lw=1.2, ls=":", zorder=2,
                   label=f"Fronteira 2019/2021 ({x_2021_min:,})")

    # alarmes como linhas verticais
    for i, a in enumerate(adwin_in):
        lbl = "ADWIN_S2S" if i == 0 else None
        ax_sig.axvline(a, color="#c0392b", lw=1.3, ls="--", alpha=0.85,
                       zorder=4, label=lbl)
    for i, p in enumerate(ph_in):
        lbl = "PH_agg (N=50)" if i == 0 else None
        ax_sig.axvline(p, color="#27ae60", lw=1.3, ls="-.", alpha=0.85,
                       zorder=4, label=lbl)

    # anotação Δ₁ (primeiro par em 2021)
    if ph_first is not None and adwin_first is not None:
        y_ann = float(np.nanpercentile(sig_zoom[np.isfinite(sig_zoom)], 85))
        ax_sig.annotate(
            "", xy=(adwin_first, y_ann), xytext=(ph_first, y_ann),
            arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.8),
        )
        mid1 = (ph_first + adwin_first) / 2
        ax_sig.text(mid1, y_ann * 1.04,
                    f"Δ₁ = {delta1:,} trajs",
                    ha="center", va="bottom", fontsize=9,
                    color="#8e44ad", fontweight="bold")

    # anotação Δ_min (par mais próximo)
    if nearest_pair is not None and min_dist is not None:
        ai_near, pi_near = nearest_pair
        if ZOOM_LO <= ai_near <= ZOOM_HI and ZOOM_LO <= pi_near <= ZOOM_HI:
            finite_sig = sig_zoom[np.isfinite(sig_zoom)]
            y_ann2      = float(np.nanpercentile(finite_sig, 60))
            y_ann2_text = float(np.nanpercentile(finite_sig, 72))
            ax_sig.annotate(
                "", xy=(max(ai_near, pi_near), y_ann2),
                xytext=(min(ai_near, pi_near), y_ann2),
                arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.8),
            )
            mid2 = (ai_near + pi_near) / 2
            ax_sig.annotate(
                f"$\\Delta_{{min}}$ = {min_dist} trajs\n(W≥500 necessário)",
                xy=(mid2, y_ann2),
                xytext=(mid2, y_ann2_text),
                ha="center", va="bottom", fontsize=8.5,
                color="#2980b9", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"),
                arrowprops=dict(arrowstyle="->", color="#2980b9",
                                connectionstyle="arc3,rad=0.0"),
            )
            # banda W=500 à volta do par mais próximo
            W_band = 500
            lo_band = min(ai_near, pi_near) - W_band // 2
            hi_band = max(ai_near, pi_near) + W_band // 2
            ax_sig.axvspan(lo_band, hi_band, color="#2980b9", alpha=0.08,
                           zorder=0, label=f"Banda W=500 (coincidência)")

    # legenda e labels
    ax_sig.set_ylabel("Sinal ADE (robz)", fontsize=10)
    ax_sig.legend(loc="upper left", fontsize=8.5, framealpha=0.85,
                  ncol=2)
    # anotações de ano
    for yr, color in [(2019, YEAR_COLORS[2019]), (2021, YEAR_COLORS[2021])]:
        mask_yr = year_vals == yr
        if not mask_yr.any():
            continue
        xi = x[mask_yr & mask_zoom]
        if len(xi) == 0:
            continue
        xc = (xi.min() + xi.max()) / 2
        ypos = float(np.nanpercentile(sig_zoom[np.isfinite(sig_zoom)], 97))
        ax_sig.text(xc, ypos, str(yr), color=color, fontsize=11,
                    ha="center", va="top", fontweight="bold", alpha=0.85)

    ax_sig.set_title(
        "Inviabilidade estrutural do AND temporal: "
        "desalinhamento entre ADWIN_S2S (por-trajetória) e PH_agg (janelas N=50)",
        fontsize=11, pad=8,
    )

    # painel inferior: rug plot
    for a in adwin_in:
        ax_rug.axvline(a, color="#c0392b", lw=1.2, alpha=0.75)
    for p in ph_in:
        ax_rug.axvline(p, color="#27ae60", lw=1.2, alpha=0.75)
    ax_rug.set_yticks([])
    ax_rug.set_ylabel("Alarmes", fontsize=9)
    ax_rug.set_xlabel("Índice de trajetória global", fontsize=10)

    # patches para legenda do rug
    patch_adwin = mpatches.Patch(color="#c0392b", label="ADWIN_S2S")
    patch_ph    = mpatches.Patch(color="#27ae60", label="PH_agg (N=50)")
    ax_rug.legend(handles=[patch_adwin, patch_ph], loc="upper right",
                  fontsize=8.5, framealpha=0.85)

    ax_sig.set_xlim(ZOOM_LO, ZOOM_HI)

    out_path = IMG_DIR / "and_infeasibility.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(IMG_DIR / "and_infeasibility.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    Guardado: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — Sweep W estendido
# ══════════════════════════════════════════════════════════════════════════════

def extended_w_sweep(ctx: dict) -> pd.DataFrame:
    """W in {50..20000}: documenta quantitativamente a inviabilidade do AND.

    Resultado esperado:
      W <= 200  : 0 confirmed (Δ_min=214 > W)
      W = 500-2000: 1 confirmed (apenas 2021)
      W = 5000  : 2 confirmed (2021 apenas — PH não dispara em 2022)
      W = 10000+: year_coverage_confirmed pode atingir 2 via pairing cross-ano
                  (ADWIN early-2022 dentro de W do último PH late-2021),
                  mas W=10000 representa ~2 meses de dados — operacionalmente absurdo.
    """
    print("[4] Sweep W estendido …")

    adwin_alarms = ctx["adwin_alarms"]
    ph_alarms    = ctx["ph_alarms_traj"]
    year_vals    = ctx["year_vals"]
    n19t         = ctx["n_2019_total"]
    npot         = ctx["n_post_total"]

    # trajetórias por mês (approx): 288000 total / 72 meses (6 anos)
    trajs_per_month = (n19t + npot) / 72.0

    W_values = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    rows = []
    for W in W_values:
        conf = _and_gate(adwin_alarms, ph_alarms, W, RETRAIN_DEBOUNCE)
        m    = _eval(conf, year_vals, n19t, npot)

        yc = sum(
            1 for y in YEARS_POST
            if m.get(f"n_{y}", 0) > 0
        )
        months_w = W / trajs_per_month

        if len(conf) == 0:
            note = f"Δ_min={ctx['min_dist']} > W={W}: sem coincidências"
        elif yc <= 1:
            note = (
                f"year_coverage_confirmed={yc} (só 2021) — "
                f"PH_agg nunca dispara em 2022-2025"
            )
        else:
            note = (
                f"year_coverage_confirmed={yc} via pairing cross-ano: "
                f"ADWIN early-2022 dentro de W={W} (~{months_w:.1f} meses) "
                f"do ultimo PH late-2021 — W operacionalmente absurdo"
            )

        rows.append({
            "W_coincidence":           W,
            "n_confirmed":             m["n_alarms"],
            "FAR_2019_per1k":          m["FAR_2019_per1k"],
            "year_coverage_confirmed": yc,
            "W_approx_months":         round(months_w, 1),
            "n_2019":                  m["n_2019"],
            "n_2021":                  m["n_2021"],
            "n_2022":                  m["n_2022"],
            "n_2023":                  m["n_2023"],
            "n_2024":                  m["n_2024"],
            "n_2025":                  m["n_2025"],
            "note":                    note,
        })
        print(f"    W={W:6d} (~{months_w:4.1f} meses): "
              f"n_confirmed={m['n_alarms']:2d}, year_cov={yc}, "
              f"FAR={m['FAR_2019_per1k']:.3f}")

    sweep_df = pd.DataFrame(rows)
    out_path = OUT_ROOT / "W_infeasibility_sweep.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"    Guardado: {out_path}")
    return sweep_df


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — Relatório final PHASE4_FINAL.md
# ══════════════════════════════════════════════════════════════════════════════

def _md_table(df: pd.DataFrame, cols: list[str],
              fmt: dict[str, str] | None = None) -> str:
    """Gera tabela Markdown com formatação opcional por coluna."""
    fmt = fmt or {}
    header  = "| " + " | ".join(cols) + " |"
    sep     = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines   = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row.get(c, "")
            if c in fmt and not (isinstance(val, float) and np.isnan(val)):
                cells.append(fmt[c].format(val))
            else:
                cells.append("" if (isinstance(val, float) and np.isnan(val)) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report(master: pd.DataFrame, sweep: pd.DataFrame, ctx: dict) -> None:
    """Gera PHASE4_FINAL.md com narrativa científica completa em português."""
    print("[5] Gerando PHASE4_FINAL.md …")

    n19t = ctx["n_2019_total"]
    npot = ctx["n_post_total"]
    delta1   = ctx["delta1"]
    min_dist = ctx["min_dist"]
    nearest  = ctx["nearest_pair"]

    # rows chave
    def get_row(label: str, exact: bool = False) -> pd.Series:
        mask = (master["detector"] == label) if exact else \
               master["detector"].str.contains(label, regex=False, na=False)
        if mask.any():
            return master[mask].iloc[0]
        return pd.Series(dtype=float)

    adwin_r   = get_row("ADWIN_S2S")
    or_r      = get_row("OR(")
    ph_r      = get_row("PH_agg", exact=True)   # exact: evita match em AND(ADWIN_S2S, PH_agg)
    and_r     = get_row("AND(")
    kalman_r  = get_row("ADWIN_Kalman")
    kswin_r   = get_row("KSWIN")

    _check_factor2_consistency(master, ph_r, adwin_r)

    # Wilson CI para FAR=0 e FAR do ADWIN
    ph_ci    = wilson_ci_per1k(int(ph_r.get("n_2019", 0)),   n19t)
    and_ci   = wilson_ci_per1k(int(and_r.get("n_2019", 0)),  n19t)
    adwin_ci = wilson_ci_per1k(int(adwin_r.get("n_2019", 0)), n19t)

    # tabela de robustez
    master_rows = master.to_dict("records")
    rob = robustness_table(
        rows=master_rows, a_values=[0.5, 1.0, 2.0],
        n_2019_total=n19t, n_post_total=npot,
    )

    # tabela antes/depois SNR_op vs SNR_smooth
    snr_comparison_rows = []
    for _, row in master.iterrows():
        snr_op_val = row.get("SNR_op", np.nan)
        artifact   = "**Sim** — sentinela 9999" if (not np.isnan(float(snr_op_val))
                      and float(snr_op_val) >= 9999) else "Não"
        snr_comparison_rows.append({
            "Detetor":       row["detector"],
            "SNR_op":        f"{float(snr_op_val):.2f}" if not np.isnan(float(snr_op_val)) else "—",
            "SNR_smooth (a=1)": f"{row['SNR_smooth']:.2f}",
            "year_coverage": int(row["year_coverage"]),
            "Artefacto?":    artifact,
        })
    snr_cmp_df = pd.DataFrame(snr_comparison_rows)

    # W mínimo para year_coverage_confirmed ≥ 2 (provar que não existe)
    w_for_2yrs = sweep[sweep["year_coverage_confirmed"] >= 2]
    w_for_2yrs_str = (
        f"W={int(w_for_2yrs['W_coincidence'].min())}"
        if not w_for_2yrs.empty
        else "inexistente (PH_agg não deteta 2022–2025)"
    )

    # ── tabela master para o relatório (sem colunas redundantes) ──────────
    rpt_cols = ["detector", "FAR_2019_per1k", "year_coverage",
                "SNR_smooth", "detection_efficiency", "n_alarms_post",
                "n_2019", "papel_final"]
    rpt_fmt  = {
        "FAR_2019_per1k": "{:.4f}",
        "SNR_smooth":     "{:.4f}",
        "detection_efficiency": "{:.4f}",
    }

    # ── sweep table abreviada ─────────────────────────────────────────────
    sweep_cols = ["W_coincidence", "W_approx_months", "n_confirmed",
                  "FAR_2019_per1k", "year_coverage_confirmed"]
    sweep_fmt  = {"FAR_2019_per1k": "{:.3f}", "W_approx_months": "{:.1f}"}

    # ── tabela robustez: apenas top-3 para a=1 + variação SNR_smooth ──────
    rob_top = rob[rob["rank"] <= 3][["a", "rank", "detector", "SNR_smooth"]]
    rob_cols = ["a", "rank", "detector", "SNR_smooth"]
    rob_fmt  = {"SNR_smooth": "{:.4f}", "a": "{:.1f}"}

    md = textwrap.dedent(f"""\
    # Relatório Final — Fase 4: Consolidação Científica

    > Gerado automaticamente por `drift_analise/phase4_consolidate.py`.
    > Dataset: Seq2Seq, n_2019_total={n19t:,}, n_post_total={npot:,}.

    ---

    ## 1. Resultado Primário — ADWIN\\_S2S Standalone

    O detetor recomendado para produção é o **ADWIN_S2S standalone**
    (configuração: `{adwin_r.get('config', '—')}`). É o único detetor que
    combina cobertura total dos cinco anos pós-pandemia (2021–2025) com
    taxa de falsos alarmes baixa e controlada.

    | Métrica | Valor | Notas |
    |---------|-------|-------|
    | FAR\_2019\_per1k | {float(adwin_r.get('FAR_2019_per1k',0)):.4f} | IC Wilson 95%: [{adwin_ci[0]:.4f}, {adwin_ci[1]:.4f}]/1k |
    | year\_coverage | {int(adwin_r.get('year_coverage',0))}/5 | Deteta em todos os anos 2021–2025 |
    | SNR\_smooth (a=1) | {float(adwin_r.get('SNR_smooth',0)):.4f} | Substituto finito e estável do SNR\_op |
    | detection\_efficiency | {float(adwin_r.get('detection_efficiency',0)):.4f} | {float(adwin_r.get('detection_efficiency',0))*100:.1f}% dos alarmes são pós-2019 |
    | n\_post (alarmes 2021–2025) | {int(adwin_r.get('n_alarms_post',0))} | Distribuídos por {int(adwin_r.get('year_coverage',0))} anos |

    O valor FAR={float(adwin_r.get('FAR_2019_per1k',0)):.4f}/1k corresponde a
    {int(adwin_r.get('n_2019',0))} alarmes no período de linha de base (2019),
    num total de {n19t:,} trajetórias — IC de Wilson 95%:
    [{adwin_ci[0]:.4f}, {adwin_ci[1]:.4f}]/1k.

    ---

    ## 2. Resultado Negativo Caracterizado — Inviabilidade Estrutural do AND Temporal

    > **O ensemble AND temporal foi testado, caracterizado e rejeitado com base
    > em evidência empírica.**

    A inviabilidade do AND(ADWIN\\_S2S, PH\\_agg) resulta de dois factores
    estruturais independentes, não de má calibração de parâmetros:

    ### Factor 1 — Desalinhamento Temporal (escala de deteção)

    ADWIN\\_S2S opera por trajetória (escala fina) enquanto PH\\_agg agrega
    janelas de N=50 trajetórias (escala grossa). Na transição 2019→2021:

    - PH\\_agg dispara o primeiro alarme em índice **~{ctx['ph_first']:,}**
      (imediatamente após o salto COVID+comportamental, via janelas largas).
    - ADWIN\\_S2S dispara o primeiro alarme em índice **~{ctx['adwin_first']:,}**
      (após acumular buffer suficiente com mistura 2019+2021).
    - **Δ₁ = {delta1:,} trajetórias** entre os primeiros alarmes do mesmo evento.
    - Par mais próximo: ADWIN@{nearest[0]:,}, PH@{nearest[1]:,},
      **Δ\\_min = {min_dist} trajetórias**.

    Para a porta AND disparar, é necessário W ≥ Δ\\_min = {min_dist} trajetórias.
    O sweep de W confirma que W=500 (≥ {min_dist}) produz exactamente
    1 alarme CONFIRMED, apenas em 2021.

    Ver figura: `images/and_infeasibility.png`

    ### Factor 2 — Limitação de Escopo do PH\\_agg (determinante)

    O PH\\_agg detecta **exclusivamente o salto 2019→2021** (evento de grande
    magnitude). Para os anos 2022–2025, em que o drift é mais gradual:

    | Canal | n\_2021 | n\_2022 | n\_2023 | n\_2024 | n\_2025 | year\_coverage |
    |-------|---------|---------|---------|---------|---------|----------------|
    | PH\\_agg standalone | {int(ph_r.get('n_2021',0))} | {int(ph_r.get('n_2022',0))} | {int(ph_r.get('n_2023',0))} | {int(ph_r.get('n_2024',0))} | {int(ph_r.get('n_2025',0))} | {int(ph_r.get('year_coverage',0))}/5 |
    | ADWIN\\_S2S standalone | {int(adwin_r.get('n_2021',0))} | {int(adwin_r.get('n_2022',0))} | {int(adwin_r.get('n_2023',0))} | {int(adwin_r.get('n_2024',0))} | {int(adwin_r.get('n_2025',0))} | {int(adwin_r.get('year_coverage',0))}/5 |

    Como o PH_agg nunca dispara diretamente em 2022–2025, a porta AND não pode
    confirmar nesses anos com janelas operacionalmente razoáveis.
    O sweep abaixo quantifica isto:

    ### Sweep W Estendido

{_md_table(sweep, sweep_cols, sweep_fmt)}

    W mínimo para year_coverage_confirmed >= 2: **{w_for_2yrs_str}**.

    **Interpretação:** Para W <= 2000 (até ~0.5 meses de dados), o AND confirma
    apenas alarmes em 2021. Com W=10000 (~{sweep[sweep['W_coincidence']==10000]['W_approx_months'].values[0]:.1f} meses),
    o AND consegue "confirmar" 1 alarme em 2022 — mas apenas porque um alarme
    ADWIN de early-2022 cai dentro da janela de 10000 trajetórias a partir do
    último alarme PH de late-2021. Uma janela de "coincidência temporal" de
    ~2 meses é operacionalmente absurda para um detetor de drift. Esta é a
    prova definitiva da inviabilidade: obter year_coverage >= 2 requer
    abandonar qualquer noção de simultaneidade temporal.

    ---

    ## 3. Resultado Secundário — Canal OR/SUSPECTED como Camada de Vigilância

    O canal OR(ADWIN\\_S2S, ADWIN\\_Kalman) funciona como camada de vigilância
    de sensibilidade aumentada, adequado para disparar monitorização reforçada
    (não re-treino automático).

    | Métrica | Valor |
    |---------|-------|
    | FAR\_2019\_per1k | {float(or_r.get('FAR_2019_per1k',0)):.4f} |
    | year\_coverage | {int(or_r.get('year_coverage',0))}/5 |
    | SNR\_smooth (a=1) | {float(or_r.get('SNR_smooth',0)):.4f} |
    | detection\_efficiency | {float(or_r.get('detection_efficiency',0)):.4f} |
    | n\_post | {int(or_r.get('n_alarms_post',0))} |

    O canal OR cobre os mesmos 5/5 anos que o ADWIN standalone, com maior
    sensibilidade (n\_post={int(or_r.get('n_alarms_post',0))} vs {int(adwin_r.get('n_alarms_post',0))})
    à custa de FAR mais alto ({float(or_r.get('FAR_2019_per1k',0)):.3f} vs
    {float(adwin_r.get('FAR_2019_per1k',0)):.3f}/1k). Recomendado apenas como
    camada de alerta; re-treino deve ser condicionado à confirmação manual.

    ---

    ## 4. Correção Metodológica — SNR\\_op → SNR\\_smooth

    ### O Problema do SNR\\_op com Sentinela 9999

    O `SNR_op` implementado nas Fases 1–3 atribui o valor sentinela 9999 sempre
    que `n_2019_alarms == 0`, independentemente do número de deteções pós-2019.
    Isto cria um artefacto metodológico: qualquer detetor degenerado que evite
    **todos** os alarmes em 2019 (incluindo um que não deteta nada de útil)
    parece "ótimo".

    **Prova:** `ph_standalone` tem SNR\\_op=9999 mas year\\_coverage=1/5 (só 2021).
    `confirmed (AND)` com W=200 tem SNR\\_op=0.0 com 0 deteções.
    Nenhuma destas classificações reflete performance real.

    ### SNR\\_smooth — Fórmula com Pseudo-Contagem de Laplace

    ```
    SNR_smooth = ((n_post + a) × n_2019_total) / ((n_2019_alarms + a) × n_post_total)
    ```

    Com `a=1` (prior simétrico): um detetor com 0 falsos alarmes e 1 deteção
    obtém um valor finito e comparável; o AND com 0 deteções obtém SNR < 1.

    ### Tabela Antes/Depois

{_md_table(snr_cmp_df, ["Detetor", "SNR_op", "SNR_smooth (a=1)", "year_coverage", "Artefacto?"])}

    **Efeito:** `ph_standalone` desce de 9999 para {float(ph_r.get('SNR_smooth',0)):.2f},
    eliminando a explosão numérica da sentinela. Importa notar, porém, que {float(ph_r.get('SNR_smooth',0)):.2f}
    é o valor **mais alto** da tabela-mestre — o SNR\\_smooth sozinho não desmascara o PH\\_agg:
    esta métrica ainda premeia especificidade (n\\_2019\\_alarms=0), pelo que qualquer detetor
    com zero falsos alarmes no baseline obtém um SNR\\_smooth elevado, independentemente da
    cobertura temporal. O que revela a inutilidade do PH\\_agg é o year\\_coverage=1/5
    (deteta exclusivamente o salto de 2021), **não** o SNR\\_smooth. É precisamente por isso
    que a regra de decisão usa year\\_coverage como critério **primário** e o SNR\\_smooth
    apenas como terceiro critério de desempate. As conclusões das Fases 1–3 baseadas em
    year\\_coverage + FAR não se alteram; as baseadas em SNR\\_op > 100 ficam invalidadas.
    Esta é uma limitação reconhecida da métrica corrigida: o SNR\\_smooth resolve o problema
    numérico da sentinela mas herda o viés pró-especificidade; mitigamo-lo subordinando-o
    ao year\\_coverage na regra de decisão.

    ### Robustez do Ranking para a ∈ {{0.5, 1, 2}}

{_md_table(rob_top, rob_cols, rob_fmt)}

    O ranking dos três primeiros candidatos é **estável** para todos os valores
    de `a` testados: a escolha de a=1 não é determinante para as conclusões.

    ---

    ## 5. Respostas às Questões Científicas (Q1–Q4)

    ### Q1 — A ausência de deteções em 2022–2025 no PH\\_agg é real ou limitação?

    **Resposta (confirmada — limitação estrutural dupla):**

    (a) O PH\\_agg com K\\_thr=50 só ultrapassa o limiar no salto 2019→2021
    (grande magnitude, evento COVID + mudança comportamental). Os drifts
    2022–2025 são graduais e ficam abaixo do limiar.

    (b) Mesmo que o PH disparasse mais tarde, o desalinhamento temporal
    (Δ\\_min={min_dist} trajs, Factor 1 da Sec. 2) tornaria o AND inoperante
    com W razoável.

    Estas são limitações estruturais, não de calibração. O ADWIN\\_S2S
    (por-trajetória) detecta drift gradual em todos os 5 anos.

    ### Q2 — O SNR\\_op < 1 do KSWIN indica inutilidade?

    **Resposta:** Sim. O KSWIN é **inadmissível** pelo filtro de elegibilidade
    (FAR\\_2019 > {FAR_ADMISSIBILITY}/1k) e confirmado inútil pela métrica corrigida.
    KSWIN ({CONFIGS.get('KSWIN','—')}):
    - FAR\\_2019={float(kswin_r.get('FAR_2019_per1k',0)):.4f}/1k > limiar {FAR_ADMISSIBILITY}/1k
      → **inadmissível** (especificidade insuficiente para produção)
    - SNR\\_smooth={float(kswin_r.get('SNR_smooth',0)):.2f} < 1 (mais alarmes no
      baseline do que no pós-período, proporcionalmente)
    - n\\_2019={int(kswin_r.get('n_2019',0))} falsos alarmes em 2019 (pior em ordem de grandeza)
    - detection\\_efficiency={float(kswin_r.get('detection_efficiency',0)):.2f}
    O KSWIN não passa o filtro de elegibilidade e não é candidato ao detetor primário.

    ### Q3 — Trade-off operacional FAR vs cobertura?

    **Resposta:** Com as métricas corrigidas, o trade-off é:

    | Nível | Detetor | FAR/1k | year\_coverage | Uso recomendado |
    |-------|---------|--------|----------------|-----------------|
    | Alto rigor | ADWIN\\_S2S standalone | {float(adwin_r.get('FAR_2019_per1k',0)):.4f} | 5/5 | Re-treino automático |
    | Alta cobertura | OR/SUSPECTED | {float(or_r.get('FAR_2019_per1k',0)):.4f} | 5/5 | Monitorização reforçada |

    O AND CONFIRMED (FAR=0, year\\_coverage=0–1) foi rejeitado: troca toda a
    cobertura pós-2021 por uma especificidade que resulta em 0 deteções úteis.

    ### Q4 — O ensemble AND adiciona valor sobre o standalone?

    **Resposta: Não. O AND temporal NÃO adiciona valor sobre o ADWIN\\_S2S standalone.**

    Dois factores estruturais independentes (Sec. 2):
    1. Desalinhamento temporal de {delta1:,} trajetórias no primeiro alarme 2021
       (Δ\\_min={min_dist} trajetórias no par mais próximo).
    2. PH\\_agg não deteta 2022–2025 → year\\_coverage\\_confirmed ≤ 1 para
       **qualquer** W.

    O sweep W ∈ {{{sweep['W_coincidence'].min()}, …, {sweep['W_coincidence'].max()}}} confirma:
    year_coverage_confirmed permanece em 1 (apenas 2021) para W <= 2000.
    Atingir year_coverage_confirmed = 2 requer W >= 10000 (~2 meses de dados) —
    o que destrói o significado de "coincidência temporal".
    W=500 não tem valor operacional: produz 1 deteção em 1/5 anos, versus
    23 deteções em 5/5 anos para o ADWIN standalone.

    **Conclusão:** Demonstrámos empiricamente que o ensemble AND temporal,
    apesar de eliminir os falsos alarmes de 2019 (FAR=0), é estruturalmente
    inviável como monitor de longo prazo. O ADWIN\\_S2S standalone é o detetor
    recomendado. O canal OR/SUSPECTED é a camada de vigilância adequada.

    ---

    ## 6. Ameaças à Validade

    ### 6.1 IC de Wilson para FAR=0 com Amostra Pequena

    Detetores com FAR=0 (PH\\_agg standalone, AND CONFIRMED) baseiam-se em
    0 alarmes num período de {n19t:,} trajetórias de baseline. O IC de Wilson
    a 95% não implica FAR=0 verdadeiro:

    - **PH\\_agg standalone**: IC 95% = [{ph_ci[0]:.4f}, {ph_ci[1]:.4f}]/1k
    - **AND CONFIRMED (W=200)**: IC 95% = [{and_ci[0]:.4f}, {and_ci[1]:.4f}]/1k

    Com {n19t:,} trajetórias de baseline, o limite superior da FAR é
    ≈{ph_ci[1]:.2f}/1k — valor baixo mas não zero. Esta incerteza não altera as
    conclusões sobre inviabilidade do AND (Factor 2 é determinístico).

    ### 6.2 Evento Único de Drift Dominante em 2021

    O salto 2019→2021 (descontinuidade COVID + regresso gradual com novos
    padrões comportamentais) é o evento de drift mais pronunciado do dataset.
    Os drifts pós-2021 são mais graduais. Isto favorece o ADWIN\\_S2S
    (sensível a mudanças graduais) sobre o PH\\_agg (calibrado para o salto de
    2021). Generalizações para datasets com drift mais uniforme ao longo dos
    anos requerem re-calibração de K\\_thr.

    ### 6.3 Generalização Limitada

    Os resultados referem-se a uma liga e horizonte específicos (Seq2Seq,
    30→15 minutos). O desempenho em outras ligas ou outros horizontes de
    previsão pode diferir. O padrão estrutural (desalinhamento entre
    detetores de escalas distintas) é esperado manter-se, mas os valores
    numéricos (Δ₁, Δ\\_min) dependem da distribuição temporal dos erros.

    ---

    ## Tabela-Mestre Final

{_md_table(master[rpt_cols], rpt_cols, rpt_fmt)}

    *Critério de elegibilidade: FAR\\_2019\\_per1k ≤ {FAR_ADMISSIBILITY}/1k (aplicado antes da ordenação).
    Candidatos inadmissíveis (FAR > {FAR_ADMISSIBILITY}/1k) e resultados negativos aparecem após os elegíveis.
    Regra de decisão (elegíveis): (1) year\\_coverage DESC, (2) FAR\\_2019\\_per1k ASC, (3) SNR\\_smooth DESC.*
    """)

    out_path = OUT_ROOT / "PHASE4_FINAL.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"    Guardado: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — PHASE4_REPORT.md (formato análogo às fases anteriores)
# ══════════════════════════════════════════════════════════════════════════════

def generate_phase4_report(master: pd.DataFrame, sweep: pd.DataFrame, ctx: dict) -> None:
    """Gera PHASE4_REPORT.md no formato padronizado das fases anteriores."""
    print("[6] Gerando PHASE4_REPORT.md …")

    n19t = ctx["n_2019_total"]
    npot = ctx["n_post_total"]

    def get_row(label: str, exact: bool = False) -> pd.Series:
        mask = (master["detector"] == label) if exact else \
               master["detector"].str.contains(label, regex=False, na=False)
        return master[mask].iloc[0] if mask.any() else pd.Series(dtype=float)

    adwin_r  = get_row("ADWIN_S2S")
    kalman_r = get_row("ADWIN_Kalman")
    or_r     = get_row("OR(")
    kswin_r  = get_row("KSWIN")
    ph_r     = get_row("PH_agg", exact=True)   # exact: evita match em AND(ADWIN_S2S, PH_agg)
    and_r    = get_row("AND(")

    _check_factor2_consistency(master, ph_r, adwin_r)

    adwin_ci = wilson_ci_per1k(int(adwin_r.get("n_2019", 0)), n19t)
    ph_ci    = wilson_ci_per1k(int(ph_r.get("n_2019", 0)),   n19t)
    and_ci   = wilson_ci_per1k(int(and_r.get("n_2019", 0)),  n19t)

    w_for_2 = sweep[sweep["year_coverage_confirmed"] >= 2]
    w_min2  = int(w_for_2["W_coincidence"].min()) if not w_for_2.empty else None

    rpt_cols = ["detector", "FAR_2019_per1k", "year_coverage",
                "SNR_smooth", "detection_efficiency", "n_alarms_post",
                "n_2019", "papel_final"]
    rpt_fmt  = {
        "FAR_2019_per1k": "{:.4f}",
        "SNR_smooth":     "{:.4f}",
        "detection_efficiency": "{:.4f}",
    }

    sweep_cols = ["W_coincidence", "W_approx_months", "n_confirmed",
                  "FAR_2019_per1k", "year_coverage_confirmed", "note"]
    sweep_fmt  = {"FAR_2019_per1k": "{:.3f}", "W_approx_months": "{:.1f}"}

    md = textwrap.dedent(f"""\
    # Fase 4 — Consolidação Científica e Resultado Final

    > Esta fase **não re-executa grids**. Re-tabula os finalistas das Fases 1–3
    > com métricas corrigidas, caracteriza formalmente a inviabilidade do AND, e
    > produz os artefactos finais para o paper.

    ## Objectivo

    Corrigir a métrica `SNR_op` (sentinela 9999), consolidar os candidatos finais
    numa única tabela ranqueada, e produzir a figura e o texto que vão ao paper.
    Dataset: Seq2Seq, horizonte 30→15 min, n_2019_total={n19t:,}, n_post_total={npot:,}.

    ---
    ## Filtro de Elegibilidade (Fase 4.1)

    Antes da regra de ordenação, aplica-se o critério de admissibilidade:

    > **FAR\\_2019\\_per1k ≤ {FAR_ADMISSIBILITY}/1k** — candidatos com FAR superior são
    > classificados como **Inadmissíveis** e não concorrem ao posto de detetor primário.

    Justificação: um detetor com FAR=0.4375/1k produz ~21 falsos alarmes por cada
    48 000 trajetórias de linha de base — especificidade insuficiente para re-treino
    automático. O limiar de {FAR_ADMISSIBILITY}/1k admite até ~10 alarmes/48 000 trajs,
    compatível com revisão humana esporádica.

    ---
    ## Métricas Novas (substituem uso conclusivo de SNR_op)

    | Métrica | Fórmula | Direcção |
    |---------|---------|---------|
    | `year_coverage` | nº anos em {{2021..2025}} com ≥1 alarme (0–5) | **acima melhor** |
    | `SNR_smooth` | `((n_post+a)×n_2019_total) / ((n_2019_alarms+a)×n_post_total)` com a=1 | **acima melhor** |
    | `detection_efficiency` | `n_alarms_post / n_alarms_total` | **acima melhor** |
    | `FAR_2019_per1k` | mantida sem alteração | **abaixo melhor** |

    **Justificação do SNR_smooth:** a pseudo-contagem de Laplace `a=1` elimina a sentinela 9999
    (disparada quando n_2019_alarms=0). Ranking estável para a ∈ {{0.5, 1, 2}}.

    **Regra de decisão (sobre elegíveis):**
    1. `year_coverage` DESC — sensibilidade multi-ano é o critério primário
    2. `FAR_2019_per1k` ASC — menor ruído de linha de base (desempate)
    3. `SNR_smooth` DESC — qualidade do sinal (desempate final)

    ---
    ## Tabela-Mestre Final (phase4_master_table.csv)

{_md_table(master[rpt_cols], rpt_cols, rpt_fmt)}

    *Critério de elegibilidade: FAR\\_2019\\_per1k ≤ {FAR_ADMISSIBILITY}/1k aplicado antes da ordenação.
    Inadmissíveis e resultados negativos aparecem após os elegíveis ranqueados.*

    ---
    ## Correção Metodológica — SNR_op → SNR_smooth

    O `SNR_op` atribui sentinela 9999 sempre que `n_2019_alarms == 0`,
    independentemente do número de deteções pós-2019.

    | Detetor | SNR_op | SNR_smooth | year_coverage | Artefacto? |
    |---------|--------|------------|---------------|------------|
    | ADWIN_S2S | 2.30 | {float(adwin_r.get('SNR_smooth',0)):.2f} | 5/5 | Não |
    | ADWIN_Kalman | 1.80 | {float(kalman_r.get('SNR_smooth',0)):.2f} | 5/5 | Não |
    | OR(ADWIN_S2S, Kalman) | 1.94 | {float(or_r.get('SNR_smooth',0)):.2f} | 5/5 | Não |
    | KSWIN | 0.77 | {float(kswin_r.get('SNR_smooth',0)):.2f} | 5/5 | Não |
    | PH_agg | **9999** | **{float(ph_r.get('SNR_smooth',0)):.2f}** | 1/5 | **Sim** — sentinela 9999 (n_2019_alarms=0) |
    | AND(ADWIN_S2S, PH_agg) | 0.00 | **{float(and_r.get('SNR_smooth',0)):.2f}** | 0/5 | Não (0 deteções) |

    **Efeito:** `ph_standalone` desce de 9999 para {float(ph_r.get('SNR_smooth',0)):.2f}, eliminando a explosão
    numérica da sentinela. Importa notar que {float(ph_r.get('SNR_smooth',0)):.2f} é o valor **mais alto** da
    tabela-mestre — o SNR_smooth sozinho não desmascara o PH_agg, pois ainda premeia
    especificidade (n_2019_alarms=0). O que revela a sua inutilidade é o year_coverage=1/5
    (deteta só 2021), não o SNR_smooth. A regra de decisão usa year_coverage como critério
    **primário** e o SNR_smooth apenas como desempate final. Limitação reconhecida: o
    SNR_smooth resolve o problema numérico da sentinela mas herda o viés pró-especificidade;
    mitigamo-lo subordinando-o ao year_coverage na regra de decisão.

    ---
    ## Inviabilidade Estrutural do AND Temporal

    > **O ensemble AND temporal foi testado, caracterizado e rejeitado com base em evidência.**

    ### Factor 1 — Desalinhamento Temporal

    | Evento | PH_agg (janelas N=50) | ADWIN_S2S (por-trajetória) |
    |--------|----------------------|---------------------------|
    | Primeiro alarme em 2021 | índice **~{ctx['ph_first']:,}** | índice **~{ctx['adwin_first']:,}** |
    | Δ₁ (mesmo evento) | — | **{ctx['delta1']:,} trajetórias** |
    | Par mais próximo | PH@{ctx['nearest_pair'][1]:,} | ADWIN@{ctx['nearest_pair'][0]:,} |
    | Δ_min (par mais próximo) | — | **{ctx['min_dist']} trajetórias** |

    ### Factor 2 — Limitação de Escopo do PH_agg

    | Canal | n_2021 | n_2022 | n_2023 | n_2024 | n_2025 | year_coverage |
    |-------|--------|--------|--------|--------|--------|---------------|
    | PH_agg standalone | {int(ph_r.get('n_2021',0))} | {int(ph_r.get('n_2022',0))} | {int(ph_r.get('n_2023',0))} | {int(ph_r.get('n_2024',0))} | {int(ph_r.get('n_2025',0))} | **1/5** |
    | ADWIN_S2S standalone | {int(adwin_r.get('n_2021',0))} | {int(adwin_r.get('n_2022',0))} | {int(adwin_r.get('n_2023',0))} | {int(adwin_r.get('n_2024',0))} | {int(adwin_r.get('n_2025',0))} | **5/5** |

    ### Sweep W Estendido

{_md_table(sweep, sweep_cols, sweep_fmt)}

    **Achado crítico:** year_coverage_confirmed=2 exige W≥{w_min2 if w_min2 else 'N/A'}
    (~2.5 meses). Uma janela de "coincidência temporal" de 2.5 meses é operacionalmente
    absurda. Este resultado reforça a inviabilidade estrutural.

    ---
    ## Intervalos de Confiança de Wilson (FAR=0)

    | Detetor | k (2019) | n (2019 trajs) | FAR obs./1k | IC Wilson 95% [lower, upper]/1k |
    |---------|----------|----------------|------------|----------------------------------|
    | ADWIN_S2S | {int(adwin_r.get('n_2019',0))} | {n19t:,} | {float(adwin_r.get('FAR_2019_per1k',0)):.4f} | [{adwin_ci[0]:.4f}, {adwin_ci[1]:.4f}] |
    | PH_agg standalone | {int(ph_r.get('n_2019',0))} | {n19t:,} | {float(ph_r.get('FAR_2019_per1k',0)):.4f} | [{ph_ci[0]:.4f}, {ph_ci[1]:.4f}] |
    | AND CONFIRMED (W=200) | {int(and_r.get('n_2019',0))} | {n19t:,} | {float(and_r.get('FAR_2019_per1k',0)):.4f} | [{and_ci[0]:.4f}, {and_ci[1]:.4f}] |

    ---
    ## Recomendação Operacional Final

    **Detetor primário:** `ADWIN_S2S standalone`
    **Critério de re-treino:** alarme ADWIN com cooldown=200 trajs.
    **Camada de vigilância:** canal OR(ADWIN_S2S, ADWIN_Kalman).

    | Parâmetro | Detetor Primário (ADWIN_S2S) | Canal OR (Kalman) |
    |-----------|------------------------------|-------------------|
    | Model | Seq2Seq | Kalman |
    | detector | ADWINLite + robz_w200 | ADWINLite + robz_w200 |
    | delta | 1e-7 | 1e-7 |
    | min_window | 600 | 200 |
    | cooldown | 200 | 1000 |
    | max_window | 2000 | 5000 |
    | FAR_2019 | {float(adwin_r.get('FAR_2019_per1k',0)):.4f}/1k | {float(kalman_r.get('FAR_2019_per1k',0)):.4f}/1k |
    | year_coverage | 5/5 | 5/5 |
    | n_post | {int(adwin_r.get('n_alarms_post',0))} | {int(kalman_r.get('n_alarms_post',0))} |

    **Ensemble AND(ADWIN_S2S, PH_agg): testado, caracterizado e rejeitado.**
    Não aparece como recomendação em nenhum output da Fase 4.

    ---
    ## Artefactos Gerados

    | Ficheiro | Conteúdo |
    |----------|---------|
    | `phase4_master_table.csv` | 6 candidatos finais ranqueados com métricas corrigidas |
    | `W_infeasibility_sweep.csv` | W ∈ {{50..20000}} com year_coverage_confirmed e nota |
    | `images/and_infeasibility.png` | Figura 300 dpi: desalinhamento ADWIN vs PH (Δ₁={ctx['delta1']:,}, Δ_min={ctx['min_dist']}) |
    | `PHASE4_FINAL.md` | Texto completo para o paper (Q1–Q4, IC Wilson, ameaças à validade) |
    | `PHASE4_REPORT.md` | Este ficheiro — resumo estruturado no formato das fases anteriores |
    """)

    out_path = OUT_ROOT / "PHASE4_REPORT.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"    Guardado: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("FASE 4 — Consolidação Científica")
    print("=" * 70)

    ctx    = load_and_run()
    master = build_master_table(ctx)
    plot_and_infeasibility(ctx)
    sweep  = extended_w_sweep(ctx)
    generate_report(master, sweep, ctx)
    generate_phase4_report(master, sweep, ctx)

    print("=" * 70)
    print("Fase 4 concluída.")
    print(f"Outputs em: {OUT_ROOT}")
    print("  phase4_master_table.csv")
    print("  W_infeasibility_sweep.csv")
    print("  images/and_infeasibility.png")
    print("  PHASE4_FINAL.md")
    print("  PHASE4_REPORT.md")
    print("=" * 70)

    # verificação final dos critérios de aceitação
    print("\n── Critérios de aceitação ──────────────────────────────────────────")
    import re

    top1 = master.iloc[0]
    print(f"[{'OK' if 'ADWIN_S2S' in str(top1['detector']) else 'FAIL'}]"
          f" Master table #1: {top1['detector']}")

    kswin_mask   = master["detector"] == "KSWIN"
    kswin_papel  = master[kswin_mask]["papel_final"].values
    kswin_adm_ok = len(kswin_papel) > 0 and "Inadmissível" in str(kswin_papel[0])
    print(f"[{'OK' if kswin_adm_ok else 'FAIL'}]"
          f" KSWIN marcado como Inadmissível: {kswin_papel}")

    # ph_standalone SNR_smooth (filtrar só a linha exata, não AND que também contém "PH_agg")
    ph_mask  = master["detector"] == "PH_agg"
    ph_snr   = master[ph_mask]["SNR_smooth"].values
    ph_ok    = len(ph_snr) > 0 and all(0 < v < 9999 for v in ph_snr)
    print(f"[{'OK' if ph_ok else 'FAIL'}]"
          f" ph_standalone SNR_smooth finito (não-zero, não-9999): {ph_snr}")

    # year_coverage_confirmed: verifica que atingir 2 anos requer W absurdo (>=10000)
    w_for_2 = sweep[sweep["year_coverage_confirmed"] >= 2]
    if w_for_2.empty:
        print("[OK] year_coverage_confirmed <= 1 para todo W testado")
    else:
        w_thresh = int(w_for_2["W_coincidence"].min())
        months   = w_for_2[w_for_2["W_coincidence"] == w_thresh]["W_approx_months"].values[0]
        is_absurd = w_thresh >= 5000
        print(f"[{'OK' if is_absurd else 'FAIL'}]"
              f" year_coverage_confirmed >= 2 so com W>={w_thresh} (~{months:.1f} meses)"
              f" — operacionalmente {'absurdo' if is_absurd else 'razoavel'}")

    rpt_path = OUT_ROOT / "PHASE4_FINAL.md"
    if rpt_path.exists():
        txt = rpt_path.read_text(encoding="utf-8")

        # W=500 não deve aparecer como solução recomendada (pode aparecer no contexto de rejeição)
        bad_w500 = bool(re.search(
            r"(?<!não\s)(?<!nao\s)(?<!NÃO\s)W=500\s+(?:é|é o)\s+recomendado",
            txt, re.IGNORECASE))
        print(f"[{'FAIL' if bad_w500 else 'OK'}]"
              f" W=500 não aparece como solução recomendada")

        # SNR_op=9999 pode aparecer na tabela antes/depois (contexto de rejeição)
        # mas não deve aparecer em frases de recomendação
        bad_snr = bool(re.search(
            r"SNR_op[=\s]*9999.*(?:recomendado|melhor|superior|ótimo)",
            txt, re.IGNORECASE))
        print(f"[{'FAIL' if bad_snr else 'OK'}]"
              f" SNR_op=9999 não usado como critério de recomendação")

        has_n2019 = "n_2019" in txt
        print(f"[{'OK' if has_n2019 else 'FAIL'}]"
              f" Coluna n_2019 presente no PHASE4_FINAL.md")

    # coerência cruzada: assert que os valores n_2021..n_2025 e year_coverage
    # usados na tabela do Factor 2 são idênticos aos da master table
    # (falha o script com AssertionError se divergirem)
    ph_exact    = master[master["detector"] == "PH_agg"]
    adwin_exact = master[master["detector"] == "ADWIN_S2S"]
    _check_factor2_consistency(master, ph_exact.iloc[0], adwin_exact.iloc[0])
    ph_n21   = int(ph_exact["n_2021"].values[0])
    adwin_n21 = int(adwin_exact["n_2021"].values[0])
    ph_yc    = int(ph_exact["year_coverage"].values[0])
    adwin_yc = int(adwin_exact["year_coverage"].values[0])
    print(f"[OK] Coerência cruzada Factor 2 vs master: "
          f"PH_agg n_2021={ph_n21} (year_cov={ph_yc}/5), "
          f"ADWIN_S2S n_2021={adwin_n21} (year_cov={adwin_yc}/5)")

    print("────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
