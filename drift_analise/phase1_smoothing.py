"""
drift_analise/phase1_smoothing.py
==================================
Fase 1 do Plano de Ensaios — Pré-processamento e Suavização do Sinal ADE.

Testa 22 combinações (método × parâmetro) de suavização sobre o stream
temporal do Seq2Seq e Kalman. Para cada combinação:
  - Aplica suavização ao sinal raw de ADE por trajetória
  - Corre ADWINLite e Page-Hinkley com calibração baseline-relative (2019)
  - Regista alarmes por ano, FAR_2019, SNR operacional, variância reduzida
  - Grava imagem PNG em images/<config>/

Saídas em Relas/results/drift/mes9_phase1/:
  phase1_results.csv    — tabela completa de métricas por experimento
  PHASE1_REPORT.md      — relatório de análise com ranking e recomendação
  images/<exp>/         — PNGs por experimento (raw + smooth + rasters ADWIN/PH)

Uso:
    python drift_analise/phase1_smoothing.py
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

from common.constants import YEARS_ALL, PERFECT_SNR
from common.detection import smooth_robz, alarms_per_year, compute_snr
from common.plotting import YEAR_COLORS, make_plot_style, df_to_md as _df_to_md
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, PageHinkley, run_detector,
    load_errors, build_stream, adaptive_params,
)

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_ROOT = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes9_phase1"
IMG_DIR  = OUT_ROOT / "images"

PLOT_STYLE = make_plot_style(**{
    "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9,  "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    # o arquivo nao definia mathtext.fontset; preserva o default do matplotlib
    "mathtext.fontset": "dejavusans",
})
plt.rcParams.update(PLOT_STYLE)

# ── suavizadores ──────────────────────────────────────────────────────────────

def smooth_sma(values: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(values).rolling(w, min_periods=1).mean().values

def smooth_med(values: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(values).rolling(w, min_periods=1).median().values

def smooth_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().values

def smooth_holt(values: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Double exponential smoothing (Holt's additive trend)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return v.copy()
    s = np.empty(n, dtype=float)
    b = np.empty(n, dtype=float)
    s[0] = v[0]
    b[0] = (v[1] - v[0]) if n > 1 else 0.0
    for i in range(1, n):
        s[i] = alpha * v[i] + (1 - alpha) * (s[i-1] + b[i-1])
        b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1]
    return s

def smooth_savgol(values: np.ndarray, w: int = 51, poly: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter
    w = min(w, len(values))
    if w % 2 == 0:
        w -= 1
    if w <= poly or w < 3:
        return values.copy()
    return savgol_filter(values.astype(float), w, poly)

# smooth_robz vem de common.detection

# ── catálogo de experimentos ───────────────────────────────────────────────────

def build_experiments() -> list[dict]:
    exps: list[dict] = []

    def add(name: str, fn, label: str, is_dimensionless: bool = False) -> None:
        exps.append(dict(name=name, fn=fn, label=label,
                         is_dimensionless=is_dimensionless))

    # controlo (sem suavização)
    add("ctrl", lambda v: v.copy(), "Sem suavização (controlo)")

    # SMA
    for w in [20, 50, 100, 200, 500]:
        add(f"sma_w{w:03d}",
            (lambda _w: lambda v: smooth_sma(v, _w))(w),
            f"SMA w={w}")

    # Mediana móvel
    for w in [20, 50, 100, 200]:
        add(f"med_w{w:03d}",
            (lambda _w: lambda v: smooth_med(v, _w))(w),
            f"Mediana móvel w={w}")

    # EWMA
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3]:
        tag = f"{alpha:.2f}".replace(".", "")
        add(f"ewma_a{tag}",
            (lambda _a: lambda v: smooth_ewma(v, _a))(alpha),
            f"EWMA α={alpha}")

    # Holt (EWMA dupla com tendência)
    for alpha, beta in [(0.05, 0.01), (0.05, 0.05), (0.1, 0.01), (0.1, 0.05)]:
        ta = f"{alpha:.2f}".replace(".", "")
        tb = f"{beta:.2f}".replace(".", "")
        add(f"holt_a{ta}_b{tb}",
            (lambda _a, _b: lambda v: smooth_holt(v, _a, _b))(alpha, beta),
            f"Holt α={alpha} β={beta}")

    # Savitzky-Golay
    add("savgol_w51_p3",
        lambda v: smooth_savgol(v, 51, 3),
        "Savitzky-Golay w=51 p=3")

    # Robust z-score (sinal adimensional)
    for w in [200, 500]:
        add(f"robz_w{w:03d}",
            (lambda _w: lambda v: smooth_robz(v, _w))(w),
            f"Robust-z w={w}", is_dimensionless=True)

    return exps

# ── métricas ──────────────────────────────────────────────────────────────────
# PERFECT_SNR / alarms_per_year / compute_snr vem de common

def variance_reduction(raw: np.ndarray, smoothed: np.ndarray) -> float:
    var_r = np.nanvar(raw)
    var_s = np.nanvar(smoothed)
    if var_r < 1e-12:
        return 0.0
    return float(100.0 * (1.0 - var_s / var_r))

# ── plot ──────────────────────────────────────────────────────────────────────

def make_plot(stream: pd.DataFrame,
              raw: np.ndarray, smoothed: np.ndarray,
              adwin_alarms: list[int], ph_alarms: list[int],
              exp_label: str, model_name: str,
              is_dimensionless: bool = False) -> plt.Figure:

    x = stream["global_idx"].values
    years_int = stream["year"].fillna(0).astype(int).values

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 1.3, 1.3], hspace=0.06,
                          left=0.07, right=0.985, top=0.91, bottom=0.09)
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax3 = fig.add_subplot(gs[2], sharex=ax)

    # sombras + rótulos de ano
    unique_years = sorted(int(y) for y in stream["year"].dropna().unique()
                          if int(y) != 0)
    for yr in unique_years:
        sub = stream[stream["year"] == yr]
        if sub.empty:
            continue
        x0 = int(sub["global_idx"].min())
        x1 = int(sub["global_idx"].max())
        c = YEAR_COLORS.get(yr, "#888888")
        for panel in (ax, ax2, ax3):
            panel.axvspan(x0, x1 + 1, color=c, alpha=0.07, zorder=0)
        ypos = float(np.nanpercentile(raw[np.isfinite(raw)], 99)) * 0.97
        ax.text((x0 + x1) / 2, ypos, str(yr), color=c, fontsize=10,
                ha="center", va="top", fontweight="bold", alpha=0.92)

    # sinal raw (só para métodos dimensionais)
    if not is_dimensionless:
        step = max(1, len(x) // 6000)
        ax.scatter(x[::step], raw[::step], s=2, color="#aaaaaa", alpha=0.15,
                   zorder=1, rasterized=True, label="ADE raw (decimado)")
        ymax = float(np.nanpercentile(raw[np.isfinite(raw)], 99.5)) * 1.08
        ax.set_ylim(0, max(ymax, 1.0))
        ax.set_ylabel("ADE (mm)")
    else:
        # para z-score: mostra apenas o sinal suavizado
        ax.set_ylabel("ADE (z-score)")
        clipped = np.clip(smoothed, -5, 5)
        ax.set_ylim(-6, 6)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6, zorder=0)

    # sinal suavizado
    smooth_color = "#1c1c8c"
    ax.plot(x, smoothed, color=smooth_color, lw=1.6, zorder=3, label=exp_label)
    ax.set_title(f"{model_name} 30→15 — {exp_label}", pad=7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # raster ADWIN
    ADWIN_C = "#2A6FDB"
    PH_C    = "#D1495B"
    if adwin_alarms:
        ax2.vlines(adwin_alarms, 0.1, 0.9, color=ADWIN_C, lw=0.7, alpha=0.75)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels([f"ADWIN ({len(adwin_alarms)})"], fontsize=9)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", alpha=0.2)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # raster PH
    if ph_alarms:
        ax3.vlines(ph_alarms, 0.1, 0.9, color=PH_C, lw=0.7, alpha=0.75)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.5])
    ax3.set_yticklabels([f"PH ({len(ph_alarms)})"], fontsize=9)
    ax3.tick_params(axis="y", length=0)
    ax3.grid(axis="x", alpha=0.2)
    ax3.set_xlabel("Trajetória (ordem temporal global)")
    plt.setp(ax.get_xticklabels(), visible=False)

    return fig

# ── relatório MD ──────────────────────────────────────────────────────────────
# _df_to_md vem de common.plotting (df_to_md)

def write_report(df: pd.DataFrame, out_dir: Path, total_exps: int) -> None:
    md_path = out_dir / "PHASE1_REPORT.md"
    L: list[str] = []

    # ── cabeçalho ────────────────────────────────────────────────────────────
    L += [
        "# Fase 1 — Relatório de Pré-processamento e Suavização do Sinal ADE",
        "",
        "## Objetivo",
        "Reduzir os **alarmes falsos em 2019** (ano de treino/baseline) sem suprimir",
        "os alarmes nos anos 2021–2025.",
        "",
        "## Métricas principais",
        "| Coluna | Descrição | Direção |",
        "|--------|-----------|---------|",
        "| `FAR_2019_per1k` | Falsos alarmes por 1.000 trajs em 2019 | **↓ melhor** |",
        "| `n_alarms_post` | Total alarmes em 2021–2025 | **↑ melhor** |",
        "| `SNR_op` | (n_post × n_2019_total) / (n_2019_alarms × n_post_total) — razão sinal/ruído | **↑ melhor** (9999=perfeito) |",
        "| `var_reduction_pct` | Redução de variância (%) vs sinal raw | contexto |",
        "| `n_2019..n_2025` | Alarmes por ano | diagnóstico |",
        "",
        "## Critério de seleção para Fase 2",
        "1. FAR\\_2019 ≤ 0.5 / 1.000 trajs",
        "2. n\\_alarms\\_2021 ≥ 50% do controlo (sem suavização)",
        "3. var\\_reduction\\_pct ≥ 10%",
        "4. Maximizar SNR\\_op dentro dos critérios acima",
        "",
        f"**Total de experimentos:** {total_exps} configurações × 2 modelos × 2 detectores",
        "",
    ]

    for model in sorted(df["model"].unique()):
        m = df[df["model"] == model]
        L += ["---", f"## Modelo: {model}", ""]

        for det in ["ADWIN", "PH"]:
            d = m[m["detector"] == det].copy()
            d = d.sort_values("FAR_2019_per1k")

            L += [f"### {det} — ordenado por FAR\\_2019 ↑ (pior → melhor)", ""]

            display_cols = [
                "exp", "label", "var_reduction_pct",
                "FAR_2019_per1k",
                "n_2019", "n_2021", "n_2022", "n_2023", "n_2024", "n_2025",
                "n_alarms_post", "SNR_op", "n_alarms_total",
            ]
            disp = d[[c for c in display_cols if c in d.columns]]
            L.append(_df_to_md(disp))
            L.append("")

            # Referência do controlo (sem suavização)
            ctrl_row = d[d["exp"] == "ctrl"]
            if not ctrl_row.empty:
                ctrl_far  = float(ctrl_row["FAR_2019_per1k"].iloc[0])
                ctrl_post = int(ctrl_row["n_alarms_post"].iloc[0])
                ctrl_2021 = int(ctrl_row.get("n_2021", pd.Series([0])).iloc[0])
                L += [
                    f"> **Controlo (sem suavização):** FAR\\_2019={ctrl_far:.3f}/1k  "
                    f"| n\\_post={ctrl_post}  | n\\_2021={ctrl_2021}",
                    "",
                ]

            # top com FAR=0
            zero_far = d[d["FAR_2019_per1k"] == 0].nlargest(5, "SNR_op")
            if not zero_far.empty:
                L += ["**✅ FAR\\_2019 = 0 (sem nenhum alarme em 2019), ordenado por SNR\\_op:**", ""]
                for _, r in zero_far.iterrows():
                    post = int(r.get("n_alarms_post", 0))
                    y21  = int(r.get("n_2021", 0))
                    L.append(
                        f"- `{r['exp']}` — {r['label']}: "
                        f"SNR\\_op={r['SNR_op']:.2f}  |  "
                        f"n\\_post={post}  |  n\\_2021={y21}  |  "
                        f"var\\_red={r['var_reduction_pct']:.1f}%"
                    )
                L.append("")

            # top com FAR ≤ 0.5
            low_far = d[d["FAR_2019_per1k"] <= 0.5].nlargest(5, "SNR_op")
            if not low_far.empty:
                L += ["**⚠️  FAR\\_2019 ≤ 0.5/1k, ordenado por SNR\\_op:**", ""]
                for _, r in low_far.iterrows():
                    post = int(r.get("n_alarms_post", 0))
                    L.append(
                        f"- `{r['exp']}` — {r['label']}: "
                        f"FAR={r['FAR_2019_per1k']:.3f}/1k  |  "
                        f"SNR\\_op={r['SNR_op']:.2f}  |  "
                        f"n\\_post={post}  |  "
                        f"var\\_red={r['var_reduction_pct']:.1f}%"
                    )
                L.append("")

    # ── ranking global ────────────────────────────────────────────────────────
    L += [
        "---",
        "## Ranking global — Seq2Seq (ambos os detectores, FAR\\_2019 ≤ 0.5)",
        "",
    ]
    seq_ok = df[
        (df["model"] == "Seq2Seq") & (df["FAR_2019_per1k"] <= 0.5)
    ].copy()
    if not seq_ok.empty:
        seq_ok = seq_ok.sort_values(["detector", "SNR_op"], ascending=[True, False])
        rank_cols = ["detector", "exp", "label", "FAR_2019_per1k",
                     "n_alarms_post", "SNR_op", "var_reduction_pct"]
        rank_cols = [c for c in rank_cols if c in seq_ok.columns]
        L.append(_df_to_md(seq_ok[rank_cols].head(20)))
        L.append("")
    else:
        L.append("_Nenhum experimento atingiu FAR\\_2019 ≤ 0.5/1k. Ver tabela completa._")
        L.append("")

    # ── recomendação ──────────────────────────────────────────────────────────
    L += [
        "---",
        "## Recomendação para Fase 2",
        "",
        "Selecionar os **2 melhores candidatos por detector** (menor FAR\\_2019, maior SNR\\_op)",
        "para entrar no Grid Search da Fase 2. O sinal suavizado pelo candidato vencedor",
        "substitui o sinal raw como input para os detectores na Fase 2.",
        "",
        "### Candidatos vencedores (a preencher após análise visual das imagens)",
        "",
        "| Detector | Candidato 1 | Candidato 2 |",
        "|----------|------------|------------|",
        "| ADWIN | `<exp>` | `<exp>` |",
        "| PH    | `<exp>` | `<exp>` |",
        "",
        "### Imagens para análise visual",
        "",
        "Cada experimento tem um PNG em `images/<exp>/`.",
        "Formato: painel superior = ADE raw (cinza) + sinal suavizado (azul escuro);",
        "painéis inferiores = rasters ADWIN e PH com alarmes por detector.",
        "",
    ]

    # ── índice de imagens ─────────────────────────────────────────────────────
    L += ["### Índice de experimentos e imagens", ""]
    exps_seen: set[str] = set()
    for _, row in df[df["model"] == "Seq2Seq"].iterrows():
        exp = row["exp"]
        if exp in exps_seen:
            continue
        exps_seen.add(exp)
        L.append(
            f"- **`{exp}`** — {row['label']}  "
            f"→ `images/{exp}/{exp}_seq2seq.png`"
        )
    L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  [ok] {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_phase1() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("[phase1] Carregando dados ...")
    df = load_errors()
    print(f"  shape={df.shape}, anos={sorted(df['year'].dropna().unique().astype(int))}")

    experiments = build_experiments()
    print(f"[phase1] {len(experiments)} experimentos × 2 modelos × 2 detectores\n")

    all_rows: list[dict] = []

    for model in ("Seq2Seq", "Kalman"):
        stream = build_stream(df, model=model, horizon="30→15")
        if stream.empty:
            print(f"[warn] sem dados para {model}; pulando.")
            continue

        raw_vals  = stream["ade_traj"].astype(float).values
        year_vals = stream["year"].fillna(0).astype(int).values
        n         = len(stream)
        print(f"=== {model} ({n:,} trajetórias) ===")

        # parâmetros adaptativos ao tamanho do stream
        p = adaptive_params(n)

        # baseline raw 2019 para fallback
        baseline_raw = raw_vals[year_vals == 2019]
        baseline_raw = baseline_raw[np.isfinite(baseline_raw)]

        # tamanhos por periodo
        n_2019_total = int((year_vals == 2019).sum())
        n_post_total = int((year_vals >= 2021).sum())

        for exp in experiments:
            name          = exp["name"]
            label         = exp["label"]
            is_dimensionless = exp["is_dimensionless"]
            print(f"  [{model}] {name:<25s}", end="", flush=True)

            # ── suavização ────────────────────────────────────────────────────
            try:
                smoothed = exp["fn"](raw_vals.copy())
            except Exception as e:
                print(f" ERRO ao suavizar: {e}")
                continue

            # baseline do sinal suavizado em 2019 (para calibrar PH)
            b2019_smooth = smoothed[year_vals == 2019]
            b2019_smooth = b2019_smooth[np.isfinite(b2019_smooth)]
            if len(b2019_smooth) < 30:
                b2019_smooth = baseline_raw   # fallback

            # ── detectores ───────────────────────────────────────────────────
            adwin = ADWINLite(
                delta=p["adwin_delta"],
                max_window=2000,
                min_window=p["adwin_min_window"],
                cooldown=p["cooldown"],
            )
            ph = PageHinkley.from_baseline(
                b2019_smooth,
                K_thr=p["K_thr"],
                K_delta=p["K_delta"],
                min_instances=p["min_instances"],
                cooldown=p["cooldown"],
            )

            adwin_alarms = run_detector(adwin, smoothed)
            ph_alarms    = run_detector(ph,    smoothed)

            # ── métricas ─────────────────────────────────────────────────────
            a_by_yr = alarms_per_year(adwin_alarms, year_vals, YEARS_ALL)
            p_by_yr = alarms_per_year(ph_alarms,    year_vals, YEARS_ALL)

            a_far19 = 1000.0 * a_by_yr["n_2019"] / max(1, n_2019_total)
            p_far19 = 1000.0 * p_by_yr["n_2019"] / max(1, n_2019_total)

            a_post  = sum(a_by_yr[f"n_{y}"] for y in [2021, 2022, 2023, 2024, 2025])
            p_post  = sum(p_by_yr[f"n_{y}"] for y in [2021, 2022, 2023, 2024, 2025])

            a_snr = compute_snr(a_post, a_by_yr["n_2019"], n_2019_total, n_post_total)
            p_snr = compute_snr(p_post, p_by_yr["n_2019"], n_2019_total, n_post_total)

            var_red = variance_reduction(
                raw_vals[np.isfinite(raw_vals)],
                smoothed[np.isfinite(smoothed)],
            )

            row_base = dict(
                model=model, exp=name, label=label,
                is_dimensionless=is_dimensionless,
                n_stream=n, var_reduction_pct=round(var_red, 2),
                K_thr_used=round(p["K_thr"], 2),
                adwin_delta=p["adwin_delta"],
                min_instances=p["min_instances"],
                cooldown=p["cooldown"],
            )
            for det, by_yr, far19, n_p, snr, n_tot in [
                ("ADWIN", a_by_yr, a_far19, a_post, a_snr, len(adwin_alarms)),
                ("PH",    p_by_yr, p_far19, p_post, p_snr, len(ph_alarms)),
            ]:
                all_rows.append({
                    **row_base,
                    "detector": det,
                    "n_alarms_total": n_tot,
                    "FAR_2019_per1k": round(far19, 4),
                    "n_alarms_post":  n_p,
                    "SNR_op":         round(min(snr, PERFECT_SNR), 2),
                    **by_yr,
                })

            print(
                f" ADWIN={len(adwin_alarms):>4d} (FAR2019={a_far19:5.2f}/1k)"
                f"  PH={len(ph_alarms):>4d} (FAR2019={p_far19:5.2f}/1k)"
                f"  var_red={var_red:.1f}%"
            )

            # ── plot ──────────────────────────────────────────────────────────
            exp_dir = IMG_DIR / name
            exp_dir.mkdir(exist_ok=True)
            try:
                fig = make_plot(
                    stream, raw_vals, smoothed,
                    adwin_alarms, ph_alarms,
                    label, model, is_dimensionless,
                )
                png_path = exp_dir / f"{name}_{model.lower()}.png"
                fig.savefig(str(png_path), dpi=110, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  [warn] plot falhou para {name}/{model}: {e}")
                plt.close("all")

        print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    results = pd.DataFrame(all_rows)
    csv_path = OUT_ROOT / "phase1_results.csv"
    results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[ok] CSV -> {csv_path}")

    # ── relatório MD ──────────────────────────────────────────────────────────
    write_report(results, OUT_ROOT, total_exps=len(experiments))
    print(f"[ok] Relatorio -> {OUT_ROOT / 'PHASE1_REPORT.md'}")
    print(f"[ok] Imagens   -> {IMG_DIR}/")


if __name__ == "__main__":
    run_phase1()
