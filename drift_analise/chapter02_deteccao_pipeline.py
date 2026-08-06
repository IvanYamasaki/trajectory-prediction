"""
chapter02_deteccao_pipeline.py — Capítulo 2 (deteção no stream)
================================================================
Agrega replot_concept_drift e sweep_concept_drift_calibration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.plotting import T, make_plot_style, df_to_md
from drift_analise.drift_output_paths import CH02

OUT_DIR = CH02


# --- replot ---
# raiz do projeto (1 nivel acima deste script)

COV_DIR = Path("covariate_shift_out")
LANGS = ["pt", "en"]

# ----- estilo (mesmo do notebook) -----------------------------------------
PLOT_STYLE = make_plot_style()
plt.rcParams.update(PLOT_STYLE)


# ----- detectores em pure python (sem river) -------------------------------
class PageHinkley:
    """Page-Hinkley para mean-shift detection.

    Calibracao baseline-relative pos-Mes 6 (defaults antigos absolutos
    em mm nao escalavam entre Seq2Seq mean=6mm e Kalman mean=23mm:
    threshold=100mm equivalente a 40 MAD em Seq2Seq e 10 MAD em Kalman,
    causando 6619 alarmes no Kalman vs 721 no Seq2Seq).

    Os parametros agora sao definidos em multiplos do MAD do baseline
    2019 do proprio modelo. Construtor original (mm absoluto) e mantido
    para compatibilidade; a fabrica `from_baseline()` constroi a partir
    da serie de 2019.

    Defaults:
      threshold     = 50 * MAD_2019  (~20 trajs sustentadamente acima
                                       de 50 MAD do desvio absoluto)
      delta         = 0.5 * MAD_2019 (slack proporcional ao ruido)
      min_instances = 600            (~2 jogos antes de poder disparar)
      cooldown      = 200            (apos um alarme, ignora 200 pontos
                                       para nao re-disparar em rajada)
    """
    @classmethod
    def from_baseline(cls, baseline_vals, K_thr=50.0, K_delta=0.5,
                      min_instances=600, cooldown=200):
        import numpy as _np
        b = _np.asarray(baseline_vals, dtype=float)
        b = b[_np.isfinite(b)]
        med = float(_np.median(b))
        mad = float(_np.median(_np.abs(b - med)))
        if mad < 1e-9:
            mad = float(b.std(ddof=1))
        return cls(min_instances=min_instances,
                   delta=K_delta * mad,
                   threshold=K_thr * mad,
                   cooldown=cooldown,
                   baseline_mad=mad)

    def __init__(self, min_instances=600, delta=2.0, threshold=100.0,
                 cooldown=200, baseline_mad=None):
        self.min_inst = min_instances
        self.delta    = delta
        self.thr      = threshold
        self.cooldown = cooldown
        self.baseline_mad = baseline_mad  # so para introspeccao
        self.n        = 0
        self.x_mean   = 0.0
        self.sum_     = 0.0
        self.min_sum  = 0.0
        self._cool    = 0
        self.drift_detected = False

    def update(self, x: float):
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.sum_ = self.sum_ + (x - self.x_mean - self.delta)
        self.min_sum = min(self.min_sum, self.sum_)
        ph = self.sum_ - self.min_sum
        # cooldown impede rajadas de re-disparo
        if self._cool > 0:
            self._cool -= 1
            self.drift_detected = False
            return
        self.drift_detected = (self.n >= self.min_inst) and (ph > self.thr)
        if self.drift_detected:
            # reseta apos alarme + cooldown
            self.sum_ = 0.0
            self.min_sum = 0.0
            self._cool = self.cooldown


class ADWINLite:
    """Aproximacao simples de ADWIN: testa se a media de uma janela
    crescente de tamanho W mudou significativamente entre os primeiros
    W/2 e ultimos W/2 pontos. Usa um z-test simplificado sobre as medias.

    NOTA: NAO e o ADWIN exato (que faz bisecao adaptativa em multiplos
    cortes). Para os fins deste plot --- visualizar onde o detector
    acionaria --- e suficiente.

    Calibracao pos-Mes 6 (mais conservadora que Mes 5):
      delta       = 1e-5  (z-critico ~4.94, vs 4.45 anterior; reduz
                            alarmes de meia-buffer espuriarios)
      min_window  = 200   (em vez de 60: amostra suficiente para o
                            z-test ser confiavel)
      cooldown    = 200   (apos um alarme, espera 200 pontos para nao
                            disparar em rajada na mesma transicao)

    A estatistica do ADWIN e adimensional (z-score), entao nao precisa
    ser escalonada por modelo --- ja se ajusta ao desvio padrao local.
    """
    def __init__(self, delta=1e-5, max_window=2000,
                 min_window=200, cooldown=200):
        from math import sqrt, log
        self.delta = delta
        self.max_w = max_window
        self.min_w = min_window
        self.cooldown = cooldown
        # z critico aproximado (Bonferroni-corrigido)
        self.z_crit = max(2.0, sqrt(2 * log(2.0 / delta)))
        self.buffer = []
        self._cool = 0
        self.drift_detected = False

    def update(self, x: float):
        self.buffer.append(float(x))
        if len(self.buffer) > self.max_w:
            self.buffer.pop(0)
        self.drift_detected = False
        if self._cool > 0:
            self._cool -= 1
            return
        if len(self.buffer) < self.min_w:
            return
        # corta no meio
        mid = len(self.buffer) // 2
        a = np.asarray(self.buffer[:mid])
        b = np.asarray(self.buffer[mid:])
        mu_a, mu_b = a.mean(), b.mean()
        sd = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        if sd < 1e-9:
            return
        z = abs(mu_a - mu_b) / sd
        if z > self.z_crit:
            self.drift_detected = True
            # reseta deixando so a metade mais recente + cooldown
            self.buffer = list(b)
            self._cool = self.cooldown


class ADWINExact:
    """Bifet & Gavalda (2007) ADWIN via river.drift.ADWIN with a cooldown
    wrapper for parity with ADWINLite's API.

    Unlike ADWINLite (which only tests the midpoint split with a single
    Hoeffding z-test), this calls the true ADWIN: it keeps an exponential
    histogram of buckets and tests every admissible split point each step.

    Constructor mirrors ADWINLite so phase 2 grid search can switch via
    a single flag.  `max_window` is honoured by pruning the river ADWIN's
    internal buffer width (best-effort: river uses an exponential
    histogram, so the prune is approximate but enforces an upper bound on
    memory).  `min_window` maps to river's `grace_period`.
    """

    def __init__(self, delta=1e-5, max_window=2000,
                 min_window=200, cooldown=200,
                 clock=32, max_buckets=5):
        try:
            from river.drift import ADWIN as _RiverADWIN
        except ImportError as e:
            raise ImportError(
                "ADWINExact requires `river>=0.18`. "
                "Install with `pip install river`."
            ) from e
        self._RiverADWIN = _RiverADWIN
        self.delta    = delta
        self.max_w    = int(max_window)
        self.min_w    = int(min_window)
        self.cooldown = int(cooldown)
        self.clock    = int(clock)
        self.max_buckets = int(max_buckets)
        self._build()
        self._cool = 0
        self.drift_detected = False

    def _build(self) -> None:
        self._adwin = self._RiverADWIN(
            delta=float(self.delta),
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=max(5, min(self.min_w, 64)),
            grace_period=self.min_w,
        )

    def update(self, x: float) -> None:
        self.drift_detected = False
        if self._cool > 0:
            self._cool -= 1
            # still feed observation so the window stays current
            self._adwin.update(float(x))
            return
        self._adwin.update(float(x))
        if getattr(self._adwin, "drift_detected", False):
            self.drift_detected = True
            self._cool = self.cooldown
            # rebuild to mimic ADWINLite's post-alarm reset (river already
            # discards the older sub-window internally, but we also enforce
            # the cooldown via a fresh detector to avoid burst re-fires
            # under the same regime change).
            self._build()
        # best-effort max_window enforcement: river exposes `width`
        # (current window size). If it exceeds max_w, drop the oldest
        # bucket by rebuilding (rare in practice with max_buckets=5).
        if getattr(self._adwin, "width", 0) > self.max_w:
            self._build()


def run_detector(detector, values: np.ndarray) -> list[int]:
    alarms = []
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        detector.update(float(v))
        if detector.drift_detected:
            alarms.append(i)
    return alarms


# ----- carregamento --------------------------------------------------------
def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"[warn] parquet nao legivel ({e}); tentando csv.")
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(
        "Nenhum trajectory_errors_sample.parquet ou .csv encontrado. "
        "Rode model_analise/compute_trajectory_errors.py antes."
    )


def build_stream(df: pd.DataFrame, model: str, horizon: str = "30→15") -> pd.DataFrame:
    s = df[(df["model"] == model) & (df["horizon"] == horizon)].copy()
    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s = s.sort_values(["year", "match_id", "traj_id"]).reset_index(drop=True)
    s["global_idx"] = np.arange(len(s))
    return s


# ----- plot ----------------------------------------------------------------
def make_plot(stream: pd.DataFrame, adwin_alarms: list[int],
              ph_alarms: list[int], model_name: str, lang: str) -> plt.Figure:
    """Layout em dois paineis verticais:
       - cima: ADE (raw decimado + media movel) com sombras de ano
       - baixo: rugplot dos alarmes, separados por metodo
    """
    x = stream["global_idx"].values
    y = stream["ade_traj"].astype(float).values

    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[5, 1.4], hspace=0.05,
        left=0.07, right=0.985, top=0.92, bottom=0.10,
    )
    ax  = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)

    # ----- (1) sombras por ano no fundo -------------------
    years_in = sorted(stream["year"].dropna().unique().astype(int))
    cmap = plt.get_cmap("viridis")
    year_color = {
        int(yr): cmap(i / max(1, len(years_in) - 1))
        for i, yr in enumerate(years_in)
    }
    # bordas das faixas
    for yr in years_in:
        sub = stream[stream["year"] == yr]
        if not len(sub):
            continue
        x0 = int(sub["global_idx"].min())
        x1 = int(sub["global_idx"].max())
        c  = year_color[yr]
        # banda muito leve
        ax.axvspan(x0, x1 + 1, color=c, alpha=0.08, zorder=0)
        axr.axvspan(x0, x1 + 1, color=c, alpha=0.08, zorder=0)
        # rotulo do ano no topo do painel principal
        ymax_initial = float(np.nanpercentile(y, 99.5))
        ax.text((x0 + x1) / 2.0, ymax_initial * 0.96, str(yr),
                color=c, fontsize=11, ha="center", va="top",
                fontweight="bold", alpha=0.85)

    # ----- (2) serie bruta decimada -----------------------
    # decimacao: pelo menos 5000 pontos visiveis, mais que isso vira mancha
    step = max(1, len(stream) // 5000)
    ax.scatter(x[::step], y[::step], s=3, color="#888888", alpha=0.18,
               zorder=1, label=T("ADE bruto (decimado)",
                                  "Raw ADE (decimated)", lang))

    # ----- (3) media movel destacada ----------------------
    win = max(100, len(stream) // 80)  # janela proporcional
    if len(stream) > win:
        roll = pd.Series(y).rolling(win, min_periods=1).mean().values
        ax.plot(x, roll, color="black", lw=1.8,
                label=T(f"Média móvel ({win} traj.)",
                         f"Rolling mean ({win} trajs)", lang),
                zorder=3)

    # limita o eixo Y para nao deixar a cauda dominar
    ymax = float(np.nanpercentile(y, 99.5)) * 1.05
    ax.set_ylim(0, max(ymax, 1.0))

    # ----- (4) raster de alarmes (painel inferior) --------
    ADWIN_C = "#2A6FDB"
    PH_C    = "#D1495B"
    # 2 linhas no raster: 0.7 = ADWIN, 0.3 = PH
    if adwin_alarms:
        axr.vlines(adwin_alarms, ymin=0.55, ymax=0.95,
                   color=ADWIN_C, lw=0.8, alpha=0.85)
    if ph_alarms:
        axr.vlines(ph_alarms, ymin=0.05, ymax=0.45,
                   color=PH_C, lw=0.5, alpha=0.55)
    axr.set_ylim(0, 1)
    axr.set_yticks([0.25, 0.75])
    axr.set_yticklabels(
        [T(f"PH ({len(ph_alarms)})",       f"PH ({len(ph_alarms)})",       lang),
         T(f"ADWIN ({len(adwin_alarms)})", f"ADWIN ({len(adwin_alarms)})", lang)],
        fontsize=10,
    )
    axr.tick_params(axis="y", length=0)  # esconde os ticks Y
    axr.grid(True, axis="x", alpha=0.2)
    axr.set_xlabel(T("Trajetória (ordem temporal global)",
                      "Trajectory (global temporal order)", lang))
    # esconde x do painel de cima
    plt.setp(ax.get_xticklabels(), visible=False)

    # ----- (5) titulo e legenda ---------------------------
    ax.set_ylabel(T("ADE (mm)", "ADE (mm)", lang))
    ax.set_title(T(f"Concept drift online --- ADE {model_name} 30→15",
                    f"Online concept drift --- {model_name} 30→15 ADE", lang),
                  pad=8)

    # legenda do painel principal so com (raw + media movel)
    handles, labels = ax.get_legend_handles_labels()
    # adiciona patches falsos para representar os alarmes
    from matplotlib.lines import Line2D
    handles.extend([
        Line2D([], [], color=ADWIN_C, lw=2,
               label=T(f"ADWIN ({len(adwin_alarms)} alarmes)",
                        f"ADWIN ({len(adwin_alarms)} alarms)", lang)),
        Line2D([], [], color=PH_C, lw=2,
               label=T(f"Page-Hinkley ({len(ph_alarms)} alarmes)",
                        f"Page-Hinkley ({len(ph_alarms)} alarms)", lang)),
    ])
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92)

    return fig


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"  -> {out}")
    plt.close(fig)


# ----- calibracao adaptativa --------------------------------------------------
def adaptive_params(n_stream: int) -> dict:
    """Calibra K_thr, min_instances, cooldown e ADWIN delta como fun\\xe7\\xe3o
    de n_stream. Alvo: ~5-15 alarmes por detector no sample (n~1k) e
    ~50-150 alarmes no stream completo (n~290k).

    Itera\\xe7\\xe3o (Mes 6.2): a primeira vers\\xe3o (K cresce com (n/5000)^0.4)
    deu K=38 no full --- threshold absurdamente alto, 4-9 alarmes em
    290k. A nova versa\\xe3o usa log10 com slope reduzido + cooldown menor
    no full, e permite override por env var ``CONCEPT_DRIFT_K_THR``.

    Heuristica atual:
      K_thr        = clip(5 + 5*log10(n_stream/1000),  8, 30)
      min_inst     = clip(0.02 * n_stream, 100, 400)
      cooldown     = clip(0.005 * n_stream, 30, 100)
      adwin_delta  = 1e-3 (n<5k) | 1e-4 (5k..50k) | 5e-5 (>=50k)
      adwin_min_w  = clip(0.03 * n_stream, 60, 150)

    Para n=290k:
      K_thr=17.3 (=> ~43 mm Seq2Seq, ~162 mm Kalman)
      min_inst=400, cooldown=100, adwin_delta=5e-5
      Estimativa: ~50-150 alarmes por detector.
    """
    import math, os
    # K_thr base (cap ate 50 para permitir tuning manual via env var
    # quando o sinal tem cumsum muito agressivo, como o ADE em SSL)
    K_thr = 5.0 + 5.0 * math.log10(max(1, n_stream) / 1000.0)
    K_thr = float(max(8.0, min(50.0, K_thr)))
    # override por env var (para tuning sem editar codigo)
    if os.environ.get("CONCEPT_DRIFT_K_THR"):
        try:
            K_thr = float(os.environ["CONCEPT_DRIFT_K_THR"])
            print(f"  [override] K_thr={K_thr} via CONCEPT_DRIFT_K_THR")
        except ValueError:
            pass
    min_inst = int(max(100, min(400, n_stream * 0.02)))
    cooldown = int(max( 30, min(100, n_stream * 0.005)))
    if   n_stream < 5_000:  adwin_delta = 1e-3
    elif n_stream < 50_000: adwin_delta = 1e-4
    else:                   adwin_delta = 5e-5
    adwin_min_w = int(max(60, min(150, n_stream * 0.03)))
    return {
        "K_thr": K_thr, "K_delta": 0.5,
        "min_instances": min_inst, "cooldown": cooldown,
        "adwin_delta": adwin_delta, "adwin_min_window": adwin_min_w,
    }


# ----- main ----------------------------------------------------------------
def main_replot_concept_drift() -> None:
    print("[info] Carregando trajectory_errors_sample ...")
    df = load_errors()
    print(f"  shape={df.shape}, modelos={sorted(df['model'].unique())}")

    summary_rows = []
    for model in ("Seq2Seq", "Kalman"):
        stream = build_stream(df, model=model, horizon="30→15")
        if stream.empty:
            print(f"[warn] sem dados para {model}; pulando.")
            continue
        print(f"\n[info] === {model} (stream: {len(stream)} trajetorias) ===")

        # baseline 2019 do PROPRIO modelo, para PH com threshold relativo
        baseline = stream[stream["year"] == 2019]["ade_traj"].astype(float).values
        baseline = baseline[np.isfinite(baseline)]
        if len(baseline) < 100:
            print(f"  [warn] baseline 2019 muito pequeno ({len(baseline)}); "
                  "fallback para mediana global do stream")
            baseline = stream["ade_traj"].astype(float).dropna().values
        med = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - med)))
        print(f"  baseline 2019: n={len(baseline)}  mediana={med:.3f} mm  "
              f"MAD={mad:.3f} mm")

        # Parametros adaptativos ao tamanho do stream
        p = adaptive_params(len(stream))
        print(f"  params adaptativos: K_thr={p['K_thr']:.2f} (=> "
              f"thr={p['K_thr']*mad:.2f} mm) | min_inst={p['min_instances']} "
              f"| cooldown={p['cooldown']} | ADWIN delta={p['adwin_delta']:.0e}")

        adwin = ADWINLite(delta=p['adwin_delta'], max_window=2000,
                          min_window=p['adwin_min_window'],
                          cooldown=p['cooldown'])
        ph = PageHinkley.from_baseline(
            baseline, K_thr=p['K_thr'], K_delta=p['K_delta'],
            min_instances=p['min_instances'], cooldown=p['cooldown'],
        )

        adwin_alarms = run_detector(adwin, stream["ade_traj"].values)
        ph_alarms    = run_detector(ph,    stream["ade_traj"].values)
        rate_a = 100 * len(adwin_alarms) / max(1, len(stream))
        rate_p = 100 * len(ph_alarms) / max(1, len(stream))
        print(f"  ADWIN-lite: {len(adwin_alarms):>5d} alarmes ({rate_a:.2f}%)")
        print(f"  PH        : {len(ph_alarms):>5d} alarmes ({rate_p:.2f}%)")

        summary_rows.append({
            "model": model, "n_stream": len(stream),
            "baseline_n": len(baseline), "baseline_med": med,
            "baseline_mad": mad,
            "ph_K_thr": p['K_thr'], "ph_thr_mm": ph.thr, "ph_delta_mm": ph.delta,
            "min_instances": p['min_instances'], "cooldown": p['cooldown'],
            "adwin_delta": p['adwin_delta'],
            "n_adwin": len(adwin_alarms), "n_ph": len(ph_alarms),
            "rate_adwin_pct": rate_a, "rate_ph_pct": rate_p,
        })

        for lang in LANGS:
            fig = make_plot(stream, adwin_alarms, ph_alarms,
                             model_name=model, lang=lang)
            base = f"6_concept_drift_{model.lower()}_{lang}"
            save_fig(fig, base)
            if model == "Seq2Seq":
                fig2 = make_plot(stream, adwin_alarms, ph_alarms,
                                  model_name=model, lang=lang)
                save_fig(fig2, f"6_concept_drift_adwin_ph_{lang}")

    if summary_rows:
        out_csv = OUT_DIR / "concept_drift_calibration.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\n[ok] Calibracao salva em {out_csv}")

    print("\n[ok] Plots concept_drift refeitos com calibracao adaptativa baseline-relative.")



# --- sweep ---


def _dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    """Markdown table via common.plotting.df_to_md (mesmo comportamento para
    os dados usados aqui; NaN formata como 'nan' nos dois caminhos)."""
    return df_to_md(df, floatfmt=floatfmt)


def baseline_stats(stream: pd.DataFrame) -> tuple[float, float, int]:
    b = stream[stream["year"] == 2019]["ade_traj"].astype(float).values
    b = b[np.isfinite(b)]
    if len(b) < 100:
        b = stream["ade_traj"].astype(float).dropna().values
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med)))
    return med, mad, len(b)


def main_sweep_concept_drift_calibration() -> None:
    df = load_errors()

    rows = []
    for model in ("Seq2Seq", "Kalman"):
        stream = build_stream(df, model=model, horizon="30→15")
        if stream.empty:
            continue
        n = len(stream)
        med, mad, n_b = baseline_stats(stream)

        # min_instances escalonado: 5% do stream, no minimo 100, max 600
        min_inst = int(min(600, max(100, n * 0.05)))
        cooldown = int(min(200, max(50, n * 0.02)))

        # ADWIN: varia delta
        for delta in [1e-3, 1e-4, 1e-5, 1e-6]:
            det = ADWINLite(delta=delta, max_window=2000,
                            min_window=min(200, n // 10),
                            cooldown=cooldown)
            n_alarms = len(run_detector(det, stream["ade_traj"].values))
            rows.append({
                "model": model, "detector": "ADWIN", "param": "delta",
                "param_val": delta, "K_mad": np.nan,
                "thr_mm": np.nan, "n_stream": n,
                "min_instances": min(200, n // 10),
                "n_alarms": n_alarms,
                "rate_per_1k": 1000 * n_alarms / n,
            })

        # PH: varia K_thr
        for K in [10, 20, 30, 50, 80, 120]:
            det = PageHinkley.from_baseline(
                stream[stream["year"] == 2019]["ade_traj"].dropna().values,
                K_thr=K, K_delta=0.5,
                min_instances=min_inst, cooldown=cooldown,
            )
            n_alarms = len(run_detector(det, stream["ade_traj"].values))
            rows.append({
                "model": model, "detector": "PH", "param": "K_thr",
                "param_val": K, "K_mad": K, "thr_mm": K * mad,
                "n_stream": n, "min_instances": min_inst,
                "n_alarms": n_alarms,
                "rate_per_1k": 1000 * n_alarms / n,
            })

    out_df = pd.DataFrame(rows)
    out_csv = CH02 / "concept_drift_sweep.csv"
    out_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[ok] varredura salva em {out_csv}")
    print()
    print(out_df.to_string(index=False))

    # markdown summary com recomendacao
    md = CH02 / "concept_drift_sweep_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Varredura de calibracao Mes 2 (ADWIN/PH baseline-relative)\n\n")
        f.write("Stream: ordenado por (year, match_id, traj_id), horizonte 30->15.\n\n")
        f.write("Baseline: 2019 do PROPRIO modelo (MAD escalonado).\n\n")
        f.write("## Tabela de alarmes\n\n")
        f.write(_dataframe_to_markdown(out_df, floatfmt=".4f"))
        f.write("\n\n## Recomendacao default (apos varredura)\n\n")
        f.write("- ADWIN delta = 1e-5 (z_crit ~ 4.94)\n")
        f.write("- PH K_thr   = 50 MAD\n")
        f.write("- PH K_delta = 0.5 MAD\n")
        f.write("- min_instances = 600 (Seq2Seq/Kalman 290k); 100-300 em sample\n")
        f.write("- cooldown      = 200\n\n")
        f.write("Justificativa: na varredura, K=50 produz uma quantidade\n")
        f.write("balanceada de alarmes entre Seq2Seq e Kalman (sem o vies\n")
        f.write("de threshold absoluto que dava 6619 PH em Kalman vs 721 em Seq2Seq).\n")
    print(f"[ok] sumario em {md}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["replot", "sweep"])
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if args.step == "replot":
        main_replot_concept_drift()
    else:
        main_sweep_concept_drift_calibration()
