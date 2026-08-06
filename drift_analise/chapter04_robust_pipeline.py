"""
chapter04_robust_pipeline.py — Capítulo 4 (robustez e validação)
================================================================
Agrega null_test, latency_curve, calibrate_arl, bocpd_run, pelt_bic_curve,
pelt_gamma_sensitivity, sample_size_sweep e detect_drift_per_game num único
módulo importável pelos notebooks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import t as student_t_dist

from sklearn.metrics.pairwise import rbf_kernel

from drift_analise.cli_argv import parse_args_skip_kernel
from drift_analise.drift_output_paths import CH02, CH04, prepend_matching_venv_site_packages
from drift_analise.notebook_utils import save_figure

prepend_matching_venv_site_packages(PROJECT_ROOT)

DATASET_CSV = PROJECT_ROOT / "drift_analise" / "dataset" / "dataset.csv"

COV_DIR = Path("covariate_shift_out")

# === null_test ===
"""
null_test.py — Teste de permutação para ADWIN/PH (Frente 0, Ação A1+A3)
=======================================================================
Carrega stream ADE por jogo do dataset.csv, roda ADWIN e Page-Hinkley,
gera N permutações e calcula p-value para cada detector.

Uso
---
    python drift_analise/chapter04_robust_pipeline.py null_test
    python drift_analise/chapter04_robust_pipeline.py null_test --debug --n_perms 50 --emit_distinct_check
    python drift_analise/chapter04_robust_pipeline.py null_test --adwin_delta 0.01 --ph_threshold 25.0 \\
        --n_perms 200 --out Relas/results/drift/04_robustez_e_validacao/null_distribution_v2.csv
"""
DEFAULT_OUT  = CH04 / "null_distribution_v2.csv"


def _null_load_stream(model: str, horizon: str) -> np.ndarray:
    df = pd.read_csv(DATASET_CSV)
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"]) if "30" in horizon
        else (df["horizon"] == horizon)
    )
    sub = df[mask].sort_values(["year", "dataset"]).dropna(subset=["value"])
    return sub["value"].astype(float).values


def _null_run_adwin_count(series: np.ndarray, delta: float) -> int:
    from river.drift import ADWIN
    det = ADWIN(delta=delta)
    n = 0
    for v in series:
        det.update(float(v))
        if det.drift_detected:
            n += 1
    return n


def _null_run_ph_count(series: np.ndarray, threshold: float) -> int:
    from river.drift import PageHinkley
    det = PageHinkley(threshold=threshold, min_instances=3)
    n = 0
    for v in series:
        det.update(float(v))
        if det.drift_detected:
            n += 1
    return n


def null_test(series: np.ndarray, detector_fn, n_perms: int,
              rng: np.random.Generator, debug: bool) -> tuple[int, np.ndarray]:
    observed = detector_fn(series)
    null_counts = np.empty(n_perms, dtype=int)
    for i in range(n_perms):
        perm = rng.permutation(series)
        null_counts[i] = detector_fn(perm)
    if debug:
        print(f"    obs={observed}  null[:5]={null_counts[:5].tolist()}  "
              f"null_mean={null_counts.mean():.1f}")
    p_value = float(np.mean(null_counts >= observed))
    return observed, null_counts, p_value


def main_null_test() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adwin_delta",  type=float, default=0.002)
    ap.add_argument("--ph_threshold", type=float, default=5.0)
    ap.add_argument("--n_perms",      type=int,   default=200)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out",          default=str(DEFAULT_OUT))
    ap.add_argument("--debug",        action="store_true")
    ap.add_argument("--emit_distinct_check", action="store_true")
    args = parse_args_skip_kernel(ap)

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ("Seq2Seq", "Kalman"):
        stream = _null_load_stream(model, "30->15")
        print(f"\n[{model}] stream len={len(stream)}, "
              f"mean={stream.mean():.3f}, std={stream.std():.3f}")

        if args.emit_distinct_check and args.debug:
            seqs = [rng.permutation(stream) for _ in range(5)]
            for i in range(4):
                same = np.array_equal(seqs[i], seqs[i+1])
                print(f"  perm[{i}] == perm[{i+1}]: {same}")
            rng = np.random.default_rng(args.seed)  # reset

        for det_name, fn in [
            ("ADWIN", lambda s: _null_run_adwin_count(s, args.adwin_delta)),
            ("PH",    lambda s: _null_run_ph_count(s, args.ph_threshold)),
        ]:
            label = f"{det_name}_{model.lower()}"
            if args.debug:
                print(f"  [{label}]", end="")
            obs, null_c, pval = null_test(stream, fn, args.n_perms, rng, args.debug)
            rows.append({
                "method": label,
                "n_alarmes_observados": obs,
                "null_mean":  round(float(null_c.mean()), 2),
                "null_p50":   float(np.percentile(null_c, 50)),
                "null_p95":   float(np.percentile(null_c, 95)),
                "p_value":    round(pval, 4),
                "adwin_delta":  args.adwin_delta,
                "ph_threshold": args.ph_threshold,
                "n_perms":      args.n_perms,
            })
            print(f"  {label}: obs={obs}, null_mean={null_c.mean():.1f}, p={pval:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    print(df[["method", "n_alarmes_observados", "null_mean", "null_p95", "p_value"]].to_string(index=False))



# === latency_curve ===
"""
latency_curve.py — Curva de latência de detecção (Ação B2/B3)
=============================================================
Gera baseline i.i.d. da ADE 2019 (Seq2Seq, 30→15), injeta shift sintético
em t*=200 com magnitude em unidades de σ, e mede latência de detecção.

Uso
---
    python drift_analise/latency_curve.py
    python drift_analise/latency_curve.py --baseline_iid \\
        --shifts_in_sigma 0 0.5 1 2 5 10 \\
        --adwin_delta 0.01 --ph_threshold 25.0 \\
        --out Relas/results/drift/04_robustez_e_validacao/latency_curve_v2.csv
"""
def _latency_get_2019_stats(model: str = "Seq2Seq") -> tuple[float, float]:
    df = pd.read_csv(DATASET_CSV)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"])
    ) & (df["year"].astype(str) == "2019")
    vals = df[mask]["value"].dropna().astype(float).values
    return float(vals.mean()), float(vals.std())


def measure_latency(detector_name: str, param: float,
                    mu: float, sigma: float, shift: float,
                    t_star: int, n_trials: int,
                    rng: np.random.Generator) -> list[int | None]:
    """Return detection latencies (in samples after t_star) for each trial."""
    from river.drift import ADWIN, PageHinkley
    latencies = []
    for _ in range(n_trials):
        # baseline phase: i.i.d. from N(mu, sigma)
        pre  = rng.normal(mu,          sigma, t_star)
        # post-shift phase: i.i.d. from N(mu + shift*sigma, sigma)
        post = rng.normal(mu + shift * sigma, sigma, 1000)
        stream = np.concatenate([pre, post])

        if detector_name == "ADWIN":
            det = ADWIN(delta=param)
        else:
            det = PageHinkley(threshold=param, min_instances=3)

        detected_at = None
        for i, v in enumerate(stream):
            det.update(float(v))
            if det.drift_detected and i >= t_star:
                detected_at = i - t_star + 1
                break
        latencies.append(detected_at)
    return latencies


def main_latency_curve() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts_in_sigma", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--adwin_delta",   type=float, default=0.002)
    ap.add_argument("--ph_threshold",  type=float, default=5.0)
    ap.add_argument("--t_star",        type=int,   default=200)
    ap.add_argument("--n_trials",      type=int,   default=100)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--model",         default="Seq2Seq")
    ap.add_argument("--baseline_iid",  action="store_true", default=True)
    ap.add_argument("--out",           default=str(CH04 / "latency_curve_v2.csv"))
    args = parse_args_skip_kernel(ap)

    rng = np.random.default_rng(args.seed)
    mu, sigma = _latency_get_2019_stats(args.model)
    print(f"[info] 2019 baseline: mean={mu:.3f} mm, std={sigma:.3f} mm")
    print(f"[info] ADWIN delta={args.adwin_delta}, PH threshold={args.ph_threshold}")

    rows = []
    for shift in args.shifts_in_sigma:
        shift_mm = shift * sigma
        for det_name, param in [("ADWIN", args.adwin_delta), ("PH", args.ph_threshold)]:
            lats = measure_latency(det_name, param, mu, sigma, shift,
                                   args.t_star, args.n_trials, rng)
            detected = [l for l in lats if l is not None]
            n_det = len(detected)
            lat_p10 = float(np.percentile(detected, 10)) if detected else float("nan")
            lat_p50 = float(np.percentile(detected, 50)) if detected else float("nan")
            lat_p90 = float(np.percentile(detected, 90)) if detected else float("nan")
            rows.append({
                "detector": det_name,
                "shift_sigma": shift,
                "shift_mm_equiv": round(shift_mm, 4),
                "n_detected": n_det,
                "n_total":    args.n_trials,
                "lat_p10":    round(lat_p10, 1),
                "lat_p50":    round(lat_p50, 1),
                "lat_p90":    round(lat_p90, 1),
            })
            print(f"  {det_name} shift={shift}sigma ({shift_mm:.2f}mm) "
                  f"detected={n_det}/{args.n_trials} lat_p50={lat_p50:.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    print(df.to_string(index=False))



# === calibrate_arl ===
"""
calibrate_arl.py — Calibração de ADWIN/PH por ARL alvo (Ação A2)
=================================================================
Gera stream i.i.d. com estatísticas da ADE de 2019 e busca o parâmetro
(delta para ADWIN, threshold para PH) que produz ARL ≈ target_arl.

Uso
---
    python drift_analise/calibrate_arl.py \\
        --target_arl 500 --detector ADWIN \\
        --param_grid 1e-4 1e-3 1e-2 1e-1 \\
        --out Relas/results/drift/04_robustez_e_validacao/arl_adwin.csv

    python drift_analise/calibrate_arl.py \\
        --target_arl 500 --detector PH \\
        --param_grid 1 5 10 25 50 100 200 \\
        --out Relas/results/drift/04_robustez_e_validacao/arl_ph.csv
"""
def _cal_get_2019_stats(model: str = "Seq2Seq") -> tuple[float, float]:
    df = pd.read_csv(DATASET_CSV)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"])
    ) & (df["year"].astype(str) == "2019")
    vals = df[mask]["value"].dropna().astype(float).values
    return float(vals.mean()), float(vals.std())


def measure_arl(detector_name: str, param: float, mu: float, sigma: float,
                n_runs: int, stream_len: int, rng: np.random.Generator) -> float:
    """Average Run Length: mean samples until first alarm under i.i.d. null."""
    from river.drift import ADWIN, PageHinkley
    times_to_alarm = []
    for _ in range(n_runs):
        stream = rng.normal(mu, sigma, stream_len)
        if detector_name == "ADWIN":
            det = ADWIN(delta=param)
        else:
            det = PageHinkley(threshold=param, min_instances=3)
        t_alarm = stream_len  # no alarm = full length
        for i, v in enumerate(stream):
            det.update(float(v))
            if det.drift_detected:
                t_alarm = i + 1
                break
        times_to_alarm.append(t_alarm)
    return float(np.mean(times_to_alarm))


def main_calibrate_arl() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector",   choices=["ADWIN", "PH"], default="ADWIN")
    ap.add_argument("--target_arl", type=float, default=500.0)
    ap.add_argument("--param_grid", type=float, nargs="+",
                    default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    ap.add_argument("--n_runs",     type=int, default=200)
    ap.add_argument("--stream_len", type=int, default=5000)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--model",      default="Seq2Seq")
    ap.add_argument("--out",        default=None)
    args = parse_args_skip_kernel(ap)

    rng = np.random.default_rng(args.seed)
    mu, sigma = _cal_get_2019_stats(args.model)
    print(f"[info] 2019 baseline ({args.model}): mean={mu:.3f} mm, std={sigma:.3f} mm")
    print(f"[info] target ARL={args.target_arl}, detector={args.detector}\n")

    rows = []
    best_param, best_diff = None, float("inf")
    for param in args.param_grid:
        arl = measure_arl(args.detector, param, mu, sigma,
                          args.n_runs, args.stream_len, rng)
        diff = abs(arl - args.target_arl)
        rows.append({"param": param, "arl_measured": round(arl, 1),
                     "target_arl": args.target_arl, "abs_diff": round(diff, 1)})
        marker = " <-- best" if diff < best_diff else ""
        if diff < best_diff:
            best_diff = diff
            best_param = param
        label = "delta" if args.detector == "ADWIN" else "threshold"
        print(f"  {label}={param:.4e}  ARL={arl:.1f}  |diff|={diff:.1f}{marker}")

    df = pd.DataFrame(rows)
    if args.out is None:
        det_lower = args.detector.lower()
        args.out = str(CH04 / f"arl_{det_lower}.csv")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] saved to {out_path}")
    label = "delta" if args.detector == "ADWIN" else "threshold"
    best_row = next(r for r in rows if r["param"] == best_param)
    print(f"[ok] best {label}={best_param} -> ARL={best_row['arl_measured']:.1f} "
          f"(target={args.target_arl})")
    print(f"\n*** Use para null_test.py / latency_curve.py: "
          f"--{'adwin_delta' if args.detector == 'ADWIN' else 'ph_threshold'} {best_param} ***")



# === bocpd_run ===
"""
bocpd_run.py — Bayesian Online Change-Point Detection (Adams & MacKay 2007)
===========================================================================
Detecta breakpoints no stream de ADE por jogo usando BOCPD com prior
Normal-InverseGamma conjugado. Serve como sanity check do Pelt-RBF (Ação C3).

Uso
---
    python drift_analise/bocpd_run.py
    python drift_analise/bocpd_run.py --hazard 0.004 \\
        --out Relas/results/drift/04_robustez_e_validacao/bocpd_breakpoints.csv
"""
def _bocpd_load_stream(model: str = "Seq2Seq") -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(DATASET_CSV)
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"])
    )
    sub = df[mask].sort_values(["year", "dataset"]).dropna(subset=["value"])
    return sub["value"].astype(float).values, sub.reset_index(drop=True)


def student_t_log_pred(x: float, alpha: float, beta: float,
                       mu: float, kappa: float) -> float:
    """Log predictive density: Student-t(2α, μ, β(κ+1)/(ακ))."""
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    return float(student_t_dist.logpdf(x, df=nu, loc=mu,
                                       scale=float(np.sqrt(scale2))))


def bocpd(data: np.ndarray, hazard: float,
          mu0: float, kappa0: float, alpha0: float, beta0: float
          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run BOCPD. Returns:
      R_MAP  — most probable run length at each timestep (len T)
      R_prob — run-length probability matrix (T+1, T+1)
    """
    T = len(data)
    R = np.full((T + 1, T + 1), -np.inf)  # log probabilities
    R[0, 0] = 0.0  # log(1) = 0

    # Sufficient statistics (one set per possible run length)
    mu    = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta  = np.array([beta0])

    R_map = np.zeros(T, dtype=int)

    for t in range(1, T + 1):
        x = data[t - 1]
        current_run_lens = np.arange(t)  # 0, 1, ..., t-1

        # Predictive log-probs for all current run lengths
        log_pred = np.array([
            student_t_log_pred(x, alpha[r], beta[r], mu[r], kappa[r])
            for r in range(len(current_run_lens))
        ])

        # Growth: run length increases by 1
        log_growth = R[:t, t - 1] + log_pred + np.log(1 - hazard)

        # Change: run length resets to 0
        log_cp = (np.logaddexp.reduce(R[:t, t - 1] + log_pred)
                  + np.log(hazard))

        R[1:t + 1, t] = log_growth
        R[0, t] = log_cp

        # Normalize
        log_norm = np.logaddexp.reduce(R[:t + 1, t])
        R[:t + 1, t] -= log_norm

        R_map[t - 1] = int(np.argmax(R[:t + 1, t]))

        # Update sufficient statistics (append new run-length 0)
        mu_new    = (kappa * mu + x) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new  = beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1))

        mu    = np.append([mu0],    mu_new)
        kappa = np.append([kappa0], kappa_new)
        alpha = np.append([alpha0], alpha_new)
        beta  = np.append([beta0],  beta_new)

    return R_map, np.exp(R)


def find_breakpoints(R_map: np.ndarray, min_gap: int = 3) -> list[int]:
    """Find indices where MAP run length drops (change-point detected)."""
    bps = []
    for t in range(1, len(R_map)):
        if R_map[t] < R_map[t - 1] and R_map[t - 1] > min_gap:
            bps.append(t)
    return bps


def main_bocpd_run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazard",  type=float, default=1 / 250,
                    help="Constant hazard rate (default 1/250)")
    ap.add_argument("--model",   default="Seq2Seq")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     default=str(CH04 / "bocpd_breakpoints.csv"))
    args = parse_args_skip_kernel(ap)

    signal, meta = _bocpd_load_stream(args.model)
    T = len(signal)
    mu0    = float(signal.mean())
    sigma0 = float(signal.std())
    # Weakly informative prior
    kappa0 = 1.0
    alpha0 = 1.0
    beta0  = sigma0 ** 2

    print(f"[info] signal len={T}, mean={mu0:.3f}, std={sigma0:.3f}")
    print(f"[info] hazard={args.hazard:.5f} (~1 CP every {1/args.hazard:.0f} samples)")
    print(f"[info] prior: mu0={mu0:.3f}, kappa0={kappa0}, alpha0={alpha0}, beta0={beta0:.4f}")

    R_map, R_prob = bocpd(signal, args.hazard, mu0, kappa0, alpha0, beta0)

    bp_indices = find_breakpoints(R_map, min_gap=3)
    print(f"\n[info] Breakpoints detectados em índices: {bp_indices}")

    rows = []
    if bp_indices:
        for idx in bp_indices:
            year = meta.loc[idx, "year"] if idx < len(meta) else float("nan")
            ade  = signal[idx]
            conf = float(R_prob[0, idx]) if R_prob[0, idx] < 1 else float(R_prob[1, idx])
            rows.append({
                "breakpoint_idx": idx,
                "year": year,
                "ade_at_bp": round(ade, 4),
                "cp_prob": round(float(R_prob[0, idx]), 4),
            })
            print(f"  t={idx:3d}  year={year}  ADE={ade:.3f}  P(CP)={R_prob[0,idx]:.3f}")
    else:
        print("  Nenhum breakpoint acima do limiar detectado.")
        rows.append({"breakpoint_idx": -1, "year": float("nan"),
                     "ade_at_bp": float("nan"), "cp_prob": float("nan")})

    # Also save MAP run-length series
    map_out = Path(args.out).parent / "bocpd_run_length.csv"
    pd.DataFrame({"t": np.arange(T), "map_run_length": R_map}).to_csv(map_out, index=False)
    print(f"[ok] run-length series -> {map_out}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] breakpoints saved to {out_path}")

    # Comparison with Pelt
    pelt_bps_path = CH02 / "ruptures_elbow.csv"
    print(f"\n[info] Comparacao com Pelt (n_bps=2 em qualquer pen):")
    print(f"  BOCPD breakpoints em índices: {bp_indices}")
    if bp_indices:
        bp_years = [meta.loc[i, "year"] for i in bp_indices if i < len(meta)]
        print(f"  BOCPD breakpoints em anos: {bp_years}")
    print(f"  Pelt reportou 2 breakpoints (ver ruptures_elbow.csv)")



# === pelt_bic_curve ===
"""
pelt_bic_curve.py — Custo BIC vs número de breakpoints (Ação C1)
================================================================
Usa Dynp (programação dinâmica exata) com kernel RBF para forçar exatamente
K breakpoints em K=0..K_max e plota a curva de custo + BIC.

Uso
---
    python drift_analise/pelt_bic_curve.py
    python drift_analise/pelt_bic_curve.py --K_range 0 10 \\
        --out Relas/results/drift/04_robustez_e_validacao/ruptures_bic_curve.csv
"""
def _bic_load_stream(model: str = "Seq2Seq") -> np.ndarray:
    df = pd.read_csv(DATASET_CSV)
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"])
    )
    sub = df[mask].sort_values(["year", "dataset"]).dropna(subset=["value"])
    return sub["value"].astype(float).values


def main_pelt_bic_curve() -> None:
    prepend_matching_venv_site_packages(PROJECT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--K_range",  type=int, nargs=2, default=[0, 10])
    ap.add_argument("--model",    default="Seq2Seq")
    ap.add_argument("--out",      default=str(CH04 / "ruptures_bic_curve.csv"))
    args = parse_args_skip_kernel(ap)

    try:
        import ruptures as rpt

        Dynp = rpt.Dynp  # type: ignore[misc, assignment]
    except ImportError:
        from drift_analise.changepoint_numpy_fallback import Dynp

        print(
            "[warn] 'ruptures' não instalado — usando Dynp+RBF em NumPy "
            "(equivalente à lógica ruptures 1.1.x; veja changepoint_numpy_fallback.py)."
        )

    signal = _bic_load_stream(args.model)
    T = len(signal)
    print(f"[info] signal len={T}, mean={signal.mean():.3f}, std={signal.std():.3f}")

    signal_2d = signal.reshape(-1, 1)

    # Fit Dynp (exact) with RBF kernel — min_size=2 required
    K_min, K_max = args.K_range
    # Dynp needs at least 1 bkps; K=0 means no breakpoint
    algo = Dynp(model="rbf", min_size=2, jump=1).fit(signal_2d)

    rows = []
    prev_cost = None
    for K in range(K_min, K_max + 1):
        if K == 0:
            # Cost with 0 breakpoints = cost of entire series as one segment
            cost = algo.cost.sum_of_costs([T])
        else:
            try:
                bkps = algo.predict(n_bkps=K)
                cost = algo.cost.sum_of_costs(bkps)
            except Exception as e:
                print(f"  K={K}: error {e}")
                break

        # modified BIC (Killick 2012): mBIC(K) = cost + (3K+1)*log(T)
        m_bic = cost + (3 * K + 1) * np.log(T)
        delta_cost = (cost - prev_cost) if prev_cost is not None else float("nan")
        rows.append({
            "K": K,
            "cost": round(cost, 4),
            "m_bic": round(m_bic, 4),
            "delta_cost": round(delta_cost, 4) if not np.isnan(delta_cost) else float("nan"),
            "model": args.model,
            "T": T,
        })
        print(f"  K={K:2d}  cost={cost:.4f}  mBIC={m_bic:.4f}  dcost={delta_cost:+.4f}")
        prev_cost = cost

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")

    # Find elbow: K with minimum mBIC
    best_K = int(df.loc[df["m_bic"].idxmin(), "K"])
    print(f"[info] mBIC seleciona K={best_K} breakpoints")
    delta_costs = df["delta_cost"].dropna()
    if len(delta_costs) > 1:
        rel_drops = delta_costs.abs() / (delta_costs.abs().max() + 1e-12)
        print(f"[info] Maiores quedas de custo: {dict(zip(df['K'][1:], rel_drops.round(3)))}")



# === pelt_gamma_sensitivity ===
"""
pelt_gamma_sensitivity.py — Sensibilidade do Pelt-RBF ao parâmetro gamma (Ação C2)
===================================================================================
Varia gamma do kernel RBF em 3 fatores (0.1×, 1×, 10× default) e reporta
n_bps para cada (gamma, pen) — mostra se o resultado é robusto à calibração.

Uso
---
    python drift_analise/pelt_gamma_sensitivity.py
    python drift_analise/pelt_gamma_sensitivity.py \\
        --gamma_factors 0.1 1.0 10.0 \\
        --out Relas/results/drift/04_robustez_e_validacao/ruptures_gamma_sensitivity.csv
"""
def _gamma_load_stream(model: str = "Seq2Seq") -> np.ndarray:
    df = pd.read_csv(DATASET_CSV)
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    mask = (df["model"] == model) & (df["metric"] == "ADE") & (
        df["horizon"].isin(["30->15", "30→15"])
    )
    sub = df[mask].sort_values(["year", "dataset"]).dropna(subset=["value"])
    return sub["value"].astype(float).values


def default_gamma(signal_1d: np.ndarray) -> float:
    sig = signal_1d.reshape(-1, 1)
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(sig, metric="sqeuclidean")
    nonzero = dists[dists > 0]
    return float(1.0 / (2.0 * np.median(nonzero))) if len(nonzero) else 1.0


class FixedGammaCostRbf:
    """Custom ruptures cost wrapping RBF kernel with a fixed gamma."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self._gram = None
        self.min_size = 2

    def fit(self, signal: np.ndarray) -> "FixedGammaCostRbf":
        sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        self.signal = sig
        self._gram = rbf_kernel(sig, gamma=self.gamma)
        # Precompute cumulative sums for O(1) segment queries
        T = len(sig)
        self._K_cumsum = np.zeros((T + 1, T + 1))
        for i in range(T):
            for j in range(T):
                self._K_cumsum[i+1, j+1] = (self._K_cumsum[i, j+1]
                                             + self._K_cumsum[i+1, j]
                                             - self._K_cumsum[i, j]
                                             + self._gram[i, j])
        return self

    def _sum_seg(self, start: int, end: int) -> float:
        return (self._K_cumsum[end, end]
                - self._K_cumsum[start, end]
                - self._K_cumsum[end, start]
                + self._K_cumsum[start, start])

    def error(self, start: int, end: int) -> float:
        length = end - start
        if length < self.min_size:
            return float("inf")
        k_sum = self._sum_seg(start, end)
        return float(length - k_sum / length)

    def sum_of_costs(self, bkps: list[int]) -> float:
        starts = [0] + bkps[:-1]
        return sum(self.error(s, e) for s, e in zip(starts, bkps))


def run_pelt_custom(signal: np.ndarray, gamma: float, pen: float) -> int:
    from drift_analise.changepoint_numpy_fallback import Pelt

    cost = FixedGammaCostRbf(gamma)
    algo = Pelt(custom_cost=cost, min_size=2, jump=1)
    algo.fit(signal.reshape(-1, 1))
    bkps = algo.predict(pen=pen)
    return len(bkps) - 1  # exclude signal end


def main_pelt_gamma_sensitivity() -> None:
    prepend_matching_venv_site_packages(PROJECT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma_factors", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0])
    ap.add_argument("--pen_values", type=float, nargs="+",
                    default=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150])
    ap.add_argument("--model",  default="Seq2Seq")
    ap.add_argument("--out",    default=str(CH04 / "ruptures_gamma_sensitivity.csv"))
    args = parse_args_skip_kernel(ap)

    signal = _gamma_load_stream(args.model)
    gamma_def = default_gamma(signal)
    print(f"[info] signal len={len(signal)}, gamma_default={gamma_def:.6f}")

    rows = []
    for factor in args.gamma_factors:
        gamma = factor * gamma_def
        for pen in args.pen_values:
            n_bps = run_pelt_custom(signal, gamma, pen)
            rows.append({"gamma_factor": factor, "gamma": round(gamma, 8),
                         "pen": pen, "n_bps": n_bps})
        n_vals = {r["n_bps"] for r in rows if r["gamma_factor"] == factor}
        print(f"  factor={factor:.1f}  gamma={gamma:.6f}  n_bps unique={sorted(n_vals)}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    pivot = df.pivot_table(index="pen", columns="gamma_factor", values="n_bps", aggfunc="first")
    print("\n--- n_bps por (pen × gamma_factor) ---")
    print(pivot.to_string())



# === sample_size_sweep ===
"""
sample_size_sweep.py  —  Ponto 7 do Mês 5
==========================================

Varre o número de jogos usados no baseline 2019 (n_jogos = 1…6) e mede
o impacto no ADE baseline e no excess relativo de cada ano alvo.

Motivação
---------
O sample_size_sensitivity.py (model_analise/) varia n por ano e roda os
scripts de erro do zero — é caro. Este script é mais leve: lê os erros
de trajetória já computados (trajectory_errors_sample) e sub-amostra
os proc_sets de 2019, recalculando o ADE baseline para cada tamanho.

Pergunta: a partir de quantos jogos de 2019 o ADE baseline estabiliza?
Se estabiliza em n=3, o n=6 escolhido é seguro; se ainda oscila em n=6,
precisamos de mais dados.

Entradas
--------
  covariate_shift_out/trajectory_errors_sample.parquet (ou .csv)

Saídas
------
  Relas/results/drift/04_robustez_e_validacao/sample_size_sweep.csv
  Relas/results/drift/04_robustez_e_validacao/sample_size_sweep_en.pdf
  Relas/results/drift/04_robustez_e_validacao/sample_size_sweep_pt.pdf

Uso
---
    MPLBACKEND=Agg python drift_analise/sample_size_sweep.py
    MPLBACKEND=Agg python drift_analise/sample_size_sweep.py \\
        --n_reps 50 --seed 42 --max_n 6
"""



MODEL   = "Seq2Seq"
HORIZON = "30→15"

YEAR_COLORS = {2021: "#ff7f0e", 2022: "#2ca02c",
               2023: "#d62728", 2024: "#9467bd", 2025: "#8c564b"}


# ─── loaders ─────────────────────────────────────────────────────────────────

def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass   # fallback to CSV if pyarrow/fastparquet unavailable
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(
        "Arquivo de erros não encontrado. "
        "Execute model_analise/compute_trajectory_errors.py primeiro."
    )


# ─── sweep ───────────────────────────────────────────────────────────────────

def sweep(df: pd.DataFrame, max_n: int, n_reps: int, seed: int) -> pd.DataFrame:
    """
    Para cada n em 1…max_n:
      - repete n_reps vezes: sorteia n proc_sets de 2019 sem reposição
      - calcula ADE baseline = média das trajetórias desses n jogos
      - calcula excess de cada ano alvo vs esse baseline
    Retorna DataFrame com linhas (n, rep, ade_2019, year_y, ade_y, excess_mm).
    """
    df_seq = df[(df["model"] == MODEL) & (df["horizon"] == HORIZON)].copy()

    # proc_sets disponíveis para 2019
    proc_2019 = sorted(df_seq[df_seq["year"] == 2019]["proc_set_file"].unique())
    n_avail = len(proc_2019)
    if n_avail == 0:
        raise ValueError("Nenhum proc_set de 2019 encontrado.")
    max_n = min(max_n, n_avail)
    print(f"[info] proc_sets 2019 disponíveis: {n_avail}  |  max_n={max_n}")

    # ADE por ano alvo (fixo — não varia com subsample)
    target_years = sorted(df_seq[df_seq["year"] != 2019]["year"].dropna().unique().astype(int))
    ade_target = {}
    for yr in target_years:
        vals = df_seq[df_seq["year"] == yr]["ade_traj"].astype(float)
        ade_target[yr] = float(vals.mean())

    rng  = np.random.default_rng(seed)
    rows = []

    for n in range(1, max_n + 1):
        for rep in range(n_reps):
            chosen = rng.choice(proc_2019, size=n, replace=False).tolist()
            mask_2019 = (df_seq["year"] == 2019) & (df_seq["proc_set_file"].isin(chosen))
            ade_base = float(df_seq[mask_2019]["ade_traj"].mean())

            rows.append({
                "n_jogos_2019": n,
                "rep":          rep,
                "ade_2019":     round(ade_base, 4),
                "year":         2019,
                "ade_y":        round(ade_base, 4),
                "excess_mm":    0.0,
            })

            for yr in target_years:
                excess = ade_target[yr] - ade_base
                rows.append({
                    "n_jogos_2019": n,
                    "rep":          rep,
                    "ade_2019":     round(ade_base, 4),
                    "year":         yr,
                    "ade_y":        round(ade_target[yr], 4),
                    "excess_mm":    round(excess, 4),
                })

    return pd.DataFrame(rows)


# ─── resumo ──────────────────────────────────────────────────────────────────

def summarize_sweep(df_sweep: pd.DataFrame) -> pd.DataFrame:
    """Mediana e IQR do ADE baseline e excess por (n_jogos, year)."""
    rows = []
    for (n, yr), grp in df_sweep.groupby(["n_jogos_2019", "year"]):
        vals = grp["ade_2019"] if yr == 2019 else grp["excess_mm"]
        rows.append({
            "n_jogos_2019": n,
            "year": yr,
            "median": round(float(np.median(vals)), 4),
            "q25":    round(float(np.percentile(vals, 25)), 4),
            "q75":    round(float(np.percentile(vals, 75)), 4),
            "std":    round(float(np.std(vals)), 4),
        })
    return pd.DataFrame(rows).sort_values(["year", "n_jogos_2019"])


# ─── plots ───────────────────────────────────────────────────────────────────

def plot_sweep(df_summary: pd.DataFrame, lang: str, out_path: Path) -> None:
    years_plot = sorted(df_summary["year"].unique())
    n_plots = len(years_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, yr in zip(axes, years_plot):
        sub = df_summary[df_summary["year"] == yr].sort_values("n_jogos_2019")
        ns   = sub["n_jogos_2019"].values
        med  = sub["median"].values
        q25  = sub["q25"].values
        q75  = sub["q75"].values

        color = "#1f77b4" if yr == 2019 else YEAR_COLORS.get(yr, "C0")
        ax.plot(ns, med, "o-", color=color, lw=1.8, ms=5)
        ax.fill_between(ns, q25, q75, alpha=0.2, color=color)
        ax.axvline(6, color="gray", linestyle="--", linewidth=0.8,
                   label="n=6 (atual)" if lang == "pt" else "n=6 (current)")

        if yr == 2019:
            ylabel = "ADE 2019 baseline (mm)" if lang == "pt" else "2019 baseline ADE (mm)"
        else:
            ylabel = "Excess vs 2019 (mm)"
        ax.set_ylabel(ylabel if yr == years_plot[0] else "")
        ax.set_xlabel("n jogos 2019" if lang == "pt" else "# 2019 games")
        ax.set_title(str(yr))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=7)

    title_pt = "Sensibilidade do baseline ao número de jogos de 2019\n(mediana ± IQR)"
    title_en = "Baseline sensitivity to number of 2019 games\n(median ± IQR)"
    fig.suptitle(title_pt if lang == "pt" else title_en, fontsize=11)
    plt.tight_layout()
    saved = save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    for p in saved:
        print(f"  -> {p}")


# ─── main ────────────────────────────────────────────────────────────────────

def main_sample_size_sweep() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps",  type=int, default=50,
                    help="Repetições por tamanho de subsample (default 50)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--max_n",   type=int, default=6,
                    help="Tamanho máximo do subsample de 2019 (default 6)")
    args = parse_args_skip_kernel(ap)

    CH04.mkdir(parents=True, exist_ok=True)

    print("[info] Carregando erros de trajetória...")
    df = load_errors()
    print(f"[info] {len(df):,} linhas  |  "
          f"anos: {sorted(df['year'].dropna().unique().astype(int))}")

    print(f"\n[info] Executando sweep n=1..{args.max_n}  reps={args.n_reps}  seed={args.seed}")
    df_sweep = sweep(df, max_n=args.max_n, n_reps=args.n_reps, seed=args.seed)
    sweep_path = CH04 / "sample_size_sweep.csv"
    df_sweep.to_csv(sweep_path, index=False)
    print(f"[ok] {sweep_path}  ({len(df_sweep):,} linhas)")

    df_summary = summarize_sweep(df_sweep)
    summary_path = CH04 / "sample_size_sweep_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[ok] {summary_path}")
    print(df_summary.to_string(index=False))

    # estabilidade: variância do ADE 2019 cai abaixo de 0.1 mm ao passar de n?
    base_std = (df_summary[df_summary["year"] == 2019]
                .set_index("n_jogos_2019")["std"])
    print("\n[info] Std do ADE baseline 2019 por n:")
    for n, s in base_std.items():
        stable = "ESTAVEL" if s < 0.10 else "variavel"
        print(f"  n={n}  std={s:.4f}  [{stable}]")
    first_stable = next((n for n, s in base_std.items() if s < 0.10), None)
    if first_stable:
        print(f"\n[resultado] Baseline estabiliza (std < 0.10 mm) a partir de n={first_stable}")
    else:
        print("\n[resultado] Baseline ainda variável em n_max — considerar mais jogos")

    for lang in ("pt", "en"):
        plot_sweep(df_summary, lang=lang,
                   out_path=CH04 / f"sample_size_sweep_{lang}.pdf")



# === detect_drift_per_game ===
"""
detect_drift_per_game.py  —  Ponto 6 do Mês 5
==============================================

Aplica ADWIN e Page-Hinkley sobre um stream de ADE *agregado por jogo*
(1 valor por jogo em vez de 1 por trajetória).

Motivação
---------
Os detectores online do Mês 2 (drift_analise.ipynb Seção 6) operam em
stream de trajetórias individuais (~600/ano), produzindo centenas de alarmes
por ano. Um stream de ADE-por-jogo tem granularidade mais grossa (~6/ano)
e é mais robusto a variância intra-jogo. A pergunta é: o sinal de breakpoint
detectado por Pelt (nível de jogo) também é detectado por ADWIN/PH nesse
stream mais fino, ou só aparece com o nível ainda mais agregado?

Entradas
--------
  drift_analise/dataset/dataset.csv  —  ADE por (jogo, modelo, horizonte)

Saídas
------
  Relas/results/drift/04_robustez_e_validacao/drift_per_game_alarms.csv
  Relas/results/drift/04_robustez_e_validacao/drift_per_game_summary.csv
  Relas/results/drift/04_robustez_e_validacao/drift_per_game_timeline_en.pdf
  Relas/results/drift/04_robustez_e_validacao/drift_per_game_timeline_pt.pdf

Uso
---
    MPLBACKEND=Agg python drift_analise/detect_drift_per_game.py
    MPLBACKEND=Agg python drift_analise/detect_drift_per_game.py \\
        --adwin_delta 0.002 --ph_threshold 5.0 --model Seq2Seq --horizon "30->15"
"""



# helper compartilhado com sample_size_sweep
def _load_parquet_or_csv(p: Path, c: Path) -> pd.DataFrame:
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Nem {p} nem {c} encontrado.")
YEAR_COLORS = {2019: "#1f77b4", 2021: "#ff7f0e", 2022: "#2ca02c",
               2023: "#d62728", 2024: "#9467bd", 2025: "#8c564b"}


# ─── loaders ─────────────────────────────────────────────────────────────────

def load_per_game(model: str, horizon: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_CSV)
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    horizon_col = "30->15" if horizon in ("30->15", "30→15") else horizon
    df_h = df[
        (df["model"]   == model) &
        (df["horizon"].isin(["30->15", "30→15"]) if horizon_col == "30->15"
         else df["horizon"] == horizon_col)
    ].copy()

    if df_h.empty:
        raise ValueError(
            f"Nenhum dado para model={model}, horizon={horizon}.\n"
            f"Horizontes disponíveis: {df['horizon'].unique().tolist()}"
        )

    # ADE médio por jogo (1 linha por dataset/proc_set)
    game_col = "dataset" if "dataset" in df_h.columns else "proc_set_file"
    grp = (df_h.groupby([game_col, "year"])["value"]
               .mean()
               .reset_index()
               .rename(columns={"value": "ade", game_col: "game_id"}))
    grp = grp.sort_values(["year", "game_id"]).reset_index(drop=True)
    grp["stream_idx"] = np.arange(len(grp))
    return grp


# ─── detectores ──────────────────────────────────────────────────────────────

def run_adwin(series: np.ndarray, delta: float) -> list[int]:
    """ADWIN sobre stream de ADE por jogo. Retorna índices de alarme."""
    try:
        from river.drift import ADWIN
    except ImportError:
        raise ImportError("river não instalado. Execute: pip install river")
    det = ADWIN(delta=delta)
    alarms = []
    for i, v in enumerate(series):
        det.update(float(v))
        if det.drift_detected:
            alarms.append(i)
    return alarms


def run_page_hinkley(series: np.ndarray, threshold: float,
                     min_instances: int = 3) -> list[int]:
    """Page-Hinkley sobre stream de ADE por jogo. Retorna índices de alarme."""
    try:
        from river.drift import PageHinkley
    except ImportError:
        raise ImportError("river não instalado. Execute: pip install river")
    det = PageHinkley(threshold=threshold, min_instances=min_instances)
    alarms = []
    for i, v in enumerate(series):
        det.update(float(v))
        if det.drift_detected:
            alarms.append(i)
    return alarms


# ─── saídas ──────────────────────────────────────────────────────────────────

def build_alarms_df(df_game: pd.DataFrame,
                    adwin_alarms: list[int],
                    ph_alarms: list[int]) -> pd.DataFrame:
    rows = []
    for method, idxs in [("ADWIN", adwin_alarms), ("PageHinkley", ph_alarms)]:
        for idx in idxs:
            r = df_game.iloc[idx]
            rows.append({
                "method":      method,
                "stream_idx":  int(idx),
                "year":        int(r["year"]) if pd.notna(r["year"]) else None,
                "game_id":     r["game_id"],
                "ade_at_alarm": round(float(r["ade"]), 4),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["method", "stream_idx", "year", "game_id", "ade_at_alarm"])


def summarize(df_alarms: pd.DataFrame) -> pd.DataFrame:
    if df_alarms.empty:
        return pd.DataFrame(columns=["method", "year", "n_alarms"])
    return (df_alarms.groupby(["method", "year"])
                     .size()
                     .rename("n_alarms")
                     .reset_index()
                     .sort_values(["method", "year"]))


def plot_timeline(df_game: pd.DataFrame, df_alarms: pd.DataFrame,
                  adwin_delta: float, ph_threshold: float,
                  lang: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = df_game["stream_idx"].values
    y = df_game["ade"].values

    # colorir por ano
    for yr in sorted(df_game["year"].dropna().unique().astype(int)):
        sub = df_game[df_game["year"] == yr]
        ax.scatter(sub["stream_idx"], sub["ade"],
                   color=YEAR_COLORS.get(yr, "C0"), s=70, alpha=0.9,
                   edgecolor="white", linewidth=0.5, label=str(yr), zorder=3)
    ax.plot(x, y, color="black", lw=0.8, alpha=0.4, zorder=2)

    # alarmes
    marker_styles = {"ADWIN": ("^", "#e377c2", 14), "PageHinkley": ("v", "#17becf", 14)}
    if not df_alarms.empty:
        for method, (mk, col, ms) in marker_styles.items():
            sub_a = df_alarms[df_alarms["method"] == method]
            for _, row in sub_a.iterrows():
                ax.scatter(row["stream_idx"], row["ade_at_alarm"],
                           marker=mk, color=col, s=ms**2, zorder=5,
                           label=method if _ == sub_a.index[0] else "")

    # separadores de ano
    year_starts = df_game.groupby("year")["stream_idx"].min()
    for yr, xi in year_starts.items():
        if xi > 0:
            ax.axvline(xi - 0.5, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.5)

    if lang == "pt":
        title = (f"Detecção de drift — stream ADE por jogo (Seq2Seq 30→15)\n"
                 f"ADWIN δ={adwin_delta}  |  Page-Hinkley θ={ph_threshold}")
        ylabel = "ADE médio por jogo (mm)"
    else:
        title = (f"Drift detection — per-game ADE stream (Seq2Seq 30→15)\n"
                 f"ADWIN δ={adwin_delta}  |  Page-Hinkley θ={ph_threshold}")
        ylabel = "Mean ADE per game (mm)"

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Índice global do jogo" if lang == "pt" else "Global game index")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=8, ncol=4, loc="upper left")
    plt.tight_layout()
    saved = save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    for p in saved:
        print(f"  -> {p}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main_detect_drift_per_game() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="Seq2Seq")
    ap.add_argument("--horizon",       default="30->15")
    ap.add_argument("--adwin_delta",   type=float, default=0.002)
    ap.add_argument("--ph_threshold",  type=float, default=5.0)
    ap.add_argument("--ph_min",        type=int,   default=3)
    args = parse_args_skip_kernel(ap)

    CH04.mkdir(parents=True, exist_ok=True)

    print(f"[info] Carregando stream per-game: model={args.model} horizon={args.horizon}")
    df_game = load_per_game(args.model, args.horizon)
    print(f"[info] {len(df_game)} jogos no stream | anos: "
          f"{sorted(df_game['year'].dropna().unique().astype(int))}")
    print(df_game[["year", "game_id", "ade"]].to_string(index=False))

    series = df_game["ade"].astype(float).values

    print(f"\n[info] ADWIN (delta={args.adwin_delta}) ...")
    adwin_alarms = run_adwin(series, delta=args.adwin_delta)
    print(f"  {len(adwin_alarms)} alarmes em índices: {adwin_alarms}")

    print(f"\n[info] Page-Hinkley (threshold={args.ph_threshold}) ...")
    ph_alarms = run_page_hinkley(series, threshold=args.ph_threshold,
                                  min_instances=args.ph_min)
    print(f"  {len(ph_alarms)} alarmes em índices: {ph_alarms}")

    df_alarms  = build_alarms_df(df_game, adwin_alarms, ph_alarms)
    df_summary = summarize(df_alarms)

    alarms_path  = CH04 / "drift_per_game_alarms.csv"
    summary_path = CH04 / "drift_per_game_summary.csv"
    df_alarms.to_csv(alarms_path,   index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[ok] {alarms_path}  ({len(df_alarms)} linhas)")
    print(f"[ok] {summary_path}")
    print(df_summary.to_string(index=False))

    for lang in ("pt", "en"):
        plot_timeline(
            df_game, df_alarms,
            adwin_delta=args.adwin_delta,
            ph_threshold=args.ph_threshold,
            lang=lang,
            out_path=CH04 / f"drift_per_game_timeline_{lang}.pdf",
        )

    # comparação com Pelt
    print("\n[info] Comparação com breakpoint Pelt (antes_2022):")
    pelt_bp_year = 2022
    adwin_years  = set(df_alarms[df_alarms["method"] == "ADWIN"]["year"].dropna().astype(int))
    ph_years     = set(df_alarms[df_alarms["method"] == "PageHinkley"]["year"].dropna().astype(int))
    print(f"  Pelt detectou transicao em: {pelt_bp_year}")
    print(f"  ADWIN alarmes em anos: {sorted(adwin_years)}")
    print(f"  PH    alarmes em anos: {sorted(ph_years)}")
    agree = pelt_bp_year in adwin_years or pelt_bp_year in ph_years
    print(f"  Concordancia com Pelt? {'SIM' if agree else 'NAO'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Capítulo 4 — pipeline agregado")
    p.add_argument(
        "step",
        choices=[
            "null_test",
            "latency_curve",
            "calibrate_arl",
            "bocpd_run",
            "pelt_bic_curve",
            "pelt_gamma_sensitivity",
            "sample_size_sweep",
            "detect_drift_per_game",
        ],
        help="Qual sub-pipeline executar (argv adicional passa ao argparse interno).",
    )
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    {
        "null_test": main_null_test,
        "latency_curve": main_latency_curve,
        "calibrate_arl": main_calibrate_arl,
        "bocpd_run": main_bocpd_run,
        "pelt_bic_curve": main_pelt_bic_curve,
        "pelt_gamma_sensitivity": main_pelt_gamma_sensitivity,
        "sample_size_sweep": main_sample_size_sweep,
        "detect_drift_per_game": main_detect_drift_per_game,
    }[args.step]()
