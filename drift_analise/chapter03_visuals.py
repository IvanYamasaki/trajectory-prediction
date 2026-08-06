"""Capítulo 3 — figuras IW + validação por divisão (ex-plot_iw_results + ex-division_validation)."""
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
import argparse
import pickle

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from dataset.load_dataset import LoadDataSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.plotting import YEAR_COLORS_TAB10 as COLOR_YEAR
from drift_analise.cli_argv import parse_args_skip_kernel
from drift_analise.drift_output_paths import CH03, ensure_drift_decomposition_csv

COV_DIR = Path("covariate_shift_out")
OUT_DIR = CH03
IW_CSV = OUT_DIR / "iw_decomposition.csv"
DIV_MAP = Path("drift_analise") / "division_map.csv"
HORIZON = "30→15"
MIN_GAMES = 2
BASELINE = 2019

COLORS = {"ade_y": "#d62728", "ade_iw": "#2ca02c", "ade_2019": "#1f77b4"}
ESS_THRESH = 0.5

# COLOR_YEAR vem de common.plotting (YEAR_COLORS_TAB10)

BASELINE_YEAR = BASELINE
ALL_TARGETS = [2021, 2022, 2023, 2024, 2025]
ESS_STABLE_THRESHOLD = ESS_THRESH
ESS_MARGINAL_THRESHOLD = 0.3
LOOK_BACK = 30
LOOK_FORTH = 15
STATS_PATH = PROJECT_ROOT / "model" / "normalization_stats_30_15.pkl"


# --- LSIF / importance weights (ex-compute_importance_weights) -------------

def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(
        "Execute model_analise/compute_trajectory_errors.py primeiro."
    )


def build_loader() -> LoadDataSet:
    loader = LoadDataSet(LOOK_BACK, LOOK_FORTH, target="dv")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    loader.robots_avg = np.asarray(stats["avg"],      dtype=np.float64)
    loader.robots_std = np.asarray(stats["std"],      dtype=np.float64)
    loader.ball_avg   = np.asarray(stats["ball_avg"], dtype=np.float64)
    loader.ball_std   = np.asarray(stats["ball_std"], dtype=np.float64)
    return loader


# ─────────────────────────── features ─────────────────────────────────────

def features_from_x(loader: LoadDataSet, x_norm: np.ndarray) -> np.ndarray:
    """Extrai vetor cinemático por trajetória: speed_mean, speed_p90, speed_p99,
    accel_mean, accel_p90, accel_p99, turn_mean, turn_p90."""
    real = x_norm[:, :, :4] * loader.robots_std[:4] + loader.robots_avg[:4]
    vx, vy = real[:, :, 2], real[:, :, 3]
    speed  = np.hypot(vx, vy)
    dv     = np.diff(real[:, :, 2:4], axis=1)
    accel  = np.linalg.norm(dv, axis=2)
    heading = np.arctan2(vy, vx)
    dheading = np.abs(np.diff(heading, axis=1))
    turn = np.pad(dheading, ((0,0),(0,1)), mode='edge')

    # clip p99.5 do pool completo (estabiliza antes de log)
    for arr in (speed, accel, turn):
        clip_hi = np.nanpercentile(arr, 99.5)
        arr.clip(0, clip_hi, out=arr)

    sp99 = np.percentile(speed,  99, axis=1)
    ac99 = np.percentile(accel, 99, axis=1)
    TINY = 1e-6
    return np.column_stack([
        np.log1p(np.mean(speed, axis=1)),
        np.log1p(np.percentile(speed, 90, axis=1)),
        np.log1p(sp99),
        np.log1p(np.mean(accel, axis=1)),
        np.log1p(np.percentile(accel, 90, axis=1)),
        np.log1p(ac99),
        np.log1p(np.mean(turn, axis=1)),
        np.log1p(np.percentile(turn, 90, axis=1)),
    ]).astype(np.float64)


def extract_features(rows: pd.DataFrame, loader: LoadDataSet) -> tuple[np.ndarray, pd.DataFrame]:
    import re
    feats, aligned = [], []
    for proc_file, grp in rows.groupby("proc_set_file", sort=True):
        m = re.search(r"(\d+)", str(proc_file))
        if not m:
            continue
        n = int(m.group(1))
        fpath = f"dataset/proc_set_{n}"
        try:
            x_all, _, _, _ = loader.load_data([fpath], for_test=True)
        except Exception as e:
            print(f"  [warn] falha ao carregar {fpath}: {e}")
            continue
        idx = grp["traj_id"].astype(int).to_numpy()
        valid = (idx >= 0) & (idx < len(x_all))
        idx = idx[valid]
        grp = grp.iloc[np.flatnonzero(valid)]
        feats.append(features_from_x(loader, x_all[idx].astype(np.float32)))
        aligned.append(grp)
    if not feats:
        raise ValueError("Nenhuma feature extraída.")
    return np.vstack(feats), pd.concat(aligned, ignore_index=True)


# ─────────────────────────── LSIF estimator ───────────────────────────────

def _kernel(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    d2 = pairwise_distances(x, centers, metric="sqeuclidean")
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _sigma(xs: np.ndarray, xt: np.ndarray, rng: np.random.Generator) -> float:
    x = np.vstack([xs, xt])
    n = min(len(x), 2000)
    s = x[rng.choice(len(x), size=n, replace=False)]
    d = pairwise_distances(s, s, metric="euclidean")
    vals = d[np.triu_indices_from(d, k=1)]
    sigma = float(np.median(vals[vals > 0]))
    return sigma if np.isfinite(sigma) and sigma > 0 else 1.0


def lsif_weights(
    xs: np.ndarray, xt: np.ndarray,
    n_centers: int = 300, reg: float = 1e-3, seed: int = 42,
) -> np.ndarray:
    """Estima w_i = p_source(x_i) / p_target(x_i) para x_i em xs (source).
    Aqui source = ano-alvo y, target = baseline 2019.
    Logo w_i = p_y(x_i) / p_2019(x_i).
    Para ADE_IW ponderamos trajetórias de y por 1/w_i = p_2019/p_y.
    """
    rng = np.random.default_rng(seed)
    nc = min(n_centers, len(xt), len(xs))
    # centros do TARGET (2019) — estimamos p_2019
    centers = xt[rng.choice(len(xt), size=nc, replace=False)]
    sigma = _sigma(xs, xt, rng)

    phi_s = _kernel(xs, centers, sigma)   # [n_y, nc]
    phi_t = _kernel(xt, centers, sigma)   # [n_2019, nc]

    # H = E_y[phi phi^T],  h = E_2019[phi]
    H = (phi_s.T @ phi_s) / len(phi_s)
    h = np.mean(phi_t, axis=0)
    theta = np.linalg.solve(H + reg * np.eye(nc), h)
    # r(x) = phi(x)^T theta  ≈  p_2019(x) / p_y(x)
    r = np.maximum(phi_s @ theta, 0.0)
    mean_r = float(np.mean(r))
    if mean_r > 0:
        r = r / mean_r
    return r.astype(np.float64)


def rulisf_weights(
    xs: np.ndarray, xt: np.ndarray,
    n_centers: int = 300, reg: float = 1e-3, alpha: float = 0.1, seed: int = 42,
) -> np.ndarray:
    """RuLSIF — Relative unconstrained LSIF (Yamada et al. 2013).

    Estimates r_alpha(x) = p_source(x) / ((1-alpha)*p_source(x) + alpha*p_target(x))
    for x in xs (source = year y).  alpha in (0,1] bounds the ratio by 1/alpha,
    making it more robust than LSIF under heavy-tailed distributions.

    H_{lm} = (1-alpha)/n_s * Phi_s^T Phi_s  +  alpha/n_t * Phi_t^T Phi_t
    h_l    = (1/n_s) * mean(Phi_s)
    theta* = (H + lambda*I)^{-1} h
    """
    rng = np.random.default_rng(seed)
    nc = min(n_centers, len(xt), len(xs))
    centers = xt[rng.choice(len(xt), size=nc, replace=False)]
    sigma = _sigma(xs, xt, rng)

    phi_s = _kernel(xs, centers, sigma)   # [n_y,    nc]
    phi_t = _kernel(xt, centers, sigma)   # [n_2019, nc]

    n_s, n_t = len(xs), len(xt)
    # Yamada 2013 eq. (7):
    # H_{lm} = (1-alpha)/n_te * Phi_te^T Phi_te  +  alpha/n_tr * Phi_tr^T Phi_tr
    # h_l    = (1/n_te) * mean(Phi_te)   ← mean over TEST (xt=2019)
    # te = xt = 2019 (distribution we want);  tr = xs = year y (reference)
    H = ((1.0 - alpha) / n_t) * (phi_t.T @ phi_t) + (alpha / n_s) * (phi_s.T @ phi_s)
    h = np.mean(phi_t, axis=0)
    theta = np.linalg.solve(H + reg * np.eye(nc), h)
    r = np.maximum(phi_s @ theta, 0.0)
    mean_r = float(np.mean(r))
    if mean_r > 0:
        r = r / mean_r
    return r.astype(np.float64)


def ess_ratio(w: np.ndarray) -> float:
    """Effective Sample Size ratio: (sum w)^2 / (n * sum w^2)."""
    s1 = float(np.sum(w))
    s2 = float(np.sum(w ** 2))
    if s2 < 1e-12:
        return 0.0
    return (s1 ** 2) / (len(w) * s2)


# ─────────────────────────── bootstrap ────────────────────────────────────

def bootstrap_iw_ade(
    ade: np.ndarray, w: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI95 do ADE importance-weighted."""
    rng = np.random.default_rng(seed)
    N = len(ade)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        wi, ai = w[idx], ade[idx]
        sw = wi.sum()
        boots[i] = (wi @ ai) / sw if sw > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    return float(np.percentile(boots, 2.5)), float(np.mean(boots)), float(np.percentile(boots, 97.5))


def bootstrap_mean_ade(
    ade: np.ndarray, n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(ade, size=len(ade), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.mean(means)), float(np.percentile(means, 97.5))


def bootstrap_recovery_block(
    ade_y: np.ndarray, w: np.ndarray, game_ids_y: np.ndarray,
    ade_2019: np.ndarray, game_ids_2019: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float, float]:
    """Block bootstrap CI95; block = game (proc_set_file).

    Resamples games with replacement to respect intra-game trajectory correlation.
    Returns (rec_ratio_lo, rec_ratio_hi, corr_lo_mm, corr_hi_mm).

    rec_ratio_*: percentile CI of the per-replicate ratio
        (m_y_b - m_iw_b) / (m_y_b - m_2_b) * 100. With only ~6 blocks per year
        the resampled excess in the denominator degenerates toward zero whenever
        the true excess is small (2022: 0.225 mm), exploding the ratio
        (e.g. [2.4; 707.6]). Kept for backward comparison only.
    corr_*_mm: percentile CI of the absolute IW correction m_y_b - m_iw_b, in mm.
        No ratio involved, so it stays stable regardless of the excess size.
        Dividing these bounds by the *observed full-sample* excess gives the
        fixed-denominator recovery CI (linearization) reported downstream.
    """
    rng = np.random.default_rng(seed)
    games_y   = np.unique(game_ids_y)
    games_2   = np.unique(game_ids_2019)
    # precompute index arrays for O(1) lookup per game
    idx_by_game_y = {g: np.where(game_ids_y   == g)[0] for g in games_y}
    idx_by_game_2 = {g: np.where(game_ids_2019 == g)[0] for g in games_2}

    recov_boots: list[float] = []
    corr_boots:  list[float] = []
    for _ in range(n_boot):
        sampled_y = rng.choice(games_y, size=len(games_y), replace=True)
        sampled_2 = rng.choice(games_2, size=len(games_2), replace=True)
        idx_y = np.concatenate([idx_by_game_y[g] for g in sampled_y])
        idx_2 = np.concatenate([idx_by_game_2[g] for g in sampled_2])

        m_y_b  = float(np.mean(ade_y[idx_y]))
        m_2_b  = float(np.mean(ade_2019[idx_2]))
        w_b    = w[idx_y]
        m_iw_b = float((w_b @ ade_y[idx_y]) / max(float(w_b.sum()), 1e-9))
        corr_boots.append(m_y_b - m_iw_b)
        ex_b   = m_y_b - m_2_b
        if ex_b > 1e-6:
            recov_boots.append((m_y_b - m_iw_b) / ex_b * 100.0)

    corr = np.array([c for c in corr_boots if np.isfinite(c)])
    corr_lo = float(np.percentile(corr, 2.5)) if corr.size >= 2 else np.nan
    corr_hi = float(np.percentile(corr, 97.5)) if corr.size >= 2 else np.nan

    valid = np.array([v for v in recov_boots if np.isfinite(v)])
    if valid.size < 2:
        return np.nan, np.nan, corr_lo, corr_hi
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)),
            corr_lo, corr_hi)


def classifier_two_sample_test(
    xs: np.ndarray, xt: np.ndarray,
    n_permut: int = 200, seed: int = 42, subsample: int = 3000,
) -> dict:
    """Lopez-Paz & Oquab (2017) classifier two-sample test.

    Trains a logistic regression to distinguish year-y features (xs) from
    2019 features (xt).  AUC > 0.5 indicates covariate shift.
    coverage_pct = clip(2*(AUC-0.5)*100, 0, 100) — data-driven covariate estimate.
    p-value from n_permut label-permutation replicates.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    # subsample each class for speed; preserves balance
    def _sub(arr: np.ndarray, n: int) -> np.ndarray:
        return arr[rng.choice(len(arr), min(n, len(arr)), replace=False)]

    xs_s = _sub(xs, subsample)
    xt_s = _sub(xt, subsample)
    X = np.vstack([xs_s, xt_s])
    y = np.concatenate([np.ones(len(xs_s)), np.zeros(len(xt_s))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=int(seed), stratify=y
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=int(seed))
    clf.fit(X_tr, y_tr)
    auc_obs = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    coverage_pct = float(np.clip(2.0 * (auc_obs - 0.5) * 100.0, 0.0, 100.0))

    # permutation p-value: reshuffle all labels, retrain
    perm_aucs = np.empty(n_permut)
    for i in range(n_permut):
        y_perm = rng.permutation(y)
        X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
            X, y_perm, test_size=0.2, random_state=int(seed + i + 1), stratify=y_perm
        )
        clf_p = LogisticRegression(max_iter=500, solver="lbfgs", random_state=int(seed))
        clf_p.fit(X_tr_p, y_tr_p)
        perm_aucs[i] = float(roc_auc_score(y_te_p, clf_p.predict_proba(X_te_p)[:, 1]))

    pvalue = float(np.mean(perm_aucs >= auc_obs))
    return {
        "auc":          round(auc_obs, 4),
        "coverage_pct": round(coverage_pct, 2),
        "pvalue":       round(pvalue, 4),
        "n_permut":     n_permut,
        "n_sub":        len(xs_s),
    }


# ─────────────────────────── main ─────────────────────────────────────────

def main_compute_importance_weights(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_years", type=int, nargs="+", default=ALL_TARGETS)
    ap.add_argument("--n_centers",    type=int,   default=300)
    ap.add_argument("--reg",          type=float, default=1e-3)
    ap.add_argument("--n_boot",       type=int,   default=2000)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--lsif_variant", choices=["lsif", "rulisf"], default="rulisf",
                    help="Estimador de density ratio: 'rulisf' (Yamada 2013, default) "
                         "ou 'lsif' clássico.")
    ap.add_argument("--rulisf_alpha", type=float, default=0.1,
                    help="Alpha do RuLSIF — limita o ratio a 1/alpha (default 0.1).")
    ap.add_argument("--classifier_test", action="store_true",
                    help="Roda classifier two-sample test (Lopez-Paz & Oquab 2017). "
                         "Saída: classifier_two_sample_test.csv.")
    ap.add_argument("--n_permut", type=int, default=200,
                    help="Permutações para o p-value do classifier test (default 200).")
    ap.add_argument("--per_feature",  action="store_true",
                    help="Roda LSIF marginal em cada feature isoladamente "
                         "para identificar a dimensão dominante. "
                         "Também gera iw_feature_corr.csv e iw_loo_analysis.csv.")
    ap.add_argument("--self_test", action="store_true",
                    help="Calibração LSIF: divide 2019 em dois subconjuntos e verifica "
                         "se recovery ≈ 0%%. Saída: iw_self_test.csv.")
    args = parse_args_skip_kernel(ap, argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COV_DIR.mkdir(parents=True, exist_ok=True)

    df_all = load_errors()
    df_all = df_all[
        (df_all["model"]   == "Seq2Seq") &
        (df_all["horizon"] == HORIZON)
    ].copy()

    loader = build_loader()

    # extrai features do baseline 2019 uma vez
    src_rows_raw = df_all[df_all["year"] == BASELINE_YEAR].reset_index(drop=True)
    print(f"\n[info] baseline 2019: {len(src_rows_raw):,} trajetórias")
    x_src_raw, src_rows = extract_features(src_rows_raw, loader)
    ade_2019_all    = src_rows["ade_traj"].astype(float).to_numpy()
    game_ids_2019   = src_rows["proc_set_file"].to_numpy()
    ade_2019_mean   = float(np.mean(ade_2019_all))
    print(f"[info] ADE 2019 (sem pesos): {ade_2019_mean:.4f} mm")

    rows_iw = []
    rows_per_feat: list[dict] = []   # preenchido se --per_feature

    for year in sorted(args.target_years):
        tgt_rows_raw = df_all[df_all["year"] == year].reset_index(drop=True)
        if tgt_rows_raw.empty:
            print(f"[warn] ano {year}: sem dados — pulando.")
            continue

        print(f"\n[info] === Ano {year} ({len(tgt_rows_raw):,} traj.) ===")
        x_tgt_raw, tgt_rows = extract_features(tgt_rows_raw, loader)
        ade_y       = tgt_rows["ade_traj"].astype(float).to_numpy()
        game_ids_y  = tgt_rows["proc_set_file"].to_numpy()
        ade_y_mean  = float(np.mean(ade_y))

        scaler = StandardScaler().fit(np.vstack([x_src_raw, x_tgt_raw]))
        xs = scaler.transform(x_tgt_raw)   # fonte = ano y
        xt = scaler.transform(x_src_raw)   # target = 2019

        # w_i = p_2019(x_i) / p_y(x_i)  para x_i em y
        if args.lsif_variant == "rulisf":
            w = rulisf_weights(xs, xt, n_centers=args.n_centers, reg=args.reg,
                               alpha=args.rulisf_alpha, seed=args.seed)
        else:
            w = lsif_weights(xs, xt, n_centers=args.n_centers, reg=args.reg, seed=args.seed)

        ess = ess_ratio(w)
        stable = ess >= ESS_STABLE_THRESHOLD
        ess_regime = ("stable"   if ess >= ESS_STABLE_THRESHOLD   else
                      "marginal" if ess >= ESS_MARGINAL_THRESHOLD  else
                      "unstable")
        status = "ok" if stable else f"INSTÁVEL (ESS < {ESS_STABLE_THRESHOLD})"
        print(f"  ESS ratio = {ess:.3f}  [{status}]  regime={ess_regime}")

        # ADE_IW = sum(w_i * ade_yi) / sum(w_i)
        sw = w.sum()
        ade_iw = float((w @ ade_y) / sw) if sw > 0 else np.nan

        # bootstrap CI95
        ci_lo_y, _, ci_hi_y = bootstrap_mean_ade(ade_y, args.n_boot, args.seed)
        if stable:
            ci_lo_iw, _, ci_hi_iw = bootstrap_iw_ade(ade_y, w, args.n_boot, args.seed)
        else:
            ci_lo_iw, ci_hi_iw = np.nan, np.nan

        # ────────── recovery_pct robusto a casos degenerados ──────────
        # Caso 1: ADE_y <= ADE_2019  -> não há excesso a recuperar.
        #         Reportar excess=0, recovery=NaN, note explicativa.
        # Caso 2: ADE_y >  ADE_2019  -> recovery clipado em [0, 100],
        #         flag overshoot=True quando excederia 100.
        excess = ade_y_mean - ade_2019_mean
        if excess <= 1e-6:
            recovery_raw   = np.nan
            recovery       = np.nan
            overshoot      = False
            recovery_note  = "no_excess"
        else:
            recovery_raw = (ade_y_mean - ade_iw) / excess * 100.0
            overshoot    = bool(recovery_raw > 100.0 or recovery_raw < 0.0)
            recovery     = float(np.clip(recovery_raw, 0.0, 100.0))
            recovery_note = ("overshoot_high" if recovery_raw > 100 else
                             "overshoot_low"  if recovery_raw < 0   else "ok")

        # bootstrap CI95 do recovery_pct (só se houver excess)
        if np.isfinite(recovery):
            rng_b = np.random.default_rng(args.seed + year)
            recov_boots = np.empty(args.n_boot)
            for b in range(args.n_boot):
                idx_y = rng_b.integers(0, len(ade_y), size=len(ade_y))
                idx_2 = rng_b.integers(0, len(ade_2019_all), size=len(ade_2019_all))
                m_y  = float(np.mean(ade_y[idx_y]))
                m_2  = float(np.mean(ade_2019_all[idx_2]))
                m_iw_b = float((w[idx_y] @ ade_y[idx_y]) / max(w[idx_y].sum(), 1e-9))
                ex_b = m_y - m_2
                if ex_b > 1e-6:
                    # Ponto 3: bootstrap honesto — sem clip dentro do loop.
                    # O clip é aplicado APENAS no ponto de reporte (recovery).
                    # Clipar aqui estreitaria artificialmente o IC.
                    recov_boots[b] = (m_y - m_iw_b) / ex_b * 100.0
                else:
                    recov_boots[b] = np.nan
            valid = recov_boots[np.isfinite(recov_boots)]
            recovery_ci_lo = float(np.percentile(valid, 2.5)) if valid.size else np.nan
            recovery_ci_hi = float(np.percentile(valid, 97.5)) if valid.size else np.nan
        else:
            recovery_ci_lo = recovery_ci_hi = np.nan

        # block bootstrap (bloco = jogo) — IC honesto respeitando correlação intra-jogo
        # Razão por réplica (instável com 6 blocos + excesso ~0) mantida para
        # comparação; o IC reportável é o da correção absoluta (mm) e sua versão
        # percentual com denominador FIXO no excesso observado (linearização).
        if np.isfinite(recovery):
            block_ci_lo, block_ci_hi, corr_ci_lo, corr_ci_hi = bootstrap_recovery_block(
                ade_y, w, game_ids_y,
                ade_2019_all, game_ids_2019,
                n_boot=args.n_boot, seed=args.seed + year,
            )
            rec_fix_lo = corr_ci_lo / excess * 100.0
            rec_fix_hi = corr_ci_hi / excess * 100.0
        else:
            block_ci_lo = block_ci_hi = np.nan
            corr_ci_lo = corr_ci_hi = rec_fix_lo = rec_fix_hi = np.nan
        iw_corr_mm = (ade_y_mean - ade_iw) if np.isfinite(ade_iw) else np.nan

        _r  = f"{recovery:.1f}" if np.isfinite(recovery) else "n/a"
        _bl = f"{block_ci_lo:.1f}" if np.isfinite(block_ci_lo) else "n/a"
        _bh = f"{block_ci_hi:.1f}" if np.isfinite(block_ci_hi) else "n/a"
        _cl = f"{corr_ci_lo:.3f}" if np.isfinite(corr_ci_lo) else "n/a"
        _ch = f"{corr_ci_hi:.3f}" if np.isfinite(corr_ci_hi) else "n/a"
        _fl = f"{rec_fix_lo:.1f}" if np.isfinite(rec_fix_lo) else "n/a"
        _fh = f"{rec_fix_hi:.1f}" if np.isfinite(rec_fix_hi) else "n/a"
        print(f"  ADE_y={ade_y_mean:.4f}  ADE_IW={ade_iw:.4f}  "
              f"ADE_2019={ade_2019_mean:.4f}  recovery={_r}%  "
              f"[{recovery_note}]  block_IC_ratio=[{_bl}, {_bh}]  "
              f"corr_mm_IC=[{_cl}, {_ch}]  rec_fix_IC=[{_fl}, {_fh}]")

        np.save(COV_DIR / f"importance_weights_{year}.npy", w)

        rows_iw.append({
            "year":               year,
            "lsif_variant":       args.lsif_variant,
            "ade_y":              round(ade_y_mean, 4),
            "ade_y_ci_lo":        round(ci_lo_y, 4),
            "ade_y_ci_hi":        round(ci_hi_y, 4),
            "ade_2019":           round(ade_2019_mean, 4),
            "ade_iw":             round(ade_iw, 4) if np.isfinite(ade_iw) else np.nan,
            "ade_iw_ci_lo":       round(ci_lo_iw, 4) if np.isfinite(ci_lo_iw) else np.nan,
            "ade_iw_ci_hi":       round(ci_hi_iw, 4) if np.isfinite(ci_hi_iw) else np.nan,
            "ess_ratio":          round(ess, 4),
            "ess_stable":         stable,
            "ess_regime":         ess_regime,
            "recovery_pct":       round(recovery, 2) if np.isfinite(recovery) else np.nan,
            "recovery_ci_lo":     round(recovery_ci_lo, 2) if np.isfinite(recovery_ci_lo) else np.nan,
            "recovery_ci_hi":     round(recovery_ci_hi, 2) if np.isfinite(recovery_ci_hi) else np.nan,
            "recovery_ci_lo_block": round(block_ci_lo, 2) if np.isfinite(block_ci_lo) else np.nan,
            "recovery_ci_hi_block": round(block_ci_hi, 2) if np.isfinite(block_ci_hi) else np.nan,
            "iw_corr_mm":           round(iw_corr_mm, 4) if np.isfinite(iw_corr_mm) else np.nan,
            "iw_corr_mm_ci_lo_block": round(corr_ci_lo, 4) if np.isfinite(corr_ci_lo) else np.nan,
            "iw_corr_mm_ci_hi_block": round(corr_ci_hi, 4) if np.isfinite(corr_ci_hi) else np.nan,
            "recovery_ci_lo_block_fixed": round(rec_fix_lo, 2) if np.isfinite(rec_fix_lo) else np.nan,
            "recovery_ci_hi_block_fixed": round(rec_fix_hi, 2) if np.isfinite(rec_fix_hi) else np.nan,
            "recovery_raw":       round(recovery_raw, 2) if np.isfinite(recovery_raw) else np.nan,
            "overshoot":          overshoot,
            "recovery_note":      recovery_note,
            "excess_total":       round(excess, 4),
            "n_source":           len(x_src_raw),
            "n_target":           len(x_tgt_raw),
        })

        # ─────────── Classifier two-sample test (Lopez-Paz & Oquab 2017) ──
        if args.classifier_test:
            print(f"  [classifier] rodando test (n_permut={args.n_permut}) ...")
            ct = classifier_two_sample_test(xs, xt, n_permut=args.n_permut, seed=args.seed)
            print(f"  [classifier] AUC={ct['auc']:.4f}  coverage={ct['coverage_pct']:.1f}%  "
                  f"p={ct['pvalue']:.4f}  (n_sub={ct['n_sub']})")
            rows_iw[-1]["clf_auc"]      = ct["auc"]
            rows_iw[-1]["clf_coverage"] = ct["coverage_pct"]
            rows_iw[-1]["clf_pvalue"]   = ct["pvalue"]

        # ─────────── IW marginal por feature (Bloco B do review) ────────
        if args.per_feature and excess > 1e-6:
            FEAT_NAMES = ["speed_mean", "speed_p90", "speed_p99",
                          "accel_mean", "accel_p90", "accel_p99",
                          "turn_mean", "turn_p90"]
            for f_idx, f_name in enumerate(FEAT_NAMES):
                xs_f = xs[:, f_idx:f_idx + 1]
                xt_f = xt[:, f_idx:f_idx + 1]
                w_f  = lsif_weights(xs_f, xt_f, n_centers=args.n_centers,
                                    reg=args.reg, seed=args.seed)
                ess_f = ess_ratio(w_f)
                ade_iw_f = float((w_f @ ade_y) / w_f.sum()) if w_f.sum() > 0 else np.nan
                rec_raw_f = ((ade_y_mean - ade_iw_f) / excess * 100.0
                             if np.isfinite(ade_iw_f) else np.nan)
                rec_f = (float(np.clip(rec_raw_f, 0.0, 100.0))
                         if np.isfinite(rec_raw_f) else np.nan)
                rows_per_feat.append({
                    "year": year, "feature": f_name,
                    "ade_iw": round(ade_iw_f, 4) if np.isfinite(ade_iw_f) else np.nan,
                    "ess_ratio": round(ess_f, 4),
                    "recovery_pct": round(rec_f, 2) if np.isfinite(rec_f) else np.nan,
                    "recovery_raw": round(rec_raw_f, 2) if np.isfinite(rec_raw_f) else np.nan,
                })
            print(f"  [per-feature] {len([r for r in rows_per_feat if r['year']==year])} "
                  f"linhas adicionadas")

    if not rows_iw:
        print("[err] nenhum ano processado.")
        sys.exit(1)

    df_iw = pd.DataFrame(rows_iw)
    iw_path = OUT_DIR / "iw_decomposition.csv"
    df_iw.to_csv(iw_path, index=False)
    # também salva cópia nomeada pela variante para comparação direta
    variant_path = OUT_DIR / f"iw_decomposition_{args.lsif_variant}.csv"
    df_iw.to_csv(variant_path, index=False)
    print(f"\n[ok] {iw_path}")
    print(f"[ok] {variant_path}")
    print(df_iw[["year", "recovery_pct", "recovery_ci_lo", "recovery_ci_hi",
                  "recovery_ci_lo_block", "recovery_ci_hi_block",
                  "ess_ratio", "lsif_variant"]].to_string(index=False))

    # classifier test summary CSV
    if args.classifier_test and "clf_auc" in df_iw.columns:
        clf_path = OUT_DIR / "classifier_two_sample_test.csv"
        df_clf = df_iw[["year", "clf_auc", "clf_coverage", "clf_pvalue"]].copy()
        df_clf.to_csv(clf_path, index=False)
        print(f"\n[ok] {clf_path}")
        print(df_clf.to_string(index=False))

    # compatibilidade legado
    legacy = df_iw[["year","ade_y","ade_iw","ess_ratio","n_source","n_target"]].copy()
    legacy.columns = ["year","ade_target","ade_weighted","ess_ratio","n_source","n_target"]
    legacy["ade_unweighted"] = ade_2019_mean
    legacy["ratio"] = legacy["ade_weighted"] / ade_2019_mean
    legacy.to_csv(OUT_DIR / "importance_weighted_ade.csv", index=False)

    # IW marginal por feature
    if rows_per_feat:
        df_pf = pd.DataFrame(rows_per_feat)
        pf_path = OUT_DIR / "iw_per_feature.csv"
        df_pf.to_csv(pf_path, index=False)
        print(f"\n[ok] {pf_path}  ({len(df_pf)} linhas)")
        print(df_pf.to_string(index=False))

    # -- Ponto 8a: matriz de correlacao Spearman entre features ---------------
    if args.per_feature:
        from scipy.stats import spearmanr
        FEAT_NAMES = ["speed_mean", "speed_p90", "speed_p99",
                      "accel_mean", "accel_p90", "accel_p99",
                      "turn_mean",  "turn_p90"]
        print("\n[info] Calculando matriz de correlacao Spearman (features x anos alvo)...")
        all_feat_rows = []
        for year in sorted(args.target_years):
            tgt_rows_raw = df_all[df_all["year"] == year].reset_index(drop=True)
            if tgt_rows_raw.empty:
                continue
            x_tgt_raw2, _ = extract_features(tgt_rows_raw, loader)
            for row_feat in x_tgt_raw2:
                all_feat_rows.append(row_feat)
        X_pool = np.array(all_feat_rows)
        corr_matrix, _ = spearmanr(X_pool)
        corr_matrix = np.array(corr_matrix)
        rows_corr = []
        for i, fi in enumerate(FEAT_NAMES):
            for j, fj in enumerate(FEAT_NAMES):
                rows_corr.append({"feature_a": fi, "feature_b": fj,
                                  "spearman_r": round(float(corr_matrix[i, j]), 4)})
        df_corr = pd.DataFrame(rows_corr)
        corr_path = OUT_DIR / "iw_feature_corr.csv"
        df_corr.to_csv(corr_path, index=False)
        print(f"[ok] {corr_path}")
        pivot = df_corr.pivot(index="feature_a", columns="feature_b", values="spearman_r")
        print(pivot.to_string())

    # -- Ponto 8b: leave-one-out de features ----------------------------------
    if args.per_feature:
        FEAT_NAMES = ["speed_mean", "speed_p90", "speed_p99",
                      "accel_mean", "accel_p90", "accel_p99",
                      "turn_mean",  "turn_p90"]
        print("\n[info] LOO de features (LSIF com cada feature excluida)...")
        rows_loo: list[dict] = []
        for year in sorted(args.target_years):
            tgt_rows_raw = df_all[df_all["year"] == year].reset_index(drop=True)
            if tgt_rows_raw.empty:
                continue
            x_tgt_raw3, tgt_rows_y3 = extract_features(tgt_rows_raw, loader)
            ade_y_loo = tgt_rows_y3["ade_traj"].astype(float).to_numpy()
            ade_y_mean_loo = float(np.mean(ade_y_loo))
            excess_loo = ade_y_mean_loo - ade_2019_mean
            if excess_loo <= 1e-6:
                continue
            scaler_loo = StandardScaler().fit(np.vstack([x_src_raw, x_tgt_raw3]))
            xs_loo = scaler_loo.transform(x_tgt_raw3)
            xt_loo = scaler_loo.transform(x_src_raw)
            for leave_out_idx, leave_out_name in enumerate(FEAT_NAMES):
                keep = [i for i in range(len(FEAT_NAMES)) if i != leave_out_idx]
                w_loo = lsif_weights(xs_loo[:, keep], xt_loo[:, keep],
                                     n_centers=args.n_centers,
                                     reg=args.reg, seed=args.seed)
                ess_loo = ess_ratio(w_loo)
                ade_iw_loo = (float((w_loo @ ade_y_loo) / w_loo.sum())
                              if w_loo.sum() > 0 else np.nan)
                rec_raw_loo = ((ade_y_mean_loo - ade_iw_loo) / excess_loo * 100.0
                               if np.isfinite(ade_iw_loo) else np.nan)
                rec_loo = (float(np.clip(rec_raw_loo, 0.0, 100.0))
                           if np.isfinite(rec_raw_loo) else np.nan)
                rows_loo.append({
                    "year":         year,
                    "excluded":     leave_out_name,
                    "ade_iw":       round(ade_iw_loo, 4) if np.isfinite(ade_iw_loo) else np.nan,
                    "ess_ratio":    round(ess_loo, 4),
                    "recovery_pct": round(rec_loo, 2) if np.isfinite(rec_loo) else np.nan,
                    "recovery_raw": round(rec_raw_loo, 2) if np.isfinite(rec_raw_loo) else np.nan,
                })
            print(f"  ano {year}: {len(FEAT_NAMES)} combinacoes LOO calculadas")

        if rows_loo:
            df_loo = pd.DataFrame(rows_loo)
            loo_path = OUT_DIR / "iw_loo_analysis.csv"
            df_loo.to_csv(loo_path, index=False)
            print(f"[ok] {loo_path}  ({len(df_loo)} linhas)")
            print(df_loo.to_string(index=False))

    # -- Ponto 9: self_test -- calibracao LSIF com source == target -----------
    if args.self_test:
        print("\n" + "=" * 60)
        print("[self_test] LSIF com source = target = 2019")
        print("Esperado: pesos ~1, ADE_IW ~ADE_2019, recovery ~0%")
        print("=" * 60)
        rng_st = np.random.default_rng(args.seed + 9999)
        n_st = len(x_src_raw)
        idx_perm = rng_st.permutation(n_st)
        half = n_st // 2
        idx_a, idx_b = idx_perm[:half], idx_perm[half:]
        xa_st = x_src_raw[idx_a]
        xb_st = x_src_raw[idx_b]
        ade_a_st = ade_2019_all[idx_a]
        scaler_st = StandardScaler().fit(np.vstack([xa_st, xb_st]))
        xs_st = scaler_st.transform(xa_st)
        xt_st = scaler_st.transform(xb_st)
        w_st  = lsif_weights(xs_st, xt_st, n_centers=min(args.n_centers, half),
                             reg=args.reg, seed=args.seed)
        ess_st = ess_ratio(w_st)
        sw_st  = w_st.sum()
        ade_iw_st = float((w_st @ ade_a_st) / sw_st) if sw_st > 0 else np.nan
        rec_st = ((ade_2019_mean - ade_iw_st) / max(abs(ade_2019_mean), 1e-9) * 100
                  if np.isfinite(ade_iw_st) else np.nan)
        print(f"  n_half         = {half}")
        print(f"  ADE_2019 full  = {ade_2019_mean:.4f}")
        print(f"  ADE_IW (self)  = {ade_iw_st:.4f}")
        print(f"  Delta          = {ade_iw_st - ade_2019_mean:+.4f}  (esperado ~0)")
        print(f"  ESS ratio      = {ess_st:.4f}")
        print(f"  Recovery proxy = {rec_st:+.2f}%  (esperado ~0)")
        calibrated = abs(ade_iw_st - ade_2019_mean) < 0.5 and ess_st >= ESS_MARGINAL_THRESHOLD
        print(f"  Calibracao OK? {'SIM' if calibrated else 'NAO -- verificar LSIF'}")
        st_path = OUT_DIR / "iw_self_test.csv"
        pd.DataFrame([{
            "n_half":             half,
            "ade_2019":           round(ade_2019_mean, 4),
            "ade_iw_self":        round(ade_iw_st, 4) if np.isfinite(ade_iw_st) else float("nan"),
            "delta_mm":           round(ade_iw_st - ade_2019_mean, 4) if np.isfinite(ade_iw_st) else float("nan"),
            "ess_ratio":          round(ess_st, 4),
            "recovery_proxy_pct": round(rec_st, 2) if np.isfinite(rec_st) else float("nan"),
            "calibrated":         calibrated,
            "seed":               args.seed,
        }]).to_csv(st_path, index=False)
        print(f"[ok] {st_path}")



# --- IW figures -------------------------------------------------------------

def load_iw() -> pd.DataFrame:
    if not IW_CSV.exists():
        raise FileNotFoundError(f"Execute main_compute_importance_weights() primeiro: {IW_CSV}")
    return pd.read_csv(IW_CSV)


def plot_ade_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    years = df["year"].astype(int).tolist()
    x = np.arange(len(years))
    w = 0.30

    ax.bar(x - w / 2, df["ade_y"], width=w, color=COLORS["ade_y"],
           alpha=0.85, label="ADE direto (ano y)")
    ax.errorbar(x - w / 2, df["ade_y"],
                yerr=[df["ade_y"] - df["ade_y_ci_lo"],
                      df["ade_y_ci_hi"] - df["ade_y"]],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    iw_vals = df["ade_iw"].where(df["ess_stable"], np.nan)
    iw_lo = (df["ade_iw"] - df["ade_iw_ci_lo"]).where(df["ess_stable"], np.nan)
    iw_hi = (df["ade_iw_ci_hi"] - df["ade_iw"]).where(df["ess_stable"], np.nan)
    ax.bar(x + w / 2, iw_vals, width=w, color=COLORS["ade_iw"],
           alpha=0.85, label="ADE importance-weighted")
    ax.errorbar(x + w / 2, iw_vals,
                yerr=[iw_lo.fillna(0), iw_hi.fillna(0)],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    b = df["ade_2019"].iloc[0]
    ax.axhline(b, color=COLORS["ade_2019"], linestyle="--", linewidth=1.5,
               label=f"ADE 2019 = {b:.2f} mm")

    for i, row in enumerate(df.itertuples()):
        if not row.ess_stable:
            ax.text(i + w / 2, 0.2, f"ESS={row.ess_ratio:.2f}\n⚠",
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
    fig, ax = plt.subplots(figsize=(7, 4.5))
    years = df["year"].astype(int).tolist()
    x = np.arange(len(years))
    rec = df["recovery_pct"].fillna(0)
    colors = ["#2ca02c" if (r > 20 and ess) else "#d62728" if not ess else "#aec7e8"
              for r, ess in zip(rec, df["ess_stable"])]
    bars = ax.bar(x, rec, color=colors, alpha=0.85, label="recovery clipped [0,100]")

    for bar, ess, rv in zip(bars, df["ess_stable"], df["recovery_pct"]):
        label = f"{rv:.1f}%" if pd.notna(rv) else "n/a"
        if not ess:
            label += "\n⚠ESS"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                label, ha="center", va="bottom", fontsize=9)

    if "recovery_raw" in df.columns:
        raw_vals = df["recovery_raw"].to_numpy(dtype=float)
        mask = np.isfinite(raw_vals)
        if mask.any():
            ax.scatter(x[mask], raw_vals[mask], marker="D", color="black",
                       s=50, zorder=5, label="recovery_raw (sem clip)",
                       linewidths=0)
            for xi, rv_raw, rv_clip in zip(x[mask], raw_vals[mask], rec.to_numpy()[mask]):
                if abs(rv_raw - rv_clip) > 1.0:
                    ax.plot([xi, xi], [rv_clip, rv_raw], color="black",
                            linewidth=0.8, linestyle=":")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--", label="100% recovery")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Recovery (% excesso de ADE eliminado)")
    ax.set_title("Recovery por importance-weighting\n"
                 "(barra = clipped; ◆ = raw sem clip; verde = >20%)", fontsize=11)
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
        ess = ((w.sum()) ** 2) / (len(w) * (w ** 2).sum())
        ax.set_title(f"{year}  ESS={ess:.2f}", fontsize=10)
        ax.set_xlabel("peso w")
        ax.set_ylabel("contagem" if year == targets[0] else "")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8)
    fig.suptitle("Distribuicao dos pesos p_2019(x)/p_y(x) por ano", fontsize=11)
    plt.tight_layout()
    path = OUT_DIR / "iw_weights_dist.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def run_iw_figures() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_iw()
    print(f"[info] {len(df)} anos no iw_decomposition.csv")
    plot_ade_comparison(df)
    plot_recovery(df)
    plot_weights_dist(df)
    print("[ok] plots IW gerados.")


# --- Division validation ----------------------------------------------------


def load_division_map() -> pd.DataFrame:
    if not DIV_MAP.exists():
        raise FileNotFoundError(f"Arquivo de divisão não encontrado: {DIV_MAP}")
    return pd.read_csv(DIV_MAP)


def build_ade_by_game_division(df: pd.DataFrame, div_map: pd.DataFrame) -> pd.DataFrame:
    df_s = df[(df["model"] == "Seq2Seq") & (df["horizon"] == HORIZON)].copy()
    grp = (
        df_s.groupby(["proc_set_file", "year"])["ade_traj"]
            .mean().reset_index()
            .rename(columns={"ade_traj": "ade_mean"})
    )
    merged = grp.merge(div_map, on="proc_set_file", how="left")
    merged["division"] = merged["division"].fillna("Unknown")
    return merged


def ade_ci_by_year_division(df_game: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, div), grp in df_game.groupby(["year", "division"]):
        vals = grp["ade_mean"].to_numpy()
        n_games = len(vals)
        if n_games < MIN_GAMES:
            rows.append({"year": year, "division": div,
                         "ade_mean": float(np.mean(vals)),
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "n_games": n_games, "insufficient": True})
        else:
            # boot_ci antigo = bootstrap_mean_ade com o mesmo RNG; o valor
            # central era a media amostral (nao a media dos bootstraps)
            lo, _, hi = bootstrap_mean_ade(vals)
            rows.append({"year": year, "division": div,
                         "ade_mean": float(vals.mean()), "ci_lo": lo, "ci_hi": hi,
                         "n_games": n_games, "insufficient": False})
    return pd.DataFrame(rows).sort_values(["division", "year"])


def plot_ade_by_division(df_ci: pd.DataFrame, out_path: Path) -> None:
    divisions = sorted(df_ci["division"].unique())
    n_divs = len(divisions)
    fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 4.5), sharey=True)
    if n_divs == 1:
        axes = [axes]
    for ax, div in zip(axes, divisions):
        sub = df_ci[df_ci["division"] == div].copy()
        for _, row in sub.iterrows():
            color = COLOR_YEAR.get(int(row["year"]), "gray")
            alpha = 0.4 if row["insufficient"] else 1.0
            ax.errorbar(
                row["year"], row["ade_mean"],
                yerr=[[row["ade_mean"] - row["ci_lo"]],
                      [row["ci_hi"] - row["ade_mean"]]]
                     if not row["insufficient"] else None,
                fmt="o", color=color, alpha=alpha, capsize=4,
            )
        ax.set_title(f"Division {div}", fontsize=12)
        ax.set_xlabel("Ano")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    axes[0].set_ylabel("ADE médio (mm)")
    fig.suptitle("ADE por Ano × Divisão — Seq2Seq 30→15", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def table_2x2(df_ci: pd.DataFrame, cutoff: int = 2022) -> pd.DataFrame:
    rows = []
    for div in sorted(df_ci["division"].unique()):
        sub = df_ci[df_ci["division"] == div]
        before = sub[sub["year"] < cutoff]["ade_mean"].mean()
        after = sub[sub["year"] >= cutoff]["ade_mean"].mean()
        rows.append({"division": div,
                     f"ade_antes_{cutoff}": round(before, 3),
                     f"ade_apos_{cutoff}": round(after, 3),
                     "delta": round(after - before, 3)})
    return pd.DataFrame(rows)


def decomp_by_division(df_ci: pd.DataFrame) -> pd.DataFrame:
    baseline = df_ci[df_ci["year"] == BASELINE].set_index("division")["ade_mean"]
    rows = []
    for div in sorted(df_ci["division"].unique()):
        sub = df_ci[(df_ci["division"] == div) & (df_ci["year"] > BASELINE)]
        ade_2019_div = baseline.get(div, np.nan)
        for _, r in sub.iterrows():
            excess = r["ade_mean"] - ade_2019_div
            rows.append({
                "division": div,
                "year": int(r["year"]),
                "ade_2019": round(ade_2019_div, 4) if pd.notna(ade_2019_div) else np.nan,
                "ade_y": round(r["ade_mean"], 4),
                "excess_mm": round(excess, 4) if pd.notna(ade_2019_div) else np.nan,
                "n_games": int(r["n_games"]),
                "insufficient": bool(r["insufficient"]),
            })
    return pd.DataFrame(rows)


def run_division_validation() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_drift_decomposition_csv()

    df = load_errors()
    div_map = load_division_map()
    print(f"[info] {len(df):,} trajetórias  |  "
          f"divisões mapeadas: {div_map['division'].value_counts().to_dict()}")

    df_game = build_ade_by_game_division(df, div_map)
    df_ci = ade_ci_by_year_division(df_game)

    plot_ade_by_division(df_ci, OUT_DIR / "by_division_ade_per_game.pdf")

    df_decomp = decomp_by_division(df_ci)
    decomp_path = OUT_DIR / "by_division_decomposition.csv"
    df_decomp.to_csv(decomp_path, index=False)
    print(f"[ok] {decomp_path}")
    print(df_decomp.to_string(index=False))

    df_2x2 = table_2x2(df_ci, cutoff=2022)
    path_2x2 = OUT_DIR / "by_division_2x2_table.csv"
    df_2x2.to_csv(path_2x2, index=False)
    print(f"\n[ok] {path_2x2}")
    print(df_2x2.to_string(index=False))

    print("\n=== CONCLUSÃO ===")
    divs = sorted(df_ci[~df_ci["insufficient"]]["division"].unique())
    for div in divs:
        sub = df_ci[(df_ci["division"] == div) & ~df_ci["insufficient"]]
        if len(sub) < 2:
            print(f"  Division {div}: amostra insuficiente — não conclusivo.")
            continue
        ade_2019 = sub[sub["year"] == BASELINE]["ade_mean"].values
        ade_post = sub[sub["year"] > BASELINE]["ade_mean"].values
        if len(ade_2019) == 0 or len(ade_post) == 0:
            print(f"  Division {div}: sem baseline ou pós-baseline — não conclusivo.")
            continue
        delta = ade_post.mean() - ade_2019.mean()
        survives = "SIM" if delta > 0.5 else "NÃO" if delta < -0.5 else "PARCIAL/INCERTO"
        print(f"  Division {div}: delta={delta:+.2f} mm → drift sobrevive ao corte? {survives}")

    if len(divs) < 2:
        print("\n[nota] Division B ausente ou insuficiente nos dados — "
              "análise limitada a Division A.")


def run_chapter03_visuals() -> None:
    """IW plots + divisão (ordem usada no notebook 3)."""
    run_iw_figures()
    run_division_validation()


if __name__ == "__main__":
    run_chapter03_visuals()
