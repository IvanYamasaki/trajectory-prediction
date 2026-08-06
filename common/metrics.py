"""Shared statistical metrics (KS, Wasserstein, bootstrap CIs, Wilson CI).

Canonical sources: drift_analise/chapter01_descriptive_pipeline.py,
drift_analise/mes10_independence.py.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

N_BOOT = 2000
SEED = 42
CI = 95


def wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    n = 2000
    qs = np.linspace(0, 1, n)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def ks_stat_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    allx = np.sort(np.unique(np.concatenate([a, b])))
    Fa = np.searchsorted(a, allx, side="right") / a.size
    Fb = np.searchsorted(b, allx, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))


def bootstrap_ci_mean(values, n_boot=N_BOOT, ci=CI, seed=SEED):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = vals.size
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = np.mean(vals[rng.integers(0, n, size=n)])
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def bootstrap_linreg_ci(x, y, x_grid=None, n_boot=N_BOOT, ci=CI, seed=SEED):
    """Retorna dict com slope, intercept, mean_pred, ci_lo, ci_hi sobre x_grid."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return None
    if x_grid is None:
        x_grid = np.linspace(np.min(x), np.max(x), 120)
    rng = np.random.default_rng(seed)
    n = x.size
    preds = np.empty((n_boot, x_grid.size))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sx, sy = x[idx], y[idx]
        if np.std(sx) < 1e-12:
            preds[b] = np.nan
            continue
        slope, intercept, *_ = stats.linregress(sx, sy)
        preds[b] = slope * x_grid + intercept
    mean_pred = np.nanmean(preds, axis=0)
    lo = np.nanpercentile(preds, (100 - ci) / 2, axis=0)
    hi = np.nanpercentile(preds, 100 - (100 - ci) / 2, axis=0)
    return {"x_grid": x_grid, "mean": mean_pred, "lo": lo, "hi": hi}


def wilson_ci(k: int, n: float, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para proporcao k/n; retorna (lo, hi) por mil."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half) * 1000.0, (center + half) * 1000.0)
