"""
compute_importance_weights.py  —  Frente 2 do Mês 3
====================================================

Estima pesos de density-ratio (LSIF) para trajetórias de CADA ano alvo
versus o baseline 2019, e computa o ADE importance-weighted.

Pergunta central:
    O ADE de ano y, ponderado por w(x) = p_2019(x) / p_y(x), se aproxima
    do ADE de 2019? Se sim, o excesso de erro é explicado por covariate shift.

Fórmula:
    w_i   = p_2019(x_i) / p_y(x_i)   para i em year-y
    ADE_IW = sum(w_i * ade_yi) / sum(w_i)
    recovery_pct = (ade_y - ADE_IW) / (ade_y - ade_2019) * 100
    ESS_ratio = (sum w_i)^2 / (n * sum w_i^2)   (0-1; < 0.3 = instável)

Saídas:
    covariate_shift_out/importance_weights_{year}.npy  (pesos por trajetória-alvo)
    Relas/results/mes3/iw_decomposition.csv
    Relas/results/mes3/importance_weighted_ade.csv  (compatibilidade legado)

Uso:
    python drift_analise/compute_importance_weights.py
    python drift_analise/compute_importance_weights.py --target_years 2021 2023 2025
    python drift_analise/compute_importance_weights.py --n_boot 2000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dataset.load_dataset import LoadDataSet

COV_DIR      = Path("covariate_shift_out")
OUT_DIR      = Path("Relas") / "results" / "mes3"
BASELINE_YEAR = 2019
ALL_TARGETS  = [2021, 2022, 2023, 2024, 2025]
HORIZON      = "30→15"
LOOK_BACK    = 30
LOOK_FORTH   = 15
STATS_PATH   = PROJECT_ROOT / "model" / "normalization_stats_30_15.pkl"


# ─────────────────────────── loaders ──────────────────────────────────────

def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Execute model_analise/compute_trajectory_errors.py primeiro.")


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


# ─────────────────────────── main ─────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_years", type=int, nargs="+", default=ALL_TARGETS)
    ap.add_argument("--n_centers",    type=int,   default=300)
    ap.add_argument("--reg",          type=float, default=1e-3)
    ap.add_argument("--n_boot",       type=int,   default=2000)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--per_feature",  action="store_true",
                    help="Roda LSIF marginal em cada feature isoladamente "
                         "para identificar a dimensão dominante.")
    args = ap.parse_args()

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
    ade_2019_all = src_rows["ade_traj"].astype(float).to_numpy()
    ade_2019_mean = float(np.mean(ade_2019_all))
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
        ade_y = tgt_rows["ade_traj"].astype(float).to_numpy()
        ade_y_mean = float(np.mean(ade_y))

        scaler = StandardScaler().fit(np.vstack([x_src_raw, x_tgt_raw]))
        xs = scaler.transform(x_tgt_raw)   # fonte = ano y
        xt = scaler.transform(x_src_raw)   # target = 2019

        # w_i = p_2019(x_i) / p_y(x_i)  para x_i em y
        w = lsif_weights(xs, xt, n_centers=args.n_centers, reg=args.reg, seed=args.seed)

        ess = ess_ratio(w)
        stable = ess >= 0.3
        status = "ok" if stable else "INSTÁVEL (ESS < 0.3)"
        print(f"  ESS ratio = {ess:.3f}  [{status}]")

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
                    rb = (m_y - m_iw_b) / ex_b * 100.0
                    recov_boots[b] = float(np.clip(rb, 0.0, 100.0))
                else:
                    recov_boots[b] = np.nan
            valid = recov_boots[np.isfinite(recov_boots)]
            recovery_ci_lo = float(np.percentile(valid, 2.5)) if valid.size else np.nan
            recovery_ci_hi = float(np.percentile(valid, 97.5)) if valid.size else np.nan
        else:
            recovery_ci_lo = recovery_ci_hi = np.nan

        print(f"  ADE_y={ade_y_mean:.4f}  ADE_IW={ade_iw:.4f}  "
              f"ADE_2019={ade_2019_mean:.4f}  "
              f"recovery={recovery if np.isfinite(recovery) else 'n/a':>5}%  "
              f"[{recovery_note}]")

        np.save(COV_DIR / f"importance_weights_{year}.npy", w)

        rows_iw.append({
            "year":           year,
            "ade_y":          round(ade_y_mean, 4),
            "ade_y_ci_lo":    round(ci_lo_y, 4),
            "ade_y_ci_hi":    round(ci_hi_y, 4),
            "ade_2019":       round(ade_2019_mean, 4),
            "ade_iw":         round(ade_iw, 4) if np.isfinite(ade_iw) else np.nan,
            "ade_iw_ci_lo":   round(ci_lo_iw, 4) if np.isfinite(ci_lo_iw) else np.nan,
            "ade_iw_ci_hi":   round(ci_hi_iw, 4) if np.isfinite(ci_hi_iw) else np.nan,
            "ess_ratio":      round(ess, 4),
            "ess_stable":     stable,
            "recovery_pct":   round(recovery, 2) if np.isfinite(recovery) else np.nan,
            "recovery_ci_lo": round(recovery_ci_lo, 2) if np.isfinite(recovery_ci_lo) else np.nan,
            "recovery_ci_hi": round(recovery_ci_hi, 2) if np.isfinite(recovery_ci_hi) else np.nan,
            "recovery_raw":   round(recovery_raw, 2) if np.isfinite(recovery_raw) else np.nan,
            "overshoot":      overshoot,
            "recovery_note":  recovery_note,
            "excess_total":   round(excess, 4),
            "n_source":       len(x_src_raw),
            "n_target":       len(x_tgt_raw),
        })

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
    print(f"\n[ok] {iw_path}")
    print(df_iw.to_string(index=False))

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


if __name__ == "__main__":
    main()
