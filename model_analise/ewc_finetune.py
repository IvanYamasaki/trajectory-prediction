"""
ewc_finetune.py — Elastic Weight Consolidation (Kirkpatrick 2017) para Seq2Seq
==============================================================================
Implementa EWC sobre o fine-tuning do Mês 3, adicionando regularização L2
ponderada pela diagonal de Fisher para reduzir catastrophic forgetting.

Equação:  L_EWC = L_MSE + λ/2 · Σ_i F_i · (θ_i − θ*_i)²

Uso
---
    python model_analise/ewc_finetune.py
    python model_analise/ewc_finetune.py --ewc_lambda 100 --n_unfreeze 2 \\
        --out Relas/results/mes7/ewc_results.csv
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bootstrap import init_project

PROJECT_ROOT = init_project()

from common.constants import PROC_YEAR, YEAR_ORDER, BP_YEAR
from model_analise.core.data import load_proc_sets
from model_analise.core.metrics import (
    cat_ratio_fraction, compute_ade, original_per_traj_dist, recovery_pct,
)
from model_analise.core.model_io import (
    build_seq2seq, freeze_encoder, load_loader as _core_load_loader,
)

LOOK_BACK  = 30
LOOK_FORTH = 15
UNITS      = 128
WEIGHTS_IN = Path("weights") / "robot_30_15_t.weights.h5"
STATS_PATH = Path("model") / "normalization_stats_30_15.pkl"
OUT_DIR    = Path("Relas") / "results" / "mes7"


def load_loader():
    return _core_load_loader(LOOK_BACK, LOOK_FORTH, stats_path=STATS_PATH)


def build_model(tf):
    return build_seq2seq(LOOK_BACK, LOOK_FORTH, units=UNITS, weights=WEIGHTS_IN)


def compute_fisher_diagonal(model, tf, x_fisher: np.ndarray,
                             y_fisher: np.ndarray | None = None,
                             batch_size: int = 256, n_samples: int = 2000,
                             sigma: float = 1.0,
                             ) -> list[np.ndarray]:
    """
    Estimate the empirical Fisher Information diagonal.

    F_i = (1/N) Σ_n ( ∂ log p(y_n | x_n; θ) / ∂θ_i )²

    Under a Gaussian observation model y ~ N(f_θ(x), σ²I):
        log p(y|x; θ) = -||y - f_θ(x)||² / (2 σ²) + const

    Two estimators are supported:
      * Observed Fisher  (default when `y_fisher` is given): uses the true
        targets y_n.  Recommended for EWC because it reflects which params
        actually mattered to fit 2019.
      * Expected Fisher  (fallback when y_fisher is None): samples
        y_n ~ N(f_θ(x_n), σ²I) with the sampled y detached from the graph,
        so the gradient flows only through f_θ(x_n).

    Bug fixed (2026-05): the previous version computed
        log_lik = -||y_pred - (y_pred + ε)||² = -||ε||²,
    whose gradient w.r.t. θ is identically zero, producing F_i ≈ 0
    independent of the data.  This is what made EWC look inert across
    λ ∈ {0, 50, 500} in the Mês 7 runs.
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_fisher), min(n_samples, len(x_fisher)), replace=False)
    x_sub = tf.constant(x_fisher[idx])
    use_observed = y_fisher is not None
    if use_observed:
        y_sub = tf.constant(y_fisher[idx])

    # Initialise accumulators
    fisher = [np.zeros(v.shape, dtype=np.float32)
              for v in model.trainable_variables]
    n_batches = 0
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    for start in range(0, len(x_sub), batch_size):
        x_b = x_sub[start:start + batch_size]
        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=False)
            if use_observed:
                y_target = y_sub[start:start + batch_size]
            else:
                # detach sampled target so it does not flow into the gradient
                noise   = tf.random.normal(tf.shape(y_pred), stddev=sigma)
                y_target = tf.stop_gradient(y_pred + noise)
            log_lik = -inv_2s2 * tf.reduce_mean(tf.square(y_target - y_pred))
        grads = tape.gradient(log_lik, model.trainable_variables)
        for i, g in enumerate(grads):
            if g is not None:
                fisher[i] += g.numpy() ** 2
        n_batches += 1

    fisher = [f / max(n_batches, 1) for f in fisher]
    total_params = sum(np.prod(f.shape) for f in fisher)
    mean_f = float(np.mean([f.mean() for f in fisher]))
    max_f  = float(np.max( [f.max()  for f in fisher]))
    kind   = "observed" if use_observed else "expected"
    print(f"  [fisher] {total_params:,} params, mean diag={mean_f:.3e}, "
          f"max diag={max_f:.3e}  ({kind} Fisher, sigma={sigma})")
    return fisher


class EWCModel:
    """Wraps a Keras model with EWC penalty in the training loop."""

    def __init__(self, model, fisher_diag: list, theta_star: list, ewc_lambda: float):
        self.model     = model
        self.fisher    = fisher_diag
        self.theta_star = theta_star
        self.lam       = ewc_lambda

    def ewc_loss(self, tf, y_true, y_pred) -> "tf.Tensor":
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        penalty = tf.constant(0.0)
        for F, th_star, th in zip(self.fisher, self.theta_star,
                                   self.model.trainable_variables):
            F_t  = tf.constant(F, dtype=tf.float32)
            diff = th - tf.constant(th_star, dtype=tf.float32)
            penalty = penalty + tf.reduce_sum(F_t * tf.square(diff))
        return mse + (self.lam / 2.0) * penalty

    def train_step_ewc(self, tf, optimizer, x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss   = self.ewc_loss(tf, y_batch, y_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss)


def train_ewc(ewc_wrap, tf, x_train, y_train, optimizer,
              epochs: int, batch_size: int = 256) -> list[float]:
    N = len(x_train)
    losses = []
    for ep in range(epochs):
        perm = np.random.permutation(N)
        ep_loss = []
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            x_b = tf.constant(x_train[idx])
            y_b = tf.constant(y_train[idx])
            l   = ewc_wrap.train_step_ewc(tf, optimizer, x_b, y_b)
            ep_loss.append(l)
        mean_loss = float(np.mean(ep_loss))
        losses.append(mean_loss)
        print(f"    epoch {ep+1}/{epochs}  loss={mean_loss:.6f}")
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewc_lambda", type=float, default=100.0,
                    help="EWC regularization strength (0 = no EWC = naive fine-tuning)")
    ap.add_argument("--n_unfreeze", type=int,   default=2)
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch",      type=int,   default=256)
    ap.add_argument("--max_per",    type=int,   default=500,
                    help="Max trajectories sampled per proc_set (default 500)")
    ap.add_argument("--out",        default=str(OUT_DIR / "ewc_results.csv"))
    ap.add_argument("--lambdas",    type=float, nargs="+",
                    default=None,
                    help="If set, sweep over multiple lambda values")
    args = ap.parse_args()

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    print(f"[info] TF {tf.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = load_loader()

    # Proc-sets before and after breakpoint
    bp_idx   = YEAR_ORDER.index(BP_YEAR)
    before_ps = [n for n, y in PROC_YEAR.items()
                 if YEAR_ORDER.index(y) < bp_idx]
    after_ps  = [n for n, y in PROC_YEAR.items()
                 if YEAR_ORDER.index(y) >= bp_idx]

    print(f"[info] BP year={BP_YEAR}: {len(before_ps)} train proc_sets, "
          f"{len(after_ps)} test proc_sets")

    # Load datasets
    x_base, y_base = load_proc_sets(sorted(before_ps), loader, max_per=args.max_per)
    x_test, y_test = load_proc_sets(sorted(after_ps),  loader, max_per=args.max_per)
    if x_base is None or x_test is None:
        sys.exit("[error] Could not load datasets. Check proc_set pickle files.")

    print(f"[info] baseline (2019): {len(x_base):,} traj | "
          f"test (post-{BP_YEAR}): {len(x_test):,} traj")

    lambdas = args.lambdas if args.lambdas else [0.0, args.ewc_lambda]
    rows = []

    # Modelo com os pesos originais (invariante ao longo do sweep de lambda;
    # construído UMA vez, fora do loop — nada abaixo o modifica).
    model_orig = build_model(tf)

    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"[run] EWC lambda={lam}  lr={args.lr}  n_unfreeze={args.n_unfreeze}")
        model = build_model(tf)
        freeze_encoder(model, args.n_unfreeze)
        optimizer = tf.keras.optimizers.Adam(args.lr)

        ade_before_base = compute_ade(model, x_base, y_base, loader)
        ade_before_test = compute_ade(model, x_test, y_test, loader)
        print(f"  ADE before: base={ade_before_base:.4f}  test={ade_before_test:.4f}")

        # Reference weights (theta*)
        theta_star = [v.numpy().copy() for v in model.trainable_variables]

        # Fisher diagonal on baseline (2019) data — observed Fisher with
        # real targets y_base (Kirkpatrick 2017 §3, eq. 3).
        print(f"  Computing Fisher diagonal on baseline data...")
        fisher_diag = compute_fisher_diagonal(
            model, tf, x_base, y_fisher=y_base,
            n_samples=min(1000, len(x_base)),
            sigma=1.0,
        )

        # Fine-tune with EWC
        ewc_wrap = EWCModel(model, fisher_diag, theta_star, ewc_lambda=lam)
        print(f"  Training with EWC lambda={lam}...")
        epoch_losses = train_ewc(ewc_wrap, tf, x_test, y_test, optimizer,
                                  epochs=args.epochs, batch_size=args.batch)

        ade_after_base = compute_ade(model, x_base, y_base, loader)
        ade_after_test = compute_ade(model, x_test, y_test, loader)

        # Per-trajectory degradation on baseline (catastrophic forgetting)
        d_after, _ = original_per_traj_dist(model, x_base, y_base, loader)
        # Per-traj ADE with original weights
        d_orig, _  = original_per_traj_dist(model_orig, x_base, y_base, loader)

        cat_ratio = cat_ratio_fraction(d_orig, d_after)

        rec_pct = recovery_pct(ade_before_base, ade_before_test, ade_after_test)

        print(f"\n  Results:")
        print(f"    ADE baseline: {ade_before_base:.4f} -> {ade_after_base:.4f} "
              f"(delta={ade_after_base-ade_before_base:+.4f})")
        print(f"    ADE test:     {ade_before_test:.4f} -> {ade_after_test:.4f} "
              f"(delta={ade_after_test-ade_before_test:+.4f})")
        print(f"    cat_ratio:    {cat_ratio:.3f} (target < 0.5; "
              f"naive FT Mes3 = 0.82)")
        print(f"    recovery_pct: {rec_pct:.1f}%")

        rows.append({
            "ewc_lambda":       lam,
            "n_unfreeze":       args.n_unfreeze,
            "lr":               args.lr,
            "epochs":           args.epochs,
            "ade_before_base":  round(ade_before_base, 4),
            "ade_after_base":   round(ade_after_base, 4),
            "ade_before_test":  round(ade_before_test, 4),
            "ade_after_test":   round(ade_after_test, 4),
            "cat_ratio":        round(cat_ratio, 4),
            "recovery_pct":     round(rec_pct, 1),
            "final_loss":       round(epoch_losses[-1], 6),
        })

        # Save EWC weights
        wpath = Path("weights") / f"robot_30_15_ewc_lam{int(lam)}.weights.h5"
        model.save_weights(str(wpath))
        print(f"  [ok] weights -> {wpath}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    print(df[["ewc_lambda", "ade_after_base", "ade_after_test",
              "cat_ratio", "recovery_pct"]].to_string(index=False))

    # Summary
    best = df.loc[df["cat_ratio"].idxmin()]
    print(f"\n*** Best EWC lambda={best['ewc_lambda']}: "
          f"cat_ratio={best['cat_ratio']:.3f} vs naive FT 0.82 ***")
    if best["cat_ratio"] < 0.5:
        print("[CRITERIO ATINGIDO] cat_ratio < 0.5 com EWC")
    else:
        print("[CRITERIO NAO ATINGIDO] cat_ratio >= 0.5 — "
              "decoder pequeno, Fisher degenerado (ver plano Mes7)")


if __name__ == "__main__":
    main()
