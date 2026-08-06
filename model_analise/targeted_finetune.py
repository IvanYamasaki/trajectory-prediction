"""
targeted_finetune.py — Adaptacao dirigida pelo diagnostico IW/accel
===================================================================
Usa o diagnostico do capitulo de covariate shift (accel de cauda domina o
drift; pesos LSIF quantificam relevancia) para DIRIGIR a adaptacao, em vez
de tratar todos os dados igualmente. Protocolo identico ao ewc_finetune /
replay_finetune (max_per=500, lr=1e-5, n_unfreeze=2, 5 epocas, batch 256):

  A) replay_iw      — replay r=0.5 amostrado proporcionalmente a
                      w(x_base) = p_novo(x)/p_antigo(x) (LSIF sobre features
                      cinematicas por janela): o buffer preserva o
                      conhecimento antigo que CONTINUA relevante no regime novo.
  B) target_accel   — fine-tuning apenas no top-50% dos dados novos por
                      aceleracao de cauda da janela (accel_p95): updates
                      concentrados onde p(x) mudou; o corpo inalterado da
                      distribuicao nao gera interferencia.
  C) target+replay  — B com buffer de replay IW (r=0.5 do subconjunto).

Comparaveis diretos (mesmo protocolo): controle r=0 (cat_ratio 0.436),
replay uniforme r=0.5 (0.418), EWC lambda-sweep (0.444-0.452).

Saida: Relas/results/mes7/targeted_results.csv
Uso:  python model_analise/targeted_finetune.py
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bootstrap import init_project

PROJECT_ROOT = init_project()

from model_analise.ewc_finetune import (
    PROC_YEAR, YEAR_ORDER, BP_YEAR, OUT_DIR,
    load_loader, load_proc_sets, build_model, freeze_encoder, compute_ade,
)
from model_analise.core.metrics import (
    cat_ratio_fraction, original_per_traj_dist, recovery_pct,
)
from drift_analise.chapter03_visuals import lsif_weights


# ─────────────────── features cinematicas por janela ──────────────────────

def window_features(x: np.ndarray) -> np.ndarray:
    """Features cinematicas [N, 8] de janelas [N, T, 6] (x,y,vx,vy,sin,cos).

    Mesma familia do pipeline IW (speed/accel/turn; media + caudas),
    computada nas unidades normalizadas do loader — suficiente para a
    razao de densidades, que e invariante a escala comum.
    """
    v = x[:, :, 2:4]                                  # [N, T, 2]
    speed = np.linalg.norm(v, axis=-1)                # [N, T]
    accel = np.linalg.norm(np.diff(v, axis=1), axis=-1)   # [N, T-1]
    psi = np.arctan2(x[:, :, 4], x[:, :, 5])          # [N, T]
    dpsi = np.abs(np.angle(np.exp(1j * np.diff(psi, axis=1))))

    feats = np.stack([
        speed.mean(axis=1), np.percentile(speed, 90, axis=1),
        np.percentile(speed, 99, axis=1),
        accel.mean(axis=1), np.percentile(accel, 90, axis=1),
        np.percentile(accel, 95, axis=1),
        dpsi.mean(axis=1), np.percentile(dpsi, 90, axis=1),
    ], axis=1)
    return np.log1p(feats)


def accel_tail_score(x: np.ndarray) -> np.ndarray:
    """Score de cauda de aceleracao da janela (p95 de ||dv||)."""
    accel = np.linalg.norm(np.diff(x[:, :, 2:4], axis=1), axis=-1)
    return np.percentile(accel, 95, axis=1)


# ────────────────────────────── runner ────────────────────────────────────

def run_variant(tf, loader, tag, x_new, y_new, x_base, y_base,
                d_orig, ade_before_base, ade_before_test,
                x_test_full, y_test_full,
                replay_idx, args) -> dict:
    """Treina uma variante e mede as metricas do protocolo do Mes 7."""
    print(f"\n{'='*60}\n[run] {tag}: {len(x_new):,} novos + "
          f"{len(replay_idx):,} replay")
    model = build_model(tf)
    freeze_encoder(model, args.n_unfreeze)

    x_tr = np.concatenate([x_new, x_base[replay_idx]], axis=0) \
        if len(replay_idx) else x_new
    y_tr = np.concatenate([y_new, y_base[replay_idx]], axis=0) \
        if len(replay_idx) else y_new

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_tr, y_tr, batch_size=args.batch, epochs=args.epochs,
              shuffle=True, verbose=2)

    ade_after_base = compute_ade(model, x_base, y_base, loader)
    ade_after_test = compute_ade(model, x_test_full, y_test_full, loader)

    d_after, _ = original_per_traj_dist(model, x_base, y_base, loader)
    cat_ratio = cat_ratio_fraction(d_orig, d_after)
    rec_pct = recovery_pct(ade_before_base, ade_before_test, ade_after_test)

    print(f"  ADE base: {ade_before_base:.4f} -> {ade_after_base:.4f}")
    print(f"  ADE test: {ade_before_test:.4f} -> {ade_after_test:.4f}")
    print(f"  cat_ratio: {cat_ratio:.3f}")

    return dict(
        variant=tag, n_new=len(x_new), n_replay=len(replay_idx),
        n_unfreeze=args.n_unfreeze, lr=args.lr, epochs=args.epochs,
        ade_before_base=round(ade_before_base, 4),
        ade_after_base=round(ade_after_base, 4),
        ade_before_test=round(ade_before_test, 4),
        ade_after_test=round(ade_after_test, 4),
        cat_ratio=round(cat_ratio, 4),
        recovery_pct=round(rec_pct, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio",   type=float, default=0.5,
                    help="Razao replay/novos (como o melhor replay uniforme)")
    ap.add_argument("--top_frac", type=float, default=0.5,
                    help="Fracao dos novos selecionada por accel de cauda")
    ap.add_argument("--lr",      type=float, default=1e-5)
    ap.add_argument("--epochs",  type=int,   default=5)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--n_unfreeze", type=int, default=2)
    ap.add_argument("--max_per", type=int,   default=500)
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--out",     default=str(OUT_DIR / "targeted_results.csv"))
    args = ap.parse_args()

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    print(f"[info] TF {tf.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = load_loader()
    rng = np.random.default_rng(args.seed)

    bp_idx = YEAR_ORDER.index(BP_YEAR)
    before_ps = sorted(n for n, y in PROC_YEAR.items()
                       if YEAR_ORDER.index(y) < bp_idx)
    after_ps  = sorted(n for n, y in PROC_YEAR.items()
                       if YEAR_ORDER.index(y) >= bp_idx)

    x_base, y_base = load_proc_sets(before_ps, loader, max_per=args.max_per)
    x_test, y_test = load_proc_sets(after_ps,  loader, max_per=args.max_per)
    if x_base is None or x_test is None:
        sys.exit("[error] Could not load datasets.")
    print(f"[info] base (pre-{BP_YEAR}): {len(x_base):,} | "
          f"novos (pos-{BP_YEAR}): {len(x_test):,}")

    # ADE original no regime antigo (para cat_ratio)
    model_orig = build_model(tf)
    d_orig, _ = original_per_traj_dist(model_orig, x_base, y_base, loader)
    ade_before_base = float(np.mean(d_orig))
    ade_before_test = compute_ade(model_orig, x_test, y_test, loader)
    print(f"[info] ADE original: base={ade_before_base:.4f}  "
          f"test={ade_before_test:.4f}")

    # ── pesos LSIF para o replay: w(base) = p_novo / p_antigo ─────────────
    f_base = window_features(x_base)
    f_new  = window_features(x_test)
    scaler = StandardScaler().fit(np.vstack([f_base, f_new]))
    w_sel = lsif_weights(scaler.transform(f_base), scaler.transform(f_new),
                         n_centers=300, reg=1e-3, seed=args.seed)
    w_sel = np.maximum(w_sel, 1e-9)
    p_sel = w_sel / w_sel.sum()
    ess = float(w_sel.sum() ** 2 / (len(w_sel) * (w_sel ** 2).sum()))
    print(f"[info] pesos replay IW: ESS={ess:.3f}  w_max={w_sel.max():.2f}")

    # ── selecao por accel de cauda nos novos ──────────────────────────────
    score = accel_tail_score(x_test)
    thr = np.quantile(score, 1.0 - args.top_frac)
    sel = score >= thr
    print(f"[info] target_accel: top-{args.top_frac:.0%} -> {sel.sum():,} "
          f"janelas (thr={thr:.4f}; media sel={score[sel].mean():.4f} "
          f"vs resto={score[~sel].mean():.4f})")

    def draw_replay(n_ref: int) -> np.ndarray:
        n_rep = int(round(args.ratio * n_ref))
        return rng.choice(len(x_base), size=n_rep, replace=True, p=p_sel)

    rows = []
    # A) replay IW, dados novos completos
    rows.append(run_variant(
        tf, loader, "replay_iw", x_test, y_test, x_base, y_base,
        d_orig, ade_before_base, ade_before_test,
        x_test, y_test, draw_replay(len(x_test)), args))

    # B) seletivo por accel, sem replay
    rows.append(run_variant(
        tf, loader, "target_accel", x_test[sel], y_test[sel], x_base, y_base,
        d_orig, ade_before_base, ade_before_test,
        x_test, y_test, np.array([], dtype=int), args))

    # C) seletivo por accel + replay IW
    rows.append(run_variant(
        tf, loader, "target_accel+replay_iw", x_test[sel], y_test[sel],
        x_base, y_base, d_orig, ade_before_base, ade_before_test,
        x_test, y_test, draw_replay(int(sel.sum())), args))

    df = pd.DataFrame(rows)
    df.insert(1, "ess_replay_iw", round(ess, 4))
    df.to_csv(args.out, index=False)
    print(f"\n[ok] {args.out}")
    print(df[["variant", "n_new", "n_replay", "ade_after_base",
              "ade_after_test", "cat_ratio", "recovery_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
