"""
replay_finetune.py — Baseline de adaptacao por replay (rehearsal)
=================================================================
Alternativa ao EWC apontada como "agenda natural" no artigo: fine-tuning
sobre os dados novos MISTURADOS com um buffer de replay do regime original.

Protocolo identico ao ewc_finetune.py (mesmos proc_sets, max_per=500,
lr=1e-5, n_unfreeze=2, 5 epocas, batch 256, Adam), variando apenas a
composicao do conjunto de treino:

    treino = dados novos (pos-2022)  +  r * |novos| amostras do regime antigo

com razao de replay r em {0.25, 0.5, 1.0}. r=0 equivale ao fine-tuning
ingenuo (cat_ratio=0.444 no protocolo do Mes 7).

Metricas: ade_before/after (base e teste), cat_ratio (fracao de trajetorias
do regime antigo que pioram vs. o modelo original) e recovery_pct.

Saida: Relas/results/mes7/replay_results.csv

Uso:
    python model_analise/replay_finetune.py
    python model_analise/replay_finetune.py --ratios 0.5 --epochs 5
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

from model_analise.ewc_finetune import (
    PROC_YEAR, YEAR_ORDER, BP_YEAR, OUT_DIR,
    load_loader, load_proc_sets, build_model, freeze_encoder, compute_ade,
)
from model_analise.core.metrics import (
    cat_ratio_fraction, original_per_traj_dist, recovery_pct,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                    help="Razoes replay/novos a testar")
    ap.add_argument("--lr",      type=float, default=1e-5)
    ap.add_argument("--epochs",  type=int,   default=5)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--n_unfreeze", type=int, default=2)
    ap.add_argument("--max_per", type=int,   default=500)
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--out",     default=str(OUT_DIR / "replay_results.csv"))
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

    # ADE por trajetoria do modelo ORIGINAL no regime antigo (para cat_ratio)
    model_orig = build_model(tf)
    d_orig, _ = original_per_traj_dist(model_orig, x_base, y_base, loader)
    ade_before_base = float(np.mean(d_orig))
    ade_before_test = compute_ade(model_orig, x_test, y_test, loader)
    print(f"[info] ADE original: base={ade_before_base:.4f}  "
          f"test={ade_before_test:.4f}")

    rows = []
    for r in args.ratios:
        print(f"\n{'='*60}\n[run] replay ratio r={r}  lr={args.lr}  "
              f"n_unfreeze={args.n_unfreeze}")
        model = build_model(tf)
        freeze_encoder(model, args.n_unfreeze)

        # conjunto de treino: novos + buffer de replay do regime antigo
        n_replay = int(round(r * len(x_test)))
        idx_rep  = rng.choice(len(x_base), size=min(n_replay, len(x_base)),
                              replace=n_replay > len(x_base))
        x_tr = np.concatenate([x_test, x_base[idx_rep]], axis=0)
        y_tr = np.concatenate([y_test, y_base[idx_rep]], axis=0)
        print(f"  treino: {len(x_test):,} novos + {len(idx_rep):,} replay "
              f"= {len(x_tr):,}")

        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss=tf.keras.losses.MeanSquaredError())
        model.fit(x_tr, y_tr, batch_size=args.batch, epochs=args.epochs,
                  shuffle=True, verbose=2)

        ade_after_base = compute_ade(model, x_base, y_base, loader)
        ade_after_test = compute_ade(model, x_test, y_test, loader)

        d_after, _ = original_per_traj_dist(model, x_base, y_base, loader)
        cat_ratio = cat_ratio_fraction(d_orig, d_after)
        rec_pct = recovery_pct(ade_before_base, ade_before_test, ade_after_test)

        print(f"  ADE base: {ade_before_base:.4f} -> {ade_after_base:.4f}")
        print(f"  ADE test: {ade_before_test:.4f} -> {ade_after_test:.4f}")
        print(f"  cat_ratio: {cat_ratio:.3f}  (EWC lam=0: 0.444)")
        print(f"  recovery_pct: {rec_pct:.1f}%")

        rows.append(dict(
            replay_ratio=r, n_replay=len(idx_rep),
            n_unfreeze=args.n_unfreeze, lr=args.lr, epochs=args.epochs,
            ade_before_base=round(ade_before_base, 4),
            ade_after_base=round(ade_after_base, 4),
            ade_before_test=round(ade_before_test, 4),
            ade_after_test=round(ade_after_test, 4),
            cat_ratio=round(cat_ratio, 4),
            recovery_pct=round(rec_pct, 1),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[ok] {args.out}")
    print(df[["replay_ratio", "ade_after_base", "ade_after_test",
              "cat_ratio", "recovery_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
