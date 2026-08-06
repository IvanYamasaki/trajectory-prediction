"""
retrain_at_breakpoints.py  —  Frente 3 do Mês 3
=================================================

Fine-tuning leve do Seq2Seq (30→15) nos breakpoints do Pelt.

Estratégia:
  1. Detecta breakpoints no ADE médio por jogo usando ruptures.Pelt (RBF).
  2. Para cada breakpoint t*:
       - Treino  = todos os proc_sets cronologicamente antes de t*
       - Teste   = todos os proc_sets depois de t*
  3. Fine-tunes: congela encoder, descongela só a Dense final do decoder.
     lr=1e-4, 10 épocas (early-stop por val_loss se possível).
  4. Avalia ADE antes e depois no conjunto de teste.
  5. Salva pesos e resultados.

Saída:
  weights/robot_30_15_t_finetuned_bp{idx}.weights.h5
  Relas/results/drift/03_covariate_shift_e_explicacao/retrain_results.csv

Uso:
    python model_analise/retrain_at_breakpoints.py
    python model_analise/retrain_at_breakpoints.py --penalty 1 --epochs 10
    python model_analise/retrain_at_breakpoints.py --grid   # Ponto 5: grid sweep
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

from common.constants import PROC_YEAR, YEAR_ORDER
from common.paths import CH03
from model_analise.core.data import load_proc_sets
from model_analise.core.metrics import compute_ade, per_traj_ade
from model_analise.core.model_io import (
    build_seq2seq, freeze_encoder, load_loader as _core_load_loader,
)

# ─── constantes ──────────────────────────────────────────────────────────
LOOK_BACK   = 30
LOOK_FORTH  = 15
HORIZON     = "30→15"
UNITS       = 128
WEIGHTS_IN  = Path("weights") / "robot_30_15_t.weights.h5"
STATS_PATH  = Path("model") / "normalization_stats_30_15.pkl"
COV_DIR     = Path("covariate_shift_out")

OUT_DIR     = CH03
WEIGHTS_DIR = Path("weights")


# ─── helpers ─────────────────────────────────────────────────────────────

def year_of_proc(n: int) -> int:
    return PROC_YEAR.get(n, 0)


def proc_sets_before_after(breakpoint_year: int):
    before, after = [], []
    for n, y in PROC_YEAR.items():
        idx = YEAR_ORDER.index(y) if y in YEAR_ORDER else -1
        bp_idx = YEAR_ORDER.index(breakpoint_year)
        if idx < bp_idx:
            before.append(n)
        elif idx >= bp_idx:
            after.append(n)
    return sorted(before), sorted(after)


def load_loader():
    return _core_load_loader(LOOK_BACK, LOOK_FORTH, stats_path=STATS_PATH)


def load_data_sets(ns: list[int], loader, max_per: int = 8000, seed: int = 42):
    # sort_idx=True reproduz o comportamento original (idx.sort() pós-amostragem)
    return load_proc_sets(ns, loader, max_per=max_per, seed=seed, sort_idx=True)


def build_model(tf):
    return build_seq2seq(LOOK_BACK, LOOK_FORTH, units=UNITS, weights=WEIGHTS_IN)


BreakpointInfo = dict  # {bp_label, train_proc_sets, test_proc_sets}


def detect_pelt_breakpoints(penalty: float) -> list[BreakpointInfo]:
    """Detecta breakpoints no ADE por jogo (dataset.csv completo) usando Pelt RBF.

    Retorna lista de breakpoints com train/test proc_sets. Cada segmento Pelt
    define a fronteira treino/teste. Breakpoints intra-ano são mapeados para a
    transição de ano mais próxima (conservadorismo: não corta um ano ao meio).
    """
    try:
        import ruptures as rpt
    except ImportError:
        from drift_analise import changepoint_numpy_fallback as rpt

    ds = pd.read_csv("drift_analise/dataset/dataset.csv")
    seq = (
        ds[(ds["model"] == "Seq2Seq") & (ds["horizon"] == "30→15")]
        .sort_values(["year", "dataset"])
        .reset_index(drop=True)
    )
    seq["ade"] = seq["value"] if "value" in seq.columns else seq["ade_30_15"]

    signal    = seq["ade"].astype(float).to_numpy().reshape(-1, 1)
    years_arr = seq["year"].tolist()
    ps_arr    = seq["dataset"].astype(int).tolist()

    algo = rpt.Pelt(model="rbf").fit(signal)
    bps_raw = algo.predict(pen=penalty)
    print(f"  [Pelt] pen={penalty}  n_jogos={len(years_arr)}  bps_raw={bps_raw[:-1]}")

    seen = set()
    results = []
    for bp_i in bps_raw[:-1]:
        # mapear bp_i para o primeiro ano do segmento seguinte
        if bp_i <= 0 or bp_i >= len(years_arr):
            continue
        y_next = int(years_arr[bp_i])
        if y_next in seen:
            continue
        seen.add(y_next)
        # train = todos os proc_sets de anos ESTRITAMENTE anteriores a y_next
        train_ps = sorted({n for n, y in PROC_YEAR.items() if y < y_next})
        test_ps  = sorted({n for n, y in PROC_YEAR.items() if y >= y_next})
        if len(train_ps) < 2 or len(test_ps) < 2:
            continue
        label = f"antes_{y_next}"
        print(f"  bp → treino(y<{y_next})={train_ps}  teste(y>={y_next})={test_ps}")
        results.append({
            "bp_label":        label,
            "bp_year_test":    y_next,
            "train_proc_sets": train_ps,
            "test_proc_sets":  test_ps,
        })
    return results


# ─── main ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty",     type=float, default=5.0,
                    help="Penalidade Pelt (default 5)")
    ap.add_argument("--epochs",      type=int,   default=20,
                    help="Numero de epocas (era 10; review pediu mais).")
    ap.add_argument("--lr",          type=float, default=1e-5,
                    help="Learning rate (era 1e-4; review pediu menor).")
    ap.add_argument("--batch",       type=int,   default=256)
    ap.add_argument("--max_per",     type=int,   default=5000,
                    help="Maximo de janelas por jogo")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--n_unfreeze",  type=int,   default=2,
                    help="Quantas camadas finais descongelar (default 2 pos-review).")
    ap.add_argument("--early_stop_cat_ratio", type=float, default=0.5,
                    help="Para o treino se catastrophic_ratio passar deste valor "
                         "(testa apos cada epoca). Default 0.5.")
    ap.add_argument("--grid", action="store_true",
                    help="Ponto 5: executa grid sweep de hiperparâmetros "
                         "(n_unfreeze × lr). Saída: retrain_grid.csv. "
                         "Não sobrescreve retrain_results.csv.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] Importando TensorFlow...")
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    df_err = pd.read_parquet(p) if p.exists() else pd.read_csv(c)

    print("[info] Detectando breakpoints Pelt (dataset.csv completo)...")
    bps = detect_pelt_breakpoints(penalty=args.penalty)

    if not bps:
        print("[err] Nenhum breakpoint inter-ano detectado. Tente reduzir --penalty.")
        sys.exit(1)

    loader = load_loader()
    results = []

    for bp_enum, bp_info in enumerate(bps):
        bp_label = bp_info["bp_label"]
        before   = bp_info["train_proc_sets"]
        after    = bp_info["test_proc_sets"]
        print(f"\n[info] === Breakpoint {bp_enum} — {bp_label} ===")
        print(f"  Treino (proc_sets): {before}")
        print(f"  Teste  (proc_sets): {after}")

        if len(before) < 2 or len(after) < 2:
            print(f"  [skip] dados insuficientes (treino={len(before)}, teste={len(after)}).")
            continue

        x_train, y_train = load_data_sets(before, loader, args.max_per, args.seed)
        x_test,  y_test  = load_data_sets(after,  loader, args.max_per, args.seed)

        if x_train is None or x_test is None:
            print("  [skip] falha ao carregar dados.")
            continue

        print(f"  x_train={x_train.shape}  x_test={x_test.shape}")

        # ─── ANTES do fine-tuning ───────────────────────────────────────
        model = build_model(tf)
        ade_before = compute_ade(model, x_test, y_test, loader)
        ade_before_per = per_traj_ade(model, x_test, y_test, loader)
        print(f"  ADE antes fine-tuning: {ade_before:.4f} mm")

        # --- fine-tuning com early-stop por catastrophic_ratio --------
        freeze_encoder(model, n_unfreeze=args.n_unfreeze)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        epoch_log = []
        for ep in range(args.epochs):
            hist = model.fit(
                x_train, y_train,
                epochs=1, batch_size=args.batch,
                validation_split=0.1, verbose=0,
            )
            ade_now_per = per_traj_ade(model, x_test, y_test, loader)
            delta_now = ade_now_per - ade_before_per
            n_imp = int(np.sum(delta_now < 0))
            n_deg = int(np.sum(delta_now > 0))
            cat   = n_deg / max(n_imp, 1)
            epoch_log.append({
                "epoch": ep + 1,
                "loss":  float(hist.history["loss"][-1]),
                "val_loss": float(hist.history.get("val_loss", [float("nan")])[-1]),
                "cat_ratio": cat,
                "n_improved": n_imp, "n_degraded": n_deg,
            })
            print(f"  ep {ep+1:>2}/{args.epochs}  "
                  f"loss={hist.history['loss'][-1]:.4f}  "
                  f"cat_ratio={cat:.3f}  ({n_imp}+/{n_deg}-)")
            if cat > args.early_stop_cat_ratio:
                print(f"  [early-stop] cat_ratio {cat:.3f} > "
                      f"{args.early_stop_cat_ratio}. Parando.")
                break

        # ─── DEPOIS do fine-tuning ──────────────────────────────────────
        ade_after = compute_ade(model, x_test, y_test, loader)
        ade_after_per = per_traj_ade(model, x_test, y_test, loader)
        print(f"  ADE após  fine-tuning: {ade_after:.4f} mm")

        delta_mm  = ade_after - ade_before
        delta_pct = delta_mm / ade_before * 100.0 if ade_before > 0 else np.nan
        excess    = df_err[(df_err["model"] == "Seq2Seq") &
                           (df_err["horizon"] == HORIZON) &
                           (df_err["year"].isin([PROC_YEAR[n] for n in after]))
                           ]["ade_traj"].mean()
        ade_2019  = df_err[(df_err["model"] == "Seq2Seq") &
                           (df_err["horizon"] == HORIZON) &
                           (df_err["year"] == 2019)]["ade_traj"].mean()
        exc_total = excess - ade_2019
        recovery  = -delta_mm / exc_total * 100.0 if abs(exc_total) > 0.1 else np.nan

        delta_per = ade_after_per - ade_before_per
        n_improved  = int(np.sum(delta_per < 0))
        n_degraded  = int(np.sum(delta_per > 0))
        catastrophic = n_degraded / max(n_improved, 1)

        print(f"  delta={delta_mm:+.4f} mm ({delta_pct:+.2f}%)  "
              f"recovery={recovery:.1f}%  "
              f"improved={n_improved}  degraded={n_degraded}  ratio={catastrophic:.2f}")

        # salva pesos fine-tuned
        w_out = WEIGHTS_DIR / f"robot_30_15_t_finetuned_bp{bp_enum}.weights.h5"
        model.save_weights(str(w_out))
        print(f"  -> pesos salvos: {w_out}")

        results.append({
            "breakpoint_label":   bp_label,
            "breakpoint_idx":     bp_enum,
            "n_train":            len(x_train),
            "n_test":             len(x_test),
            "n_unfreeze":         args.n_unfreeze,
            "lr":                 args.lr,
            "epochs_run":         len(epoch_log),
            "epochs_budget":      args.epochs,
            "ade_before":         round(ade_before, 4),
            "ade_after":          round(ade_after, 4),
            "delta_mm":           round(delta_mm, 4),
            "delta_pct":          round(delta_pct, 2) if np.isfinite(delta_pct) else np.nan,
            "recovery_pct":       round(recovery, 2) if np.isfinite(recovery) else np.nan,
            "n_traj_improved":    n_improved,
            "n_traj_degraded":    n_degraded,
            "catastrophic_ratio": round(catastrophic, 3),
        })

        # log por epoca (auditavel)
        pd.DataFrame(epoch_log).to_csv(
            OUT_DIR / f"retrain_epoch_log_bp{bp_enum}.csv", index=False)

    if not results:
        print("[err] nenhum breakpoint processado.")
        sys.exit(1)

    df_res = pd.DataFrame(results)
    out_path = OUT_DIR / "retrain_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    print(df_res.to_string(index=False))

    # critério de aceite
    for _, row in df_res.iterrows():
        ok_rec = row["recovery_pct"] > 20 if pd.notna(row["recovery_pct"]) else False
        ok_cat = (row["n_traj_degraded"] / max(row["n_traj_improved"], 1)) < 0.3
        if ok_rec and ok_cat:
            print(f"\n[ACEITE] bp={row['breakpoint_label']}: "
                  f"recovery={row['recovery_pct']:.1f}%  "
                  f"catastrophic_ratio={row['catastrophic_ratio']:.2f}")
        else:
            print(f"\n[info] bp={row['breakpoint_label']}: criterios nao atingidos "
                  f"(recovery={row['recovery_pct']}, "
                  f"cat_ratio={row['catastrophic_ratio']:.2f})")

    # Ponto 5: grid sweep de hiperparametros
    if args.grid:
        print("\n" + "=" * 60)
        print("[grid] Iniciando sweep de hiperparametros")
        print("=" * 60)

        GRID_N_UNFREEZE = [1, 2]
        GRID_LR         = [1e-4, 5e-5, 1e-5]
        GRID_EPOCHS     = args.epochs

        grid_results = []

        bps_grid = detect_pelt_breakpoints(penalty=args.penalty)
        if not bps_grid:
            print("[grid] Nenhum breakpoint -- abortando grid.")
        else:
            bp_info_grid = bps_grid[0]
            before_g = bp_info_grid["train_proc_sets"]
            after_g  = bp_info_grid["test_proc_sets"]

            x_train_g, y_train_g = load_data_sets(before_g, loader, args.max_per, args.seed)
            x_test_g,  y_test_g  = load_data_sets(after_g,  loader, args.max_per, args.seed)

            if x_train_g is None or x_test_g is None:
                print("[grid] Falha ao carregar dados -- abortando grid.")
            else:
                for nu in GRID_N_UNFREEZE:
                    for lr_g in GRID_LR:
                        tag = f"nu{nu}_lr{lr_g:.0e}"
                        print(f"\n[grid] config={tag} ...")

                        model_g = build_model(tf)
                        ade_bef_g     = compute_ade(model_g, x_test_g, y_test_g, loader)
                        ade_bef_per_g = per_traj_ade(model_g, x_test_g, y_test_g, loader)

                        freeze_encoder(model_g, n_unfreeze=nu)
                        model_g.compile(
                            optimizer=tf.keras.optimizers.Adam(lr_g),
                            loss=tf.keras.losses.MeanSquaredError(),
                        )

                        epochs_run_g = 0
                        cat_g = 0.0
                        for ep_g in range(GRID_EPOCHS):
                            model_g.fit(x_train_g, y_train_g,
                                        epochs=1, batch_size=args.batch,
                                        validation_split=0.1, verbose=0)
                            ade_now_per_g = per_traj_ade(model_g, x_test_g, y_test_g, loader)
                            delta_g = ade_now_per_g - ade_bef_per_g
                            n_imp_g = int(np.sum(delta_g < 0))
                            n_deg_g = int(np.sum(delta_g > 0))
                            cat_g   = n_deg_g / max(n_imp_g, 1)
                            epochs_run_g += 1
                            print(f"  ep {ep_g+1}  cat_ratio={cat_g:.3f}")
                            if cat_g > args.early_stop_cat_ratio:
                                print(f"  [early-stop] cat_ratio={cat_g:.3f}")
                                break

                        ade_aft_g  = compute_ade(model_g, x_test_g, y_test_g, loader)
                        delta_mm_g = ade_aft_g - ade_bef_g
                        ade_2019_g = df_err[
                            (df_err["model"] == "Seq2Seq") &
                            (df_err["horizon"] == HORIZON) &
                            (df_err["year"] == 2019)
                        ]["ade_traj"].mean()
                        ade_post_g = df_err[
                            (df_err["model"] == "Seq2Seq") &
                            (df_err["horizon"] == HORIZON) &
                            (df_err["year"].isin([PROC_YEAR[n] for n in after_g]))
                        ]["ade_traj"].mean()
                        exc_g = ade_post_g - ade_2019_g
                        rec_g = (-delta_mm_g / exc_g * 100.0
                                 if abs(exc_g) > 0.1 else float("nan"))

                        print(f"  ADE: {ade_bef_g:.4f} -> {ade_aft_g:.4f}  "
                              f"delta={delta_mm_g:+.4f}  recovery={rec_g:.1f}%  "
                              f"cat_ratio={cat_g:.3f}  epochs={epochs_run_g}")

                        grid_results.append({
                            "breakpoint_label":   bp_info_grid["bp_label"],
                            "n_unfreeze":         nu,
                            "lr":                 lr_g,
                            "epochs_budget":      GRID_EPOCHS,
                            "epochs_run":         epochs_run_g,
                            "ade_before":         round(ade_bef_g, 4),
                            "ade_after":          round(ade_aft_g, 4),
                            "delta_mm":           round(delta_mm_g, 4),
                            "recovery_pct":       round(rec_g, 2) if rec_g == rec_g else float("nan"),
                            "catastrophic_ratio": round(cat_g, 3),
                        })

                if grid_results:
                    df_grid = pd.DataFrame(grid_results)
                    grid_path = OUT_DIR / "retrain_grid.csv"
                    df_grid.to_csv(grid_path, index=False)
                    print(f"\n[ok] {grid_path}")
                    print(df_grid.to_string(index=False))
                    best = df_grid.loc[df_grid["recovery_pct"].fillna(-999).idxmax()]
                    print(f"\n[grid] Melhor config: n_unfreeze={best['n_unfreeze']}  "
                          f"lr={best['lr']:.0e}  recovery={best['recovery_pct']:.1f}%  "
                          f"cat_ratio={best['catastrophic_ratio']:.3f}")


if __name__ == "__main__":
    main()
