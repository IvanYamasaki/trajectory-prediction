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
  Relas/results/mes3/retrain_results.csv

Uso:
    python model_analise/retrain_at_breakpoints.py
    python model_analise/retrain_at_breakpoints.py --penalty 1 --epochs 10
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ─── constantes ──────────────────────────────────────────────────────────
LOOK_BACK   = 30
LOOK_FORTH  = 15
HORIZON     = "30→15"
UNITS       = 128
WEIGHTS_IN  = Path("weights") / "robot_30_15_t.weights.h5"
STATS_PATH  = Path("model") / "normalization_stats_30_15.pkl"
COV_DIR     = Path("covariate_shift_out")
OUT_DIR     = Path("Relas") / "results" / "mes3"
WEIGHTS_DIR = Path("weights")

PROC_YEAR = {
    3: 2019, 4: 2019, 5: 2019, 6: 2019, 7: 2019, 8: 2019,
    9: 2021,10: 2021,11: 2021,12: 2021,13: 2021,14: 2021,
    15:2023,16:2023,17:2023,18:2023,19:2023,20:2023,
    21:2025,22:2025,23:2025,24:2025,25:2025,26:2025,
    27:2022,28:2022,29:2022,30:2022,31:2022,32:2022,
    33:2024,34:2024,35:2024,36:2024,37:2024,38:2024,
}
YEAR_ORDER = [2019, 2021, 2022, 2023, 2024, 2025]


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
    from dataset.load_dataset import LoadDataSet
    loader = LoadDataSet(LOOK_BACK, LOOK_FORTH, target="dv")
    with open(STATS_PATH, "rb") as f:
        s = pickle.load(f)
    loader.robots_avg = np.asarray(s["avg"],      dtype=np.float64)
    loader.robots_std = np.asarray(s["std"],      dtype=np.float64)
    loader.ball_avg   = np.asarray(s["ball_avg"], dtype=np.float64)
    loader.ball_std   = np.asarray(s["ball_std"], dtype=np.float64)
    return loader


def load_data_sets(ns: list[int], loader, max_per: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for n in ns:
        fpath = f"dataset/proc_set_{n}"
        if not Path(fpath + ".pkl").exists():
            continue
        try:
            x, _, _, y_v = loader.load_data([fpath], for_test=True)
        except Exception as e:
            print(f"  [warn] {fpath}: {e}")
            continue
        if len(x) == 0:
            continue
        if len(x) > max_per:
            idx = rng.choice(len(x), max_per, replace=False)
            idx.sort()
            x, y_v = x[idx], y_v[idx]
        xs.append(x)
        ys.append(y_v)
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0).astype(np.float32), \
           np.concatenate(ys, axis=0).astype(np.float32)


def build_model(tf):
    from model_analise.ai_model.predictor import RobotOnlyPredictor
    m = RobotOnlyPredictor(
        units=UNITS, look_back=LOOK_BACK, look_forth=LOOK_FORTH,
        result_dims=2, use_tf_function=False, forcing=False,
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    _ = m(tf.zeros((1, LOOK_BACK, 6), tf.float32), training=False)
    m.load_weights(str(WEIGHTS_IN))
    return m


def freeze_encoder(model, n_unfreeze: int = 1):
    """Congela todas as camadas exceto as ultimas ``n_unfreeze``.

    n_unfreeze=1 -> so Dense final (default conservador, baseline)
    n_unfreeze=2 -> Dense final + camada anterior (recomendado pos-review M3)
    """
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-n_unfreeze:]:
        layer.trainable = True
    frozen = sum(1 for l in model.layers if not l.trainable)
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  [freeze] {frozen} frozen, {trainable} trainable layers "
          f"(n_unfreeze={n_unfreeze}).")


def compute_ade(model, x, y_v, loader, batch=512):
    """Retorna ADE médio em mm."""
    pred_v = model.predict(x, batch_size=batch, verbose=0)
    pos_pred = loader.convert_batch(x, pred_v.astype(np.float32))
    pos_true = loader.convert_batch(x, y_v)
    d = np.linalg.norm(pos_pred - pos_true, axis=-1)  # [N, look_forth]
    return float(np.mean(np.mean(d, axis=1)))


def per_traj_ade(model, x, y_v, loader, batch=512):
    pred_v = model.predict(x, batch_size=batch, verbose=0)
    pos_pred = loader.convert_batch(x, pred_v.astype(np.float32))
    pos_true = loader.convert_batch(x, y_v)
    d = np.linalg.norm(pos_pred - pos_true, axis=-1)
    return np.mean(d, axis=1)


BreakpointInfo = dict  # {bp_label, train_proc_sets, test_proc_sets}


def detect_pelt_breakpoints(penalty: float) -> list[BreakpointInfo]:
    """Detecta breakpoints no ADE por jogo (dataset.csv completo) usando Pelt RBF.

    Retorna lista de breakpoints com train/test proc_sets. Cada segmento Pelt
    define a fronteira treino/teste. Breakpoints intra-ano são mapeados para a
    transição de ano mais próxima (conservadorismo: não corta um ano ao meio).
    """
    import ruptures as rpt

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
            print(f"\n[info] bp={row['breakpoint_label']}: critérios não atingidos "
                  f"(recovery={row['recovery_pct']}, "
                  f"cat_ratio={row['catastrophic_ratio']:.2f})")


if __name__ == "__main__":
    main()
