"""
compute_trajectory_errors.py
============================

Gera `covariate_shift_out/trajectory_errors_sample.parquet` com erros por
janela (uma "trajetória" no contexto Mês 1) para uma AMOSTRA aleatória de
3 jogos por ano (seed fixo).

Saída (uma linha por janela × modelo × horizonte):
    match_id, year, log_file, proc_set_file, model, horizon, traj_id,
    ade_traj, fde_traj, n_steps_pred

Reaproveita `LoadDataSet` e o `RobotOnlyPredictor`. Estatísticas de
normalização são lidas dos arquivos `normalization_stats_{lb}_{lf}.pkl`
salvos no treino.

Uso a partir da raiz do projeto:
    python model_analise/compute_trajectory_errors.py
    python model_analise/compute_trajectory_errors.py --n_per_year 3 --seed 42
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# As importações pesadas (TF + LoadDataSet) são lazy
def _import_heavy():
    import tensorflow as tf  # noqa: F401
    from dataset.load_dataset import LoadDataSet
    from model_analise.ai_model.predictor import RobotOnlyPredictor
    return tf, LoadDataSet, RobotOnlyPredictor


# ----------------------- configuração ---------------------------------------
OUTDIR = "covariate_shift_out"
DATASET_CSV = "drift_analise/dataset/dataset.csv"

# Mapping proc_set -> ano (igual ao plot_covariate_shift.py)
PROC_YEAR = {
    3: 2019, 4: 2019, 5: 2019, 6: 2019, 7: 2019, 8: 2019,
    9: 2021, 10: 2021, 11: 2021, 12: 2021, 13: 2021, 14: 2021,
    15: 2023, 16: 2023, 17: 2023, 18: 2023, 19: 2023, 20: 2023,
    21: 2025, 22: 2025, 23: 2025, 24: 2025, 25: 2025, 26: 2025,
    27: 2022, 28: 2022, 29: 2022, 30: 2022, 31: 2022, 32: 2022,
    33: 2024, 34: 2024, 35: 2024, 36: 2024, 37: 2024, 38: 2024,
}

CFG = {
    "30→15": {"look_back": 30, "look_forth": 15, "weights": "weights/robot_30_15_t.weights.h5",
              "stats": "model/normalization_stats_30_15.pkl"},
    "60→30": {"look_back": 60, "look_forth": 30, "weights": "weights/robot_60_30_t.weights.h5",
              "stats": "model/normalization_stats_60_30.pkl"},
}

UNITS = 128


# ----------------------- helpers --------------------------------------------
def short_match_id(year, fname, log_file, teams):
    if teams:
        t = teams.replace(" vs ", "_vs_").replace(" ", "")
        t = re.sub(r"[^A-Za-z0-9_]+", "", t)
        return f"{year}_{t}"
    if log_file:
        base = os.path.basename(log_file).replace(".log", "")
        base = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_", "", base)
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", base)
        return f"{year}_{base}"[:80]
    return f"{year}_{fname.replace('.pkl','')}"


def load_dataset_csv_mapping(csv_path):
    """Retorna dict {proc_set_N -> (log_file, teams)} a partir de dataset.csv (se existir)."""
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    df = pd.read_csv(csv_path)
    for _, row in df[["dataset", "log_file", "teams"]].drop_duplicates().iterrows():
        n = row["dataset"]
        try:
            n_int = int(n)
        except Exception:
            continue
        mapping[n_int] = (str(row.get("log_file", "")), str(row.get("teams", "")))
    return mapping


def sample_games(n_per_year=6, seed=42, exclude_proc=(1, 2),
                 full_baseline=True, baseline_year=2019) -> List[int]:
    """Retorna lista de proc_set_N amostrados (n por ano), excluindo treino.

    Se ``full_baseline=True``, ``baseline_year`` (default 2019) recebe TODOS
    os proc_sets disponiveis (sem amostragem). Reduz a instabilidade do ADE
    de referencia observada no review do Mes 3.
    """
    rng = np.random.default_rng(seed)
    by_year = {}
    for n, y in PROC_YEAR.items():
        if n in exclude_proc:
            continue
        by_year.setdefault(y, []).append(n)

    chosen = []
    for y in sorted(by_year):
        pool = sorted(by_year[y])
        if full_baseline and y == baseline_year:
            chosen.extend(pool)
            continue
        k = min(n_per_year, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        chosen.extend([pool[i] for i in idx])
    return sorted(chosen)


def build_loader(LoadDataSet, look_back, look_forth, stats_path):
    loader = LoadDataSet(look_back, look_forth, target="dv")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Arquivo de estatísticas não encontrado: {stats_path}. "
            f"Sem ele a normalização ficaria errada."
        )
    with open(stats_path, "rb") as f:
        s = pickle.load(f)
    # injeta stats ANTES do load_data para evitar recálculo
    loader.robots_avg = np.asarray(s["avg"], dtype=np.float64)
    loader.robots_std = np.asarray(s["std"], dtype=np.float64)
    loader.ball_avg   = np.asarray(s["ball_avg"], dtype=np.float64)
    loader.ball_std   = np.asarray(s["ball_std"], dtype=np.float64)
    return loader


def load_seq2seq(RobotOnlyPredictor, tf, look_back, look_forth, weights):
    m = RobotOnlyPredictor(
        units=UNITS, look_back=look_back, look_forth=look_forth,
        result_dims=2, use_tf_function=False, forcing=False,
    )
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.MeanSquaredError())
    _ = m(tf.zeros((1, look_back, 6), tf.float32), training=False)
    m.load_weights(weights)
    return m


def kalman_preds_norm(x_norm, look_forth):
    # target="dv": constant-velocity baseline means zero velocity increments.
    return np.zeros((x_norm.shape[0], look_forth, 2), dtype=np.float32)


def v_to_pos_mm(loader, x_norm, v_norm):
    return loader.convert_batch(x_norm, v_norm)


def per_traj_metrics(pos_pred, pos_true):
    d = pos_pred - pos_true
    dist = np.linalg.norm(d, axis=-1)  # [N, look_forth]
    ade = np.mean(dist, axis=1)        # [N]
    fde = dist[:, -1]                  # [N]
    return ade.astype(np.float32), fde.astype(np.float32)


# ----------------------- main -----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_year", type=int, default=6,
                    help="Quantos jogos por ano amostrar (default 6)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_windows_per_game", type=int, default=8000,
                    help="Limite de janelas por jogo p/ nao estourar memoria")
    ap.add_argument("--no_full_baseline", action="store_true",
                    help="Se setado, AMOSTRA tambem o baseline 2019. "
                         "Default: usa todos os proc_sets de 2019.")
    ap.add_argument("--baseline_year", type=int, default=2019,
                    help="Ano de referencia (baseline) - default 2019.")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[info] Importando TensorFlow / LoadDataSet ...")
    tf, LoadDataSet, RobotOnlyPredictor = _import_heavy()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    csv_map = load_dataset_csv_mapping(DATASET_CSV)
    chosen = sample_games(
        n_per_year=args.n_per_year, seed=args.seed,
        full_baseline=not args.no_full_baseline,
        baseline_year=args.baseline_year,
    )
    print(f"[info] Jogos amostrados (proc_set N): {chosen}")
    print(f"[info] full_baseline={not args.no_full_baseline}  "
          f"baseline_year={args.baseline_year}  n_per_year={args.n_per_year}")

    rows = []
    rng = np.random.default_rng(args.seed)

    # carrega cada modelo UMA vez por horizonte
    for horizon, cfg in CFG.items():
        if not os.path.exists(cfg["weights"]):
            print(f"[warn] Pesos não encontrados para {horizon}: {cfg['weights']} — pulando.")
            continue
        print(f"\n[info] === Horizonte {horizon} ===")
        seq = load_seq2seq(RobotOnlyPredictor, tf, cfg["look_back"], cfg["look_forth"],
                           cfg["weights"])

        for n in chosen:
            fname = f"proc_set_{n}.pkl"
            fpath = f"dataset/proc_set_{n}"
            if not os.path.exists(fpath + ".pkl"):
                print(f"[warn] sem arquivo {fpath}.pkl — pulando.")
                continue

            year = PROC_YEAR.get(n, None)
            log_file, teams = csv_map.get(n, ("", ""))
            mid = short_match_id(year, fname, log_file, teams)

            # loader com stats fixas do treino (NÃO recompute)
            loader = build_loader(LoadDataSet, cfg["look_back"], cfg["look_forth"],
                                  cfg["stats"])
            try:
                x_all, _, _, y_v_all = loader.load_data([fpath], for_test=True)
            except Exception as e:
                print(f"[warn] falha ao carregar {fpath}: {e}")
                continue

            n_total = len(x_all)
            if n_total == 0:
                print(f"[warn] {fname}: 0 janelas — pulando.")
                continue

            # subamostra se passar do limite
            if n_total > args.max_windows_per_game:
                idx = rng.choice(n_total, size=args.max_windows_per_game, replace=False)
                idx.sort()
            else:
                idx = np.arange(n_total)

            x = x_all[idx].astype(np.float32)
            y_v = y_v_all[idx].astype(np.float32)

            # ground truth em mm
            pos_true = v_to_pos_mm(loader, x, y_v)

            # Seq2Seq
            pred_s = seq.predict(x, verbose=0).astype(np.float32)
            pos_s = v_to_pos_mm(loader, x, pred_s)
            ade_s, fde_s = per_traj_metrics(pos_s, pos_true)

            # Kalman
            pred_k = kalman_preds_norm(x, cfg["look_forth"])
            pos_k = v_to_pos_mm(loader, x, pred_k)
            ade_k, fde_k = per_traj_metrics(pos_k, pos_true)

            for j, (g_idx) in enumerate(idx):
                rows.append({
                    "match_id": mid, "year": year,
                    "log_file": log_file, "proc_set_file": fname,
                    "model": "Seq2Seq", "horizon": horizon,
                    "traj_id": int(g_idx),
                    "ade_traj": float(ade_s[j]), "fde_traj": float(fde_s[j]),
                    "n_steps_pred": int(cfg["look_forth"]),
                })
                rows.append({
                    "match_id": mid, "year": year,
                    "log_file": log_file, "proc_set_file": fname,
                    "model": "Kalman", "horizon": horizon,
                    "traj_id": int(g_idx),
                    "ade_traj": float(ade_k[j]), "fde_traj": float(fde_k[j]),
                    "n_steps_pred": int(cfg["look_forth"]),
                })

            print(f"  {fname:>20}  year={year}  N={len(idx):5d}  "
                  f"ADE_seq2seq_mean={float(np.mean(ade_s)):.2f}  "
                  f"ADE_kalman_mean={float(np.mean(ade_k)):.2f}")

    if not rows:
        print("[err] Nenhuma linha gerada.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTDIR, "trajectory_errors_sample.parquet")
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        # fallback p/ csv se pyarrow/fastparquet faltarem
        out_path = os.path.join(OUTDIR, "trajectory_errors_sample.csv")
        df.to_csv(out_path, index=False)
        print(f"[warn] parquet indisponível ({e}); salvo CSV em {out_path}")
    print(f"\n[ok] {len(df):,} linhas -> {out_path}")


if __name__ == "__main__":
    main()
