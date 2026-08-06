import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bootstrap import init_project

PROJECT_ROOT = init_project()

from dataset.load_dataset import LoadDataSet
from model_analise.core.model_io import build_seq2seq

SEED = 7
DATASETS = ["dataset/proc_set_27"]
N_SAMPLES = 5000
UNITS = 128

CFG = {
    "30_15": {"look_back": 30, "look_forth": 15, "seq_w": "weights/robot_30_15_t.weights.h5", "mlp_w": "weights/mlp_30_15.weights.h5"},
    "60_30": {"look_back": 60, "look_forth": 30, "seq_w": "weights/robot_60_30_t.weights.h5", "mlp_w": "weights/mlp_60_30.weights.h5"},
}

def seed_all(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_mlp(look_back, look_forth):
    inp = tf.keras.Input(shape=(look_back, 6))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(look_forth * 2)(x)
    out = tf.keras.layers.Reshape((look_forth, 2))(x)
    return tf.keras.Model(inp, out)

def load_mlp(look_back, look_forth, weights):
    m = build_mlp(look_back, look_forth)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
    _ = m(tf.zeros((1, look_back, 6), tf.float32), training=False)
    m.load_weights(weights)
    return m

def kalman_preds_norm(x_norm, look_forth):
    v0 = x_norm[:, -1:, 2:4]
    return np.repeat(v0, look_forth, axis=1).astype(np.float32)

def dv_to_pos_mm(loader, x_norm, dv_norm):
    last_pos = x_norm[:, loader.look_back - 1, 0:2] * loader.robots_std[0:2] + loader.robots_avg[0:2]
    v0 = x_norm[:, loader.look_back - 1, 2:4] * loader.robots_std[2:4] + loader.robots_avg[2:4]
    dv = dv_norm * loader.robots_std[2:4]
    v = v0[:, None, :] + np.cumsum(dv, axis=1)
    pos = last_pos[:, None, :] + np.cumsum(v, axis=1)
    return pos

def v_to_pos_mm(loader, x_norm, v_norm):
    return loader.convert_batch(x_norm, v_norm)

def metrics_mm(pos_pred, pos_true):
    d = pos_pred - pos_true
    dist = np.linalg.norm(d, axis=-1)
    ade = float(np.mean(dist))
    fde = float(np.mean(dist[:, -1]))
    mae = float(np.mean(np.abs(d)))
    return ade, fde, mae

def print_table(tag, n, mk, ms, mm, mode_s):
    print("\n" + "=" * 64)
    print(f"{tag}  (N={n})")
    print("-" * 64)
    print(f"{'Kalman':<8} ADE={mk[0]:8.3f}  FDE={mk[1]:8.3f}  MAE={mk[2]:8.3f}")
    print(f"{'Seq2Seq':<8} ADE={ms[0]:8.3f}  FDE={ms[1]:8.3f}  MAE={ms[2]:8.3f}   (as {mode_s})")
    print(f"{'MLP':<8} ADE={mm[0]:8.3f}  FDE={mm[1]:8.3f}  MAE={mm[2]:8.3f}")
    print("=" * 64)

def pick_seq2seq_mode(loader, x, y_v, pred_s):
    k = min(256, len(x))
    pos_true = v_to_pos_mm(loader, x[:k], y_v[:k])
    pos_as_v = v_to_pos_mm(loader, x[:k], pred_s[:k])
    pos_as_dv = dv_to_pos_mm(loader, x[:k], pred_s[:k])
    ade_v = metrics_mm(pos_as_v, pos_true)[0]
    ade_dv = metrics_mm(pos_as_dv, pos_true)[0]
    return ("v", ade_v) if ade_v <= ade_dv else ("dv", ade_dv)

def run_case(tag, cfg, datasets_to_test):
    look_back, look_forth = cfg["look_back"], cfg["look_forth"]
    seq_w, mlp_w = cfg["seq_w"], cfg["mlp_w"]
    if not os.path.exists(seq_w): raise FileNotFoundError(f"Não achei {seq_w}")
    if not os.path.exists(mlp_w): raise FileNotFoundError(f"Não achei {mlp_w}")

    loader = LoadDataSet(look_back, look_forth)  # aqui y é VELOCIDADE v (padrão do projeto)
    x_all, _, _, y_v_all = loader.load_data(datasets_to_test, for_test=True)

    n = min(N_SAMPLES, len(x_all))
    idx = np.random.choice(len(x_all), n, replace=False)
    x = x_all[idx].astype(np.float32)
    y_v = y_v_all[idx].astype(np.float32)

    seq = build_seq2seq(look_back, look_forth, units=UNITS, weights=seq_w)
    mlp = load_mlp(look_back, look_forth, mlp_w)

    pred_k = kalman_preds_norm(x, look_forth)
    pred_s = seq.predict(x, verbose=0).astype(np.float32)
    pred_m = mlp.predict(x, verbose=0).astype(np.float32)

    mode_s, _ = pick_seq2seq_mode(loader, x, y_v, pred_s)

    pos_true = v_to_pos_mm(loader, x, y_v)
    pos_k = v_to_pos_mm(loader, x, pred_k)
    pos_m = v_to_pos_mm(loader, x, pred_m)
    pos_s = v_to_pos_mm(loader, x, pred_s) if mode_s == "v" else dv_to_pos_mm(loader, x, pred_s)

    mk = metrics_mm(pos_k, pos_true)
    ms = metrics_mm(pos_s, pos_true)
    mm = metrics_mm(pos_m, pos_true)
    print_table(tag, n, mk, ms, mm, mode_s)

    return mk, ms, mm

def main():
    seed_all(SEED)
    
    print("\n" + "="*80)
    print("COMPARAÇÃO DE MODELOS POR DATASET")
    print("="*80)
    
    all_results = []
    
    # Processar cada dataset individualmente
    for dataset_file in DATASETS:
        print(f"\n>>> Processando dataset: {dataset_file}")
        results_30_15 = run_case(f"30→15 ({dataset_file})", CFG["30_15"], [dataset_file])
        results_60_30 = run_case(f"60→30 ({dataset_file})", CFG["60_30"], [dataset_file])
        all_results.append((dataset_file, results_30_15, results_60_30))
    
    # Resumo final com todos os datasets
    print("\n" + "="*80)
    print("RESUMO FINAL - TODOS OS DATASETS")
    print("="*80)
    print(f"{'Dataset':<30} {'Config':<8} {'ADE':<10} {'FDE':<10} {'MAE':<10}")
    print("-"*80)
    for dataset_file, (mk_30, ms_30, mm_30), (mk_60, ms_60, mm_60) in all_results:
        fname = dataset_file.split('/')[-1]
        print(f"{fname:<30} {'30→15':<8} K:{mk_30[0]:<8.2f} S:{ms_30[0]:<8.2f} M:{mm_30[0]:<8.2f}")
        print(f"{'':<30} {'60→30':<8} K:{mk_60[0]:<8.2f} S:{ms_60[0]:<8.2f} M:{mm_60[0]:<8.2f}")

if __name__ == "__main__":
    main()
