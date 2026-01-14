import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.load_dataset import LoadDataSet
from ai_model.predictor import RobotOnlyPredictor

SEED = 7
DATASETS = ["dataset/proc_set_4"]
N_SAMPLES = 5000
N_PLOTS = 3
UNITS = 128

CFG = {
    "30_15": {"look_back": 30, "look_forth": 15, "seq_w": "robot_30_15_t.weights.h5", "mlp_w": "mlp_30_15.weights.h5"},
    "60_30": {"look_back": 60, "look_forth": 30, "seq_w": "robot_60_30_t.weights.h5", "mlp_w": "mlp_60_30.weights.h5"},
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

def load_seq2seq(look_back, look_forth, weights):
    m = RobotOnlyPredictor(units=UNITS, look_back=look_back, look_forth=look_forth, result_dims=2, use_tf_function=False, forcing=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
    _ = m(tf.zeros((1, look_back, 6), tf.float32), training=False)
    m.load_weights(weights)
    return m

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

def denorm_past(loader, x_norm_one):
    return x_norm_one[:, 0:2] * loader.robots_std[0:2] + loader.robots_avg[0:2]

def plot_samples(tag, idx_global, x, pos_true, pos_k, pos_s, pos_m, loader, picks, mode_s):
    os.makedirs("plots", exist_ok=True)
    for p, j in enumerate(picks, start=1):
        past = denorm_past(loader, x[j])
        real = pos_true[j]
        pk = pos_k[j]
        ps = pos_s[j]
        pm = pos_m[j]
        plt.figure(figsize=(7, 6))
        plt.plot(past[:, 0], past[:, 1], "b-o", label="Passado")
        plt.plot(real[:, 0], real[:, 1], "g-o", label="Real")
        plt.plot(pk[:, 0], pk[:, 1], "k-o", label="Kalman")
        plt.plot(ps[:, 0], ps[:, 1], "r-o", label=f"Seq2Seq ({mode_s})")
        plt.plot(pm[:, 0], pm[:, 1], "m-o", label="MLP")
        plt.title(f"{tag} sample {p} idx={int(idx_global[j])}")
        plt.axis("equal"); plt.grid(True); plt.legend(); plt.tight_layout()
        out = f"plots/{tag.replace('→','_')}_sample{p}_idx{int(idx_global[j])}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] saved {out}")

def pick_seq2seq_mode(loader, x, y_v, pred_s):
    k = min(256, len(x))
    pos_true = v_to_pos_mm(loader, x[:k], y_v[:k])
    pos_as_v = v_to_pos_mm(loader, x[:k], pred_s[:k])
    pos_as_dv = dv_to_pos_mm(loader, x[:k], pred_s[:k])
    ade_v = metrics_mm(pos_as_v, pos_true)[0]
    ade_dv = metrics_mm(pos_as_dv, pos_true)[0]
    return ("v", ade_v) if ade_v <= ade_dv else ("dv", ade_dv)

def run_case(tag, cfg):
    look_back, look_forth = cfg["look_back"], cfg["look_forth"]
    seq_w, mlp_w = cfg["seq_w"], cfg["mlp_w"]
    if not os.path.exists(seq_w): raise FileNotFoundError(f"Não achei {seq_w}")
    if not os.path.exists(mlp_w): raise FileNotFoundError(f"Não achei {mlp_w}")

    loader = LoadDataSet(look_back, look_forth)  # aqui y é VELOCIDADE v (padrão do projeto)
    x_all, _, _, y_v_all = loader.load_data(DATASETS, for_test=True)

    n = min(N_SAMPLES, len(x_all))
    idx = np.random.choice(len(x_all), n, replace=False)
    x = x_all[idx].astype(np.float32)
    y_v = y_v_all[idx].astype(np.float32)

    seq = load_seq2seq(look_back, look_forth, seq_w)
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

    picks = np.random.choice(n, min(N_PLOTS, n), replace=False)
    plot_samples(tag, idx, x, pos_true, pos_k, pos_s, pos_m, loader, picks, mode_s)

def main():
    seed_all(SEED)
    run_case("30→15", CFG["30_15"])
    run_case("60→30", CFG["60_30"])

if __name__ == "__main__":
    main()
