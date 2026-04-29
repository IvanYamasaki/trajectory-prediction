import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dataset.load_dataset import LoadDataSet
from model_analise.ai_model.predictor import RobotOnlyPredictor

LOOK_BACK = 30
LOOK_FORTH = 15
UNITS = 128
MODEL_NAME = "robot_30_15_t"
TEST_FILES = ["dataset/proc_set_1"]

BATCH = 256
SAMPLES_PLOT = 6
SEED = 7


def seed_all(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_loader():
    return LoadDataSet(look_back=LOOK_BACK, look_forth=LOOK_FORTH)


def build_model():
    return RobotOnlyPredictor(
        units=UNITS,
        look_back=LOOK_BACK,
        look_forth=LOOK_FORTH,
        result_dims=2,
        forcing=True
    )

def try_load_weights(model, model_name=MODEL_NAME):
    candidates = [
        model_name,
        f"{model_name}.weights.h5",
        f"{model_name}.h5",
        os.path.join("weights", f"{model_name}.weights.h5"),
        os.path.join("weights", f"{model_name}.h5"),
        os.path.join("models", model_name),
        os.path.join("models", f"{model_name}.weights.h5"),
        os.path.join("models", f"{model_name}.h5"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                model.load_weights(p)
                print(f"[OK] load_weights: {p}")
                return True
            except Exception:
                pass
    print("[WARN] não achei pesos; vou rodar sem carregar (só sanity/baselines).")
    return False


def denorm_v(loader, v_norm):
    return v_norm * loader.robots_std[2:4] + loader.robots_avg[2:4]


def denorm_pos(loader, pos_norm):
    return pos_norm * loader.robots_std[0:2] + loader.robots_avg[0:2]


def integrate(last_pos_mm, v_mm_per_frame):
    return last_pos_mm + np.cumsum(v_mm_per_frame, axis=0)


def integrate_batch(last_pos_mm, v_mm_per_frame):
    return last_pos_mm[:, None, :] + np.cumsum(v_mm_per_frame, axis=1)


def baseline_last_v(robot_x_norm, loader):
    v0_norm = robot_x_norm[:, -1:, 2:4]
    v0 = denorm_v(loader, v0_norm)
    v = np.repeat(v0, LOOK_FORTH, axis=1)
    return v


def metrics_from_positions(pos_pred, pos_true):
    step_err = np.linalg.norm(pos_pred - pos_true, axis=-1)
    ade = np.mean(step_err, axis=1)
    fde = step_err[:, -1]
    jump = step_err[:, 0]
    return {
        "jump_mean": float(np.mean(jump)),
        "ade_mean": float(np.mean(ade)),
        "fde_mean": float(np.mean(fde)),
    }


def summarize_array(name, a):
    a = np.asarray(a)
    print(f"{name}: shape={a.shape}  mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")


def sanity_dataset(robot_x, robot_y, loader, n=5000):
    n = min(n, len(robot_x))
    idx = np.random.choice(len(robot_x), n, replace=False)
    x = robot_x[idx]
    y = robot_y[idx]

    summarize_array("x_norm", x)
    summarize_array("y_norm", y)

    v_true = denorm_v(loader, y)
    speeds = np.linalg.norm(v_true.reshape(-1, 2), axis=1)
    print(f"speed(mm/frame): mean={speeds.mean():.2f} std={speeds.std():.2f} p50={np.percentile(speeds,50):.2f} p90={np.percentile(speeds,90):.2f} max={speeds.max():.2f}")

    last_pos = denorm_pos(loader, x[:, -1, 0:2])
    pos_true = integrate_batch(last_pos, v_true)

    dpos = pos_true[:, 1:, :] - pos_true[:, :-1, :]
    summarize_array("dpos_from_integrated(mm)", dpos)

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        print("[BAD] tem NaN/Inf no dataset")
    else:
        print("[OK] sem NaN/Inf no dataset")


def eval_model_vs_baselines(model, robot_x, robot_y, loader, n=2000):
    n = min(n, len(robot_x))
    idx = np.random.choice(len(robot_x), n, replace=False)
    x = robot_x[idx]
    y = robot_y[idx]

    last_pos = denorm_pos(loader, x[:, -1, 0:2])
    v_true = denorm_v(loader, y)
    pos_true = integrate_batch(last_pos, v_true)

    v_base = baseline_last_v(x, loader)
    pos_base = integrate_batch(last_pos, v_base)
    base_m = metrics_from_positions(pos_base, pos_true)

    preds_norm = model.predict(x, verbose=0)
    v_pred = denorm_v(loader, preds_norm)
    pos_pred = integrate_batch(last_pos, v_pred)
    model_m = metrics_from_positions(pos_pred, pos_true)

    print("\n== Baseline (repete v no último frame) ==")
    print(base_m)
    print("== Modelo ==")
    print(model_m)

    delta = {k: model_m[k] - base_m[k] for k in model_m}
    print("== Modelo - Baseline ==")
    print(delta)

    summarize_array("preds_norm", preds_norm)
    vpred_flat = v_pred.reshape(-1, 2)
    print(f"pred_speed(mm/frame): mean={np.linalg.norm(vpred_flat,axis=1).mean():.2f} std={np.linalg.norm(vpred_flat,axis=1).std():.2f}")

    return idx[:SAMPLES_PLOT], x, y, preds_norm


def overfit_one_batch(model, robot_x, robot_y, steps=300):
    x = robot_x[:BATCH]
    y = robot_y[:BATCH]
    model.forcing = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    losses = []
    for i in range(steps):
        out = model.train_on_batch(x, y, return_dict=True)
        losses.append(float(out["batch_loss"]))
    print(f"\n[Overfit 1 batch] loss start={losses[0]:.6f} end={losses[-1]:.6f}")
    return losses


def plot_cases(loader, x, y, preds_norm, idx_show):
    for i, idx in enumerate(idx_show):
        xi = x[i:i+1]
        yi = y[i:i+1]
        pi = preds_norm[i:i+1]

        past = denorm_pos(loader, xi[0, :, 0:2])
        last_pos = past[-1]
        v_true = denorm_v(loader, yi[0])
        v_pred = denorm_v(loader, pi[0])

        pos_true = integrate(last_pos, v_true)
        pos_pred = integrate(last_pos, v_pred)

        step_err = np.linalg.norm(pos_pred - pos_true, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(past[:, 0], past[:, 1], 'b-o', label="Passado")
        ax1.plot(pos_true[:, 0], pos_true[:, 1], 'g-o', label="Real")
        ax1.plot(pos_pred[:, 0], pos_pred[:, 1], 'r-o', label="Predito")
        ax1.set_title(f"Amostra {int(idx)}")
        ax1.axis("equal")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(range(1, LOOK_FORTH + 1), step_err, 'm-o')
        ax2.set_xlabel("Passo futuro")
        ax2.set_ylabel("Erro (mm)")
        ax2.set_title("Erro por timestep")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def plot_speed_hist(loader, robot_y, n=20000):
    n = min(n, len(robot_y))
    idx = np.random.choice(len(robot_y), n, replace=False)
    v = denorm_v(loader, robot_y[idx]).reshape(-1, 2)
    s = np.linalg.norm(v, axis=1)
    plt.figure(figsize=(8, 4))
    plt.hist(s, bins=60)
    plt.title("Histograma de velocidade (mm/frame)")
    plt.xlabel("speed")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    seed_all()

    loader = build_loader()
    robot_x, _, _, robot_y = loader.load_data(TEST_FILES, for_test=True)

    print("== Dataset ==")
    print(f"robot_x: {robot_x.shape}  robot_y: {robot_y.shape}")
    sanity_dataset(robot_x, robot_y, loader)
    plot_speed_hist(loader, robot_y)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    try_load_weights(model)

    print("\n== Modelo vs baselines ==")
    idx_show, x_sub, y_sub, p_sub = eval_model_vs_baselines(model, robot_x, robot_y, loader)

    print("\n== Plot de casos ==")
    plot_cases(loader, x_sub[:SAMPLES_PLOT], y_sub[:SAMPLES_PLOT], p_sub[:SAMPLES_PLOT], idx_show)

    print("\n== Overfit 1 batch (debug pipeline) ==")
    losses = overfit_one_batch(model, robot_x, robot_y, steps=250)

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("Overfit 1 batch: loss por step")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
