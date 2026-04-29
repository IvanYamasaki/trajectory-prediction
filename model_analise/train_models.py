# train_models.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pickle, numpy as np, tensorflow as tf
from dataset.load_dataset import LoadDataSet
from model_analise.ai_model.predictor import RobotOnlyPredictor
from model_analise.ai_model.losses import PositionMMMetrics
from tensorflow.keras import mixed_precision

# -------------------- CONFIG --------------------
FILES_TRAIN = ["dataset/proc_set_1", "dataset/proc_set_2"]
UNITS = 128
BATCH_SIZE = 1024
VAL_SPLIT = 0.10
SEED = 7

EPOCHS_30_15 = 30
EPOCHS_60_30 = 30

SEQ_30_15 = "weights/robot_30_15_t.weights.h5"
SEQ_60_30 = "weights/robot_60_30_t.weights.h5"

MLP_30_15 = "weights/mlp_30_15.weights.h5"
MLP_60_30 = "weights/mlp_60_30.weights.h5"

RESUME_IF_EXISTS = True
USE_MIXED_PRECISION = True
USE_XLA = True
# ------------------------------------------------

def setup_tf():
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
    if USE_XLA:
        tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

def seed_all(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_norm_stats(loader, path):
    stats = {
        "avg": loader.robots_avg, "std": loader.robots_std,
        "ball_avg": loader.ball_avg, "ball_std": loader.ball_std
    }
    with open(path, "wb") as f:
        pickle.dump(stats, f)

def make_fixed_val_split(x, y, val_frac=VAL_SPLIT, seed=SEED):
    n = len(x)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

class EpochFDE(tf.keras.callbacks.Callback):
    def __init__(self, loader_mm, x_tr_ref, y_tr_ref, x_val_ref, y_val_ref):
        super().__init__()
        self.mm = PositionMMMetrics(loader_mm)
        self.x_tr_ref, self.y_tr_ref = x_tr_ref, y_tr_ref
        self.x_val_ref, self.y_val_ref = x_val_ref, y_val_ref

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_tr_pred = self.model.predict(self.x_tr_ref, verbose=0)
        _, tr_fde, _ = self.mm.metrics_mm(self.x_tr_ref, self.y_tr_ref, y_tr_pred)
        y_val_pred = self.model.predict(self.x_val_ref, verbose=0)
        _, val_fde, _ = self.mm.metrics_mm(self.x_val_ref, self.y_val_ref, y_val_pred)
        logs["train_FDE_mm"] = float(tr_fde)
        logs["val_FDE_mm"] = float(val_fde)

def build_seq2seq(look_back, look_forth):
    m = RobotOnlyPredictor(
        units=UNITS, look_back=look_back, look_forth=look_forth,
        result_dims=2, use_tf_function=False, forcing=False
    )
    m.forcing = False
    return m

def train_seq2seq_block(look_back, look_forth, epochs, out_weights):
    # treina em dv (o que te deu os melhores resultados)
    loader = LoadDataSet(look_back, look_forth, target="dv")
    x, _, _, y = loader.load_data(FILES_TRAIN, for_test=False)
    x = x.astype(np.float32); y = y.astype(np.float32)

    # salva stats do loader dv
    save_norm_stats(loader, f"model/normalization_stats_{look_back}_{look_forth}.pkl")

    x_tr, y_tr, x_val, y_val = make_fixed_val_split(x, y, VAL_SPLIT, SEED)

    model = build_seq2seq(look_back, look_forth)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=tf.keras.losses.MeanSquaredError(),
        run_eagerly=False
    )

    # build + resume
    _ = model(tf.zeros((1, look_back, 6), tf.float32), training=False)
    if RESUME_IF_EXISTS and os.path.exists(out_weights):
        model.load_weights(out_weights)

    fde_cb = EpochFDE(loader, x_tr[:512], y_tr[:512], x_val[:512], y_val[:512])

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_weights, monitor="val_FDE_mm", mode="min",
        save_best_only=True, save_weights_only=True, verbose=0
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_FDE_mm", mode="min",
        patience=12, restore_best_weights=True, verbose=0
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_FDE_mm", mode="min",
        factor=0.5, patience=4, min_lr=1e-5, verbose=0
    )

    print(f"\n--- Seq2Seq {look_back}->{look_forth}  target=dv  epochs={epochs}  batch={BATCH_SIZE}")
    model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[fde_cb, reduce, ckpt, early],
        verbose=1
    )

    model.load_weights(out_weights)
    print(f"[OK] saved: {out_weights}")

# -------------------- MLP (opcional) --------------------
def build_mlp(look_back, look_forth, h1=256, h2=1024, h3=256):
    inp = tf.keras.Input(shape=(look_back, 6))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(h1, activation="relu")(x)
    x = tf.keras.layers.Dense(h2, activation="relu")(x)
    x = tf.keras.layers.Dense(h3, activation="relu")(x)
    x = tf.keras.layers.Dense(look_forth * 2)(x)
    out = tf.keras.layers.Reshape((look_forth, 2))(x)
    return tf.keras.Model(inp, out, name=f"mlp_{look_back}_{look_forth}")

def train_mlp_block(look_back, look_forth, out_weights, epochs=10):
    loader = LoadDataSet(look_back, look_forth, target="pos")
    x, _, _, y = loader.load_data(FILES_TRAIN, for_test=False)
    x = x.astype(np.float32); y = y.astype(np.float32)
    x_tr, y_tr, x_val, y_val = make_fixed_val_split(x, y, VAL_SPLIT, SEED)

    model = build_mlp(look_back, look_forth)
    ckpt = tf.keras.callbacks.ModelCheckpoint(out_weights, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=0)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True, verbose=0)

    print(f"\n--- MLP {look_back}->{look_forth} (TRAIN_MLP=True)")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_tr, y_tr, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), shuffle=True, callbacks=[ckpt, early], verbose=1)

    model.load_weights(out_weights)
    print(f"[OK] saved: {out_weights}")
# --------------------------------------------------------

def main():
    seed_all(SEED)
    setup_tf()

    # train_seq2seq_block(30, 15, EPOCHS_30_15, SEQ_30_15)
    # train_seq2seq_block(60, 30, EPOCHS_60_30, SEQ_60_30)

    train_mlp_block(30, 15, MLP_30_15, epochs=10)
    train_mlp_block(60, 30, MLP_60_30, epochs=10)

if __name__ == "__main__":
    main()
