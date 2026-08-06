"""Construção/carregamento do Seq2Seq e do loader normalizado.

Código movido verbatim de retrain_at_breakpoints.py / ewc_finetune.py /
compute_trajectory_errors.py. TensorFlow é importado LAZY (dentro das
funções). Nenhuma função lê PROC_YEAR/constantes de experimento — tudo
chega por parâmetro.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from common.paths import norm_stats_path


def build_seq2seq(look_back, look_forth, units=128, weights=None, lr=1e-3):
    """Constrói (e opcionalmente carrega pesos de) um RobotOnlyPredictor.

    Corpo movido de retrain_at_breakpoints.build_model (idêntico em
    ewc_finetune.build_model, compute_trajectory_errors.load_seq2seq e
    compare.load_seq2seq).
    """
    import tensorflow as tf
    from model_analise.ai_model.predictor import RobotOnlyPredictor
    m = RobotOnlyPredictor(
        units=units, look_back=look_back, look_forth=look_forth,
        result_dims=2, use_tf_function=False, forcing=False,
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    _ = m(tf.zeros((1, look_back, 6), tf.float32), training=False)
    if weights:
        m.load_weights(str(weights))
    return m


def freeze_encoder(model, n_unfreeze: int = 2):
    """Congela todas as camadas exceto as últimas ``n_unfreeze``.

    n_unfreeze=1 -> só Dense final (baseline conservador do Mês 3)
    n_unfreeze=2 -> Dense final + camada anterior (pós-review M3 / Mês 7)

    Corpo movido de ewc_finetune.freeze_encoder (retrain_at_breakpoints
    diferia apenas no print).
    """
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-n_unfreeze:]:
        layer.trainable = True
    trainable_params = sum(
        np.prod(v.shape) for v in model.trainable_variables)
    print(f"  [freeze] n_unfreeze={n_unfreeze}, "
          f"trainable params={trainable_params:,}")


def load_loader(look_back, look_forth, stats_path=None):
    """LoadDataSet(target="dv") com stats de normalização fixas do treino.

    Corpo movido de compute_trajectory_errors.build_loader (versão que
    valida a existência do pkl de estatísticas).
    """
    from dataset.load_dataset import LoadDataSet
    if stats_path is None:
        stats_path = norm_stats_path(look_back, look_forth)
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
