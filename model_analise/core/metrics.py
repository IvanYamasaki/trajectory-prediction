"""Métricas por trajetória (ADE/FDE em mm) e métricas do protocolo Mês 7.

Código movido verbatim de compute_trajectory_errors.py, ewc_finetune.py e
retrain_at_breakpoints.py. Importável sem TensorFlow (os modelos chegam
já construídos por parâmetro).

NOTA: o cat_ratio do Mês 3 (retrain_at_breakpoints: n_deg/max(n_imp,1))
tem semântica DIFERENTE de ``cat_ratio_fraction`` (fração de trajetórias
degradadas, Mês 7) e permanece em retrain_at_breakpoints.py.
"""
from __future__ import annotations

import numpy as np


def per_traj_metrics(pos_pred, pos_true):
    """ADE/FDE por trajetória a partir de posições em mm.

    Verbatim de compute_trajectory_errors.per_traj_metrics.
    """
    d = pos_pred - pos_true
    dist = np.linalg.norm(d, axis=-1)  # [N, look_forth]
    ade = np.mean(dist, axis=1)        # [N]
    fde = dist[:, -1]                  # [N]
    return ade.astype(np.float32), fde.astype(np.float32)


def compute_ade(model, x, y_v, loader, batch=512) -> float:
    """Retorna ADE médio em mm. Verbatim de ewc_finetune.compute_ade."""
    pred_v = model.predict(x, batch_size=batch, verbose=0)
    pos_pred = loader.convert_batch(x, pred_v.astype(np.float32))
    pos_true = loader.convert_batch(x, y_v)
    d = np.linalg.norm(pos_pred - pos_true, axis=-1)
    return float(np.mean(np.mean(d, axis=1)))


def per_traj_ade(model, x, y_v, loader, batch=512):
    """ADE por trajetória (mm). Verbatim de retrain_at_breakpoints.per_traj_ade."""
    pred_v = model.predict(x, batch_size=batch, verbose=0)
    pos_pred = loader.convert_batch(x, pred_v.astype(np.float32))
    pos_true = loader.convert_batch(x, y_v)
    d = np.linalg.norm(pos_pred - pos_true, axis=-1)
    return np.mean(d, axis=1)


def original_per_traj_dist(model, x_base, y_base, loader, batch=512):
    """ADE por trajetória de ``model`` no regime antigo + posições verdadeiras.

    Bloco compartilhado de replay_finetune/targeted_finetune (d_orig e
    d_after), movido verbatim. Retorna ``(d, pos_true_b)``.
    """
    pred = model.predict(x_base, batch_size=batch, verbose=0)
    pos_true_b = loader.convert_batch(x_base, y_base)
    d = np.mean(np.linalg.norm(
        loader.convert_batch(x_base, pred.astype(np.float32)) - pos_true_b,
        axis=-1), axis=1)
    return d, pos_true_b


def cat_ratio_fraction(d_orig, d_after) -> float:
    """Fração de trajetórias do regime antigo que PIORARAM após a adaptação.

    Semântica do Mês 7 (ewc_finetune / replay_finetune / targeted_finetune):
    mean(d_after > d_orig), com d_* = ADE por trajetória (mm).
    """
    return float(np.mean(d_after > d_orig))


def recovery_pct(ade_before_base, ade_before_test, ade_after_test) -> float:
    """Percentual do excesso de erro recuperado no regime novo.

    Fórmula verbatim de ewc_finetune (eps=1e-10).
    """
    return 100.0 * (ade_before_test - ade_after_test) / (
        ade_before_test - ade_before_base + 1e-10)
