"""Carga de proc_sets para os scripts de fine-tuning/avaliação.

Unificação de ewc_finetune.load_proc_sets e
retrain_at_breakpoints.load_data_sets (idênticos exceto pelo
``idx.sort()`` e pelo default de ``max_per``). Recebe a lista de
proc_sets por parâmetro — nunca lê PROC_YEAR internamente.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_proc_sets(ns: list[int], loader, max_per: int = 6000, seed: int = 42,
                   sort_idx: bool = False):
    """Concatena janelas dos proc_sets ``ns`` (subamostrando até ``max_per``).

    ``sort_idx=True`` reproduz retrain_at_breakpoints.load_data_sets
    (índices ordenados após a subamostragem); ``False`` reproduz
    ewc_finetune.load_proc_sets.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for n in ns:
        fpath = f"dataset/proc_set_{n}"
        if not Path(fpath + ".pkl").exists():
            continue
        try:
            x, _, _, y_v = loader.load_data([fpath], for_test=True)
        except Exception as e:
            print(f"  [warn] proc_set {n}: {e}")
            continue
        if len(x) == 0:
            continue
        if len(x) > max_per:
            idx = rng.choice(len(x), max_per, replace=False)
            if sort_idx:
                idx.sort()
            x, y_v = x[idx], y_v[idx]
        xs.append(x); ys.append(y_v)
    if not xs:
        return None, None
    return (np.concatenate(xs, 0).astype(np.float32),
            np.concatenate(ys, 0).astype(np.float32))
