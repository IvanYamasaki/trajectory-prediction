import numpy as np
import pandas as pd

from drift_analise.plot_covariate_shift import ks_stat_1d, wasserstein_1d


def test_ks_self_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(size=512)
    assert ks_stat_1d(x, x) < 1e-9


def test_wasserstein_translate():
    rng = np.random.default_rng(42)
    x = rng.normal(size=512)
    assert abs(wasserstein_1d(x, x + 5.0) - 5.0) < 1e-9


def test_per_game_context_shape():
    df = pd.read_csv("covariate_shift_out/per_game_context.csv")
    assert len(df) == 36
    assert "avg_robots_per_stoppage" in df.columns
