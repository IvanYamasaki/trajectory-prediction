"""Pure NumPy/SciPy reimplementation of ruptures Dynp/Pelt + CostRbf used in drift_analise.

Used when ``ruptures`` is not installed (e.g. Python 3.14 on Windows without C++ build tools).
Logic matches ``ruptures`` 1.1.x (CostRbf, Dynp.seg, Pelt._seg) for reproducibility.
"""
from __future__ import annotations

from functools import lru_cache
from math import ceil, floor

import numpy as np
from scipy.spatial.distance import pdist, squareform

# --- exceptions (same names as ruptures for familiar tracebacks) ----------------


class NotEnoughPoints(Exception):
    pass


class BadSegmentationParameters(Exception):
    pass


# --- utils (ruptures.utils) -----------------------------------------------------


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)
    for b in it:
        yield a, b
        a = b


def sanity_check(n_samples: int, n_bkps: int, jump: int, min_size: int) -> bool:
    n_adm_bkps = n_samples // jump
    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True


# --- Cost RBF (ruptures.costs.costrbf.CostRbf) ---------------------------------


class CostRbf:
    model = "rbf"
    min_size = 1

    def __init__(self, gamma: float | None = None) -> None:
        self.gamma = gamma
        self._gram: np.ndarray | None = None
        self.signal: np.ndarray | None = None

    @property
    def gram(self) -> np.ndarray:
        if self._gram is None:
            assert self.signal is not None
            K = pdist(self.signal, metric="sqeuclidean")
            if self.gamma is None:
                self.gamma = 1.0
                k_median = float(np.median(K))
                if k_median != 0.0:
                    self.gamma = 1.0 / k_median
            K = K * float(self.gamma)
            np.clip(K, 1e-2, 1e2, K)
            self._gram = np.exp(squareform(-K))
        return self._gram

    def fit(self, signal: np.ndarray) -> CostRbf:
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = np.asarray(signal, dtype=float)
        self._gram = None
        if self.gamma is None:
            _ = self.gram
        return self

    def error(self, start: int, end: int) -> float:
        if self.signal is None:
            raise RuntimeError("CostRbf.fit must be called before error()")
        if end - start < self.min_size:
            raise NotEnoughPoints
        g = self.gram
        sub_gram = g[start:end, start:end]
        val = float(np.diagonal(sub_gram).sum())
        val -= float(sub_gram.sum()) / (end - start)
        return val

    def sum_of_costs(self, bkps: list[int]) -> float:
        return float(
            sum(self.error(s, e) for s, e in pairwise([0] + list(bkps)))
        )


def cost_factory(model: str = "l2", **params):
    if model == "rbf":
        return CostRbf(**params)
    raise NotImplementedError(
        f"changepoint_numpy_fallback only implements model='rbf', got {model!r}"
    )


# --- Dynp (ruptures.detection.dynp.Dynp) ----------------------------------------


class Dynp:
    def __init__(
        self,
        model: str = "l2",
        custom_cost=None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
    ) -> None:
        if custom_cost is not None:
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, int(self.cost.min_size))
        self.jump = jump
        self.n_samples: int | None = None

    @lru_cache(maxsize=None)
    def seg(self, start: int, end: int, n_bkps: int) -> dict:
        jump, min_size = self.jump, self.min_size

        if n_bkps == 0:
            cost = self.cost.error(start, end)
            return {(start, end): cost}
        multiple_of_jump = (k for k in range(start, end) if k % jump == 0)
        admissible_bkps: list[int] = []
        for bkp in multiple_of_jump:
            n_samples = bkp - start
            if sanity_check(
                n_samples=n_samples,
                n_bkps=n_bkps - 1,
                jump=jump,
                min_size=min_size,
            ):
                if end - bkp >= min_size:
                    admissible_bkps.append(bkp)

        if not admissible_bkps:
            raise RuntimeError(
                f"No admissible last breakpoints: start={start}, end={end}, n_bkps={n_bkps}."
            )

        sub_problems: list[dict] = []
        for bkp in admissible_bkps:
            left_partition = self.seg(start, bkp, n_bkps - 1)
            right_partition = self.seg(bkp, end, 0)
            tmp_partition = dict(left_partition)
            tmp_partition[(bkp, end)] = right_partition[(bkp, end)]
            sub_problems.append(tmp_partition)

        return min(sub_problems, key=lambda d: sum(d.values()))

    def fit(self, signal: np.ndarray) -> Dynp:
        self.seg.cache_clear()
        self.cost.fit(signal)
        self.n_samples = int(signal.shape[0])
        return self

    def predict(self, n_bkps: int) -> list[int]:
        if self.n_samples is None:
            raise RuntimeError("Dynp.fit must be called before predict()")
        if self.cost.signal is None:
            raise RuntimeError("Cost not fitted")
        n_s = int(self.cost.signal.shape[0])
        if not sanity_check(
            n_samples=n_s,
            n_bkps=n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        partition = self.seg(0, self.n_samples, n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps


# --- Pelt (ruptures.detection.pelt.Pelt) ----------------------------------------


class Pelt:
    def __init__(
        self,
        model: str = "l2",
        custom_cost=None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
    ) -> None:
        if custom_cost is not None:
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, int(self.cost.min_size))
        self.jump = jump
        self.n_samples: int | None = None

    def _seg(self, pen: float) -> dict:
        partitions: dict = {}
        partitions[0] = {(0, 0): 0}
        admissible: list[int] = []

        ind = [k for k in range(0, int(self.n_samples), self.jump) if k >= self.min_size]
        ind += [int(self.n_samples)]

        for bkp in ind:
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(int(new_adm_pt))

            subproblems: list[dict] = []
            for t in admissible:
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:
                    continue
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
            ]

        best_partition = dict(partitions[int(self.n_samples)])
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal: np.ndarray) -> Pelt:
        sig = np.asarray(signal, dtype=float)
        self.cost.fit(sig)
        if sig.ndim == 1:
            (n_samples,) = sig.shape
        else:
            n_samples = int(sig.shape[0])
        self.n_samples = n_samples
        return self

    def predict(self, pen: float) -> list[int]:
        if self.n_samples is None:
            raise RuntimeError("Pelt.fit must be called before predict()")
        sig = self.cost.signal
        if sig is None:
            raise RuntimeError("Cost not fitted")
        n_s = int(sig.shape[0])
        if not sanity_check(
            n_samples=n_s,
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        partition = self._seg(float(pen))
        bkps = sorted(e for s, e in partition.keys())
        return bkps
