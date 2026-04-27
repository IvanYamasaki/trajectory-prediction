import numpy as np

def parse_list(xs):
    out = []
    for v in xs:
        if isinstance(v, str):
            v = v.strip().replace("%", "").replace(",", ".")
        out.append(float(v))
    return np.array(out, dtype=float)

def bootstrap_delta_ci(a, b, n_boot=50000, ci=0.95, seed=42):
    """
    Two-sample bootstrap CI for delta = mean(b) - mean(a)
    Returns: (ci_low, delta_hat, ci_high)
    """
    A = parse_list(a)
    B = parse_list(b)
    if A.size < 2 or B.size < 2:
        raise ValueError("Need at least 2 values in each list.")

    rng = np.random.default_rng(seed)

    delta_hat = B.mean() - A.mean()

    idx_a = rng.integers(0, A.size, size=(n_boot, A.size))
    idx_b = rng.integers(0, B.size, size=(n_boot, B.size))
    boot_delta = B[idx_b].mean(axis=1) - A[idx_a].mean(axis=1)

    alpha = 1.0 - ci
    lo = np.quantile(boot_delta, alpha / 2)
    hi = np.quantile(boot_delta, 1 - alpha / 2)

    return float(lo), float(delta_hat), float(hi)

def print_delta(base_year, base_vals, year, vals, n_boot=50000, ci=0.95, seed=1):
    lo, d, hi = bootstrap_delta_ci(base_vals, vals, n_boot=n_boot, ci=ci, seed=seed)
    print(f"{year}-{base_year} {lo:.2f} {d:.2f} {hi:.2f}")

# 15,30
y2019 = ["68.69%","70.16%","76.18%","73.48%","70.05%","71.64%"]
y2021 = ["72.44%","70.37%","71.79%","77.82%","71.80%","69.82%"]
y2023 = ["76.05%","77.82%","76.03%","74.62%","75.32%","76.02%"]
y2025 = ["78.45%","78.73%","78.63%","75.56%","76.11%","75.73%"]

# 30,60
y2019 = ["40.03%","45.11%","54.17%","50.74%","42.14%","47.07%"]
y2021 = ["47.46%","37.94%","42.03%","54.55%","45.77%","35.78%"]
y2023 = ["52.03%","51.43%","50.58%","47.05%","50.00%","49.88%"]
y2025 = ["55.43%","54.12%","55.96%","50.03%","48.43%","50.93%"]

# ---- imprime deltas vs 2019 ----
print_delta(2019, y2019, 2021, y2021, seed=1)
print_delta(2019, y2019, 2023, y2023, seed=1)
print_delta(2019, y2019, 2025, y2025, seed=1)
