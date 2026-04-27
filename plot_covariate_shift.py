# plot_covariate_shift.py
from email.mime import base
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

BASELINE_DATASETS = [
    "dataset/proc_set_1",
    "dataset/proc_set_2",
]
DATASETS = [
    "dataset/proc_set_3",
    "dataset/proc_set_4",
    "dataset/proc_set_5",
    "dataset/proc_set_6",
    "dataset/proc_set_7",
    "dataset/proc_set_8",
    "dataset/proc_set_9",
    "dataset/proc_set_10",
    "dataset/proc_set_11",
    "dataset/proc_set_12",
    "dataset/proc_set_13",
    "dataset/proc_set_14",
    "dataset/proc_set_15",
    "dataset/proc_set_16",
    "dataset/proc_set_17",
    "dataset/proc_set_18",
    "dataset/proc_set_19",
    "dataset/proc_set_20",
    "dataset/proc_set_21",
    "dataset/proc_set_22",
    "dataset/proc_set_23",
    "dataset/proc_set_24",
    "dataset/proc_set_25",
    "dataset/proc_set_26",
    "dataset/proc_set_27",
    "dataset/proc_set_28",
    "dataset/proc_set_29",
    "dataset/proc_set_30",
    "dataset/proc_set_31",
    "dataset/proc_set_32",
    "dataset/proc_set_33",
    "dataset/proc_set_34",
    "dataset/proc_set_35",
    "dataset/proc_set_36",
    "dataset/proc_set_37",
    "dataset/proc_set_38"
]

YEARS = [
    2019, 2019, 2019, 2019, 2019, 2019, 
    2021, 2021, 2021, 2021, 2021, 2021,
    2023, 2023, 2023, 2023, 2023, 2023,
    2025, 2025, 2025, 2025, 2025, 2025,
    2022, 2022, 2022, 2022, 2022, 2022,
    2024, 2024, 2024, 2024, 2024, 2024
]


OUTDIR = "covariate_shift_out"
BASELINE_YEAR = None  # None = menor ano disponível
INCLUDE_BALL = False
MAX_SAMPLES = 200000
SEED = 1

# ---------- utils ----------
def parse_year_from_filename(path: str):
    m = re.search(r"(19|20)\d{2}", os.path.basename(path))
    return int(m.group(0)) if m else None

def resolve_path(p):
    # aceita "dataset/proc_set_20" ou "dataset/proc_set_20.pkl"
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".pkl"):
        return p + ".pkl"
    raise FileNotFoundError(f"Arquivo não encontrado: {p} (nem {p}.pkl)")

def ecdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def wrap_angle(dtheta):
    return (dtheta + np.pi) % (2*np.pi) - np.pi

def downsample(x, max_n, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= max_n:
        return x
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]

def wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    n = 2000
    qs = np.linspace(0, 1, n)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))

def ks_stat_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    allx = np.sort(np.unique(np.concatenate([a, b])))
    Fa = np.searchsorted(a, allx, side="right") / a.size
    Fb = np.searchsorted(b, allx, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))

# ---------- extraction ----------
def load_proc(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_robot_trajs(proc, include_ball=False):
    for team in ["blue", "yellow"]:
        if team not in proc:
            continue
        vx_list = proc[team].get("speed", {}).get("x", [])
        vy_list = proc[team].get("speed", {}).get("y", [])
        stop_ids = proc[team].get("stop_id", [])
        n = min(len(vx_list), len(vy_list), len(stop_ids))
        for i in range(n):
            vx = np.asarray(vx_list[i], dtype=float)
            vy = np.asarray(vy_list[i], dtype=float)
            if vx.size < 3 or vy.size < 3:
                continue
            yield {"source": team, "stop_id": stop_ids[i], "vx": vx, "vy": vy}

    if include_ball and "ball" in proc and isinstance(proc["ball"], dict):
        for stop_id, seg in proc["ball"].items():
            vx = np.asarray(seg.get("v_x", []), dtype=float)
            vy = np.asarray(seg.get("v_y", []), dtype=float)
            if vx.size < 3 or vy.size < 3:
                continue
            yield {"source": "ball", "stop_id": stop_id, "vx": vx, "vy": vy}

def traj_features(vx, vy):
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    speed = np.hypot(vx, vy)
    accel = np.hypot(np.diff(vx), np.diff(vy))
    theta = np.arctan2(vy, vx)
    turn = np.abs(wrap_angle(np.diff(theta)))

    def safe_stats(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"mean": np.nan, "p50": np.nan, "p90": np.nan}
        return {
            "mean": float(np.mean(x)),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
        }

    return {
        "vx": safe_stats(vx),
        "vy": safe_stats(vy),
        "speed": safe_stats(speed),
        "accel": safe_stats(accel),
        "turn": safe_stats(turn),
        "n": int(speed.size),
    }

def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    files = [resolve_path(p) for p in DATASETS]

    # define features UMA vez, antes de usar
    features = ["vx", "vy", "speed", "accel", "turn"]

    # resolve anos
    years = []
    if YEARS and len(YEARS) != len(files):
        raise ValueError("YEARS deve ter o mesmo tamanho de DATASETS (ou ficar vazio).")

    for i, fp in enumerate(files):
        if YEARS:
            y = YEARS[i]
        else:
            y = parse_year_from_filename(fp)
        if y is None:
            raise ValueError(f"Não consegui inferir ano do arquivo: {fp}. Preencha YEARS.")
        years.append(y)

    # coleta amostras por ano
    by_year = {}
    per_traj_rows = []
    file_stats = {}
    by_file = {}
    for fp, y in zip(files, years):
        proc = load_proc(fp)
        for tr in iter_robot_trajs(proc, include_ball=INCLUDE_BALL):
            vx = tr["vx"]
            vy = tr["vy"]

            speed = np.hypot(vx, vy)
            accel = np.hypot(np.diff(vx), np.diff(vy))
            theta = np.arctan2(vy, vx)
            turn = np.abs(wrap_angle(np.diff(theta)))

            # --- por ano ---
            by_year.setdefault(y, {}).setdefault("vx", []).append(vx)
            by_year.setdefault(y, {}).setdefault("vy", []).append(vy)
            by_year[y].setdefault("speed", []).append(speed)
            by_year[y].setdefault("accel", []).append(accel)
            by_year[y].setdefault("turn", []).append(turn)

            # --- por arquivo (DATASETS) ---
            fname = os.path.basename(fp)
            by_file.setdefault(fname, {}).setdefault("vx", []).append(vx)
            by_file.setdefault(fname, {}).setdefault("vy", []).append(vy)
            by_file[fname].setdefault("speed", []).append(speed)
            by_file[fname].setdefault("accel", []).append(accel)
            by_file[fname].setdefault("turn", []).append(turn)
            by_file[fname]["year"] = y

            file_stats.setdefault(os.path.basename(fp), {"speed": [], "accel": []})
            file_stats[os.path.basename(fp)]["speed"].append(speed)
            file_stats[os.path.basename(fp)]["accel"].append(accel)

            feats = traj_features(vx, vy)
            per_traj_rows.append({
                "year": y,
                "file": os.path.basename(fp),
                "source": tr["source"],
                "stop_id": tr["stop_id"],
                "n": feats["n"],
                "vx_mean": feats["vx"]["mean"], "vx_p50": feats["vx"]["p50"], "vx_p90": feats["vx"]["p90"],
                "vy_mean": feats["vy"]["mean"], "vy_p50": feats["vy"]["p50"], "vy_p90": feats["vy"]["p90"],
                "speed_mean": feats["speed"]["mean"], "speed_p50": feats["speed"]["p50"], "speed_p90": feats["speed"]["p90"],
                "accel_mean": feats["accel"]["mean"], "accel_p50": feats["accel"]["p50"], "accel_p90": feats["accel"]["p90"],
                "turn_mean": feats["turn"]["mean"], "turn_p50": feats["turn"]["p50"], "turn_p90": feats["turn"]["p90"],
            })

    for y in list(by_year.keys()):
        for k in list(by_year[y].keys()):
            by_year[y][k] = np.concatenate(by_year[y][k]) if len(by_year[y][k]) else np.array([])
    
    for fname in list(by_file.keys()):
        for k in ["vx", "vy", "speed", "accel", "turn"]:
            by_file[fname][k] = np.concatenate(by_file[fname][k]) if len(by_file[fname][k]) else np.array([])

    # ---------------- baseline = treino (proc_set_1 e proc_set_2) ----------------
    baseline_files = [resolve_path(p) for p in BASELINE_DATASETS]
    baseline = {feat: [] for feat in features}

    for fp in baseline_files:
        proc = load_proc(fp)
        for tr in iter_robot_trajs(proc, include_ball=INCLUDE_BALL):
            vx = tr["vx"]
            vy = tr["vy"]
            baseline["vx"].append(vx)
            baseline["vy"].append(vy)
            baseline["speed"].append(np.hypot(vx, vy))
            baseline["accel"].append(np.hypot(np.diff(vx), np.diff(vy)))
            theta = np.arctan2(vy, vx)
            baseline["turn"].append(np.abs(wrap_angle(np.diff(theta))))

    for feat in features:
        baseline[feat] = downsample(np.concatenate(baseline[feat]), MAX_SAMPLES, rng)

    base = baseline
    all_years_sorted = sorted(by_year.keys())

    CLIP_Q = 0.999  # 99.9% do baseline
    clip_hi = {feat: float(np.quantile(base[feat], CLIP_Q)) for feat in features}

    def clip_feat(x, feat):
        x = np.asarray(x, float)
        if feat == "speed":
            return np.clip(x, a_min=None, a_max=8000.0)
        if feat == "accel":
            return np.clip(x, a_min=None, a_max=500.0)
        return x

    # ---------------- KS por arquivo (DATASETS) vs baseline(train) ----------------
    ks_file_rows = []
    ks_per_file = {}  # Armazenar KS por arquivo para usar no print
    files_sorted = sorted(by_file.keys(), key=lambda fn: (by_file[fn].get("year", 0), fn))

    for fname in files_sorted:
        y = by_file[fname].get("year", "")
        ks_per_file[fname] = {"speed": None, "accel": None}  # Inicializar
        for feat in ["speed", "accel", "turn"]:
            cur = downsample(by_file[fname][feat], MAX_SAMPLES, rng)

            # robust (usa seu clip_feat)
            b_clip = clip_feat(base[feat], feat)
            c_clip = clip_feat(cur, feat)

            ks_value = ks_stat_1d(b_clip, c_clip)
            ks_file_rows.append({
                "file": fname,
                "year": y,
                "feature": feat,
                "ks": ks_value,
            })
            
            # Armazenar KS de speed e accel para o print
            if feat in ["speed", "accel"]:
                ks_per_file[fname][feat] = ks_value

    # ---------------- KS + Wasserstein por arquivo (DATASETS) vs baseline(train) - CSV ----------------
    ks_wd_rows = []
    for fname in files_sorted:
        y = by_file[fname].get("year", "")
        for feat in ["speed", "accel", "turn"]:
            cur = downsample(by_file[fname][feat], MAX_SAMPLES, rng)
            b_clip = clip_feat(base[feat], feat)
            c_clip = clip_feat(cur, feat)
            ks_v = ks_stat_1d(b_clip, c_clip)
            wd_v = wasserstein_1d(b_clip, c_clip)
            ks_wd_rows.append({
                "file": fname,
                "year": y,
                "feature": feat,
                "ks": ks_v,
                "wasserstein": wd_v,
            })

    # (CSV write for ks_wd will be performed after write_csv is defined)

    def plot_ks_per_file(feat, rows, outpath):
        r = [x for x in rows if x["feature"] == feat]
        labels = [f'{x["year"]}-{x["file"].replace(".pkl","")}' for x in r]
        vals = [x["ks"] for x in r]

        plt.figure(figsize=(max(10, 0.45*len(vals)), 5))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylim(0, 1.0)
        plt.ylabel("KS (vs baseline train)")
        plt.title(f"KS per DATASET - {feat}")
        plt.grid(True, axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    plot_ks_per_file("speed", ks_file_rows, os.path.join(OUTDIR, "ks_per_dataset_speed.png"))
    plot_ks_per_file("accel", ks_file_rows, os.path.join(OUTDIR, "ks_per_dataset_accel.png"))
    plot_ks_per_file("turn", ks_file_rows, os.path.join(OUTDIR, "ks_per_dataset_turn.png"))

    print("-> ks_per_dataset_speed.png / ks_per_dataset_accel.png / ks_per_dataset_turn.png")

    # ---------------- drift vs baseline(train) ----------------
    drift_rows_raw = []
    drift_rows_robust = []

    for y in all_years_sorted:
        for feat in features:
            cur = downsample(by_year[y][feat], MAX_SAMPLES, rng)

            # RAW
            drift_rows_raw.append({
                "year": y,
                "feature": feat,
                "wasserstein": wasserstein_1d(base[feat], cur),
                "ks": ks_stat_1d(base[feat], cur),
            })

            # ROBUST (clip no quantil do baseline)
            b_clip = clip_feat(base[feat], feat)
            c_clip = clip_feat(cur, feat)

            drift_rows_robust.append({
                "year": y,
                "feature": feat,
                "wasserstein": wasserstein_1d(b_clip, c_clip),
                "ks": ks_stat_1d(b_clip, c_clip),
                "clip_q": CLIP_Q,
                "clip_hi": clip_hi[feat],
            })
        
    def write_csv(path, rows, header):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(h, "")) for h in header) + "\n")

    write_csv(os.path.join(OUTDIR, "drift_vs_baseline.csv"), drift_rows_raw,
            header=["year", "feature", "wasserstein", "ks"])

    write_csv(os.path.join(OUTDIR, "drift_vs_baseline_robust.csv"), drift_rows_robust,
            header=["year", "feature", "wasserstein", "ks", "clip_q", "clip_hi"])

    # write ks+wasserstein per dataset (moved here because write_csv is defined above)
    try:
        write_csv(os.path.join(OUTDIR, "ks_wd_per_dataset.csv"), ks_wd_rows,
                  header=["file", "year", "feature", "ks", "wasserstein"])
    except NameError:
        # ks_wd_rows may not exist in some flows; ignore if missing
        pass

    # ---------------- ECDF plots (baseline=train) ----------------
    for feat in ["speed", "accel", "turn"]:
        plt.figure(figsize=(10, 6))

        # baseline como referência visual (CLIP AQUI)
        xb = downsample(base[feat], MAX_SAMPLES, rng)
        xb = clip_feat(xb, feat)  # <<<
        xs, ys = ecdf(xb)
        if xs.size > 0:
            plt.plot(xs, ys, linewidth=2, label="baseline(train)")

        for y in all_years_sorted:
            x = downsample(by_year[y][feat], MAX_SAMPLES, rng)
            x = clip_feat(x, feat)  # <<<
            xs, ys = ecdf(x)
            if xs.size == 0:
                continue
            plt.plot(xs, ys, linewidth=2, label=str(y))

        plt.title(f"ECDF - {feat} (baseline=train)")
        plt.xlabel(feat)
        plt.ylabel("F(x)")
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"ecdf_{feat}.png"), dpi=200)
        plt.close()

    # ---------------- Hist overlays ----------------
    def plot_hist_overlay(
        name, baseline_arr, year_to_arr, years_sorted, outdir,
        bins=140, qlo=0.001, qhi=0.999, min_x=None
    ):
        """
        Histograma como PDF (density=True): área total = 1.
        Altura do bin = (freq_relativa_no_bin) / (largura_do_bin).
        min_x: se não for None, remove valores < min_x (útil p/ tirar pico em 0).
        """
        def prep(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if min_x is not None:
                x = x[x >= float(min_x)]
            return x

        baseline_arr = prep(baseline_arr)
        year_to_arr = {y: prep(year_to_arr[y]) for y in years_sorted}

        plt.figure(figsize=(10, 6))

        all_samples = [baseline_arr] + [year_to_arr[y] for y in years_sorted]
        all_cat = np.concatenate([a for a in all_samples if a.size > 0]) if any(a.size > 0 for a in all_samples) else np.array([])
        if all_cat.size == 0:
            return

        lo = np.quantile(all_cat, qlo)
        hi = np.quantile(all_cat, qhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(all_cat)), float(np.max(all_cat))

        # baseline
        plt.hist(
            baseline_arr, bins=bins, range=(lo, hi),
            density=True, histtype="step", linewidth=2,
            label="baseline(train)"
        )

        # anos
        for y in years_sorted:
            plt.hist(
                year_to_arr[y], bins=bins, range=(lo, hi),
                density=True, histtype="step", linewidth=2,
                label=str(y)
            )

        plt.title(f"Histogram (PDF) - {name} (baseline=train)")
        plt.xlabel(name)
        plt.ylabel("density (area=1)")
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{name}.png"), dpi=200)
        plt.close()

    years_sorted = all_years_sorted
    speed_year = {y: downsample(by_year[y]["speed"], MAX_SAMPLES, rng) for y in years_sorted}
    accel_year = {y: downsample(by_year[y]["accel"], MAX_SAMPLES, rng) for y in years_sorted}

    speed_base = downsample(base["speed"], MAX_SAMPLES, rng)
    accel_base = downsample(base["accel"], MAX_SAMPLES, rng)

    # plot_hist_overlay("speed", speed_base, speed_year, years_sorted, OUTDIR, bins=140)
    # plot_hist_overlay("accel", accel_base, accel_year, years_sorted, OUTDIR, bins=140)
    plot_hist_overlay("speed", speed_base, speed_year, years_sorted, OUTDIR, bins=140, min_x=0.5)
    plot_hist_overlay("accel", accel_base, accel_year, years_sorted, OUTDIR, bins=140, min_x=0.05)
    
    def summarize_arr(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return {
            "n": int(x.size),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
            "frac_gt_100": float(np.mean(x > 100.0)),
        }

    print("\n=== Per-file stats (suspeitos) ===")
    rows = []
    for fname, d in file_stats.items():
        sp = summarize_arr(np.concatenate(d["speed"]))
        ac = summarize_arr(np.concatenate(d["accel"]))
        ks_sp = ks_per_file.get(fname, {}).get("speed", None)
        ks_ac = ks_per_file.get(fname, {}).get("accel", None)
        rows.append((fname, sp["p99"], sp["max"], sp["frac_gt_100"], ac["p99"], ac["max"], ks_sp, ks_ac))

    # ordena por max speed
    rows.sort(key=lambda r: r[2], reverse=True)

    for r in rows:
        fname, sp_p99, sp_max, sp_frac, ac_p99, ac_max, ks_sp, ks_ac = r
        ks_sp_str = f"ks_speed={ks_sp:.4f}" if ks_sp is not None else "ks_speed=N/A"
        ks_ac_str = f"ks_accel={ks_ac:.4f}" if ks_ac is not None else "ks_accel=N/A"
        print(f"{fname}  speed_p99={sp_p99:.2f} speed_max={sp_max:.2f} frac(speed>100)={sp_frac:.4f}  accel_p99={ac_p99:.2f} accel_max={ac_max:.2f}  {ks_sp_str}  {ks_ac_str}")
        

    print("OK!")
    print("->", os.path.join(OUTDIR, "drift_vs_baseline_raw.csv"))
    print("->", os.path.join(OUTDIR, "drift_vs_baseline_robust.csv"))
    print("-> ks_per_dataset_speed.png / ks_per_dataset_accel.png / ks_per_dataset_turn.png")
    if per_traj_rows:
        print("->", os.path.join(OUTDIR, "per_trajectory_features.csv"))
    print("-> ecdf_speed.png / ecdf_accel.png / ecdf_turn.png")
    print("-> hist_speed.png / hist_accel.png")

if __name__ == "__main__":
    main()
