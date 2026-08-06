"""
chapter01_descriptive_pipeline.py — Capítulo 1 (degradação + figuras descritivas)
================================================================================
Agrega plot_covariate_shift e descriptive_figures. Exporta `ks_stat_1d` e
`wasserstein_1d` para testes.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.metrics import (
    ks_stat_1d, wasserstein_1d, bootstrap_ci_mean, bootstrap_linreg_ci,
)
from common.plotting import T, make_plot_style, save_fig_bilingual
from drift_analise.drift_output_paths import CH01, COVARIATE_SHIFT_OUT, ensure_chapter_dirs

ROOT = PROJECT_ROOT
DATASET_DIR = ROOT / "drift_analise" / "dataset"
COV_DIR = ROOT / "covariate_shift_out"
MODEL_DIR = ROOT / "model"
OUT_DIR = CH01


# --- plot_covariate_shift ---

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


# CSV / pipeline artifacts stay in covariate_shift_out; figures go to CH01 (Relas/results/drift/…).
DATA_OUTDIR = os.fspath(COVARIATE_SHIFT_OUT)
FIGURES_OUTDIR = os.fspath(CH01)
BASELINE_YEAR = 2019   # valor efetivo (era redefinido mais abaixo)
INCLUDE_BALL = False
MAX_SAMPLES = 200000
SEED = 42              # valor efetivo (era redefinido mais abaixo)

# ---- novos parâmetros (Mês 1, revisado pós Mês 2) ----
# N=500 dava 1 janela por jogo (granularidade efetivamente "anual"), mascarando
# variação intra-jogo. Reduzido p/ 30 -> tipicamente 5-10 janelas por jogo,
# habilita KSWIN e mostra evolução intra-jogo nos plots 2a/2b.
N_TRAJS_PER_WINDOW = 30
FRAME_RATE_HZ = 60         # cadência aproximada das logs SSL para estimativa de duração
DATASET_CSV_PATH = "drift_analise/dataset/dataset.csv"  # usado p/ mapear proc_set_N -> log_file
DIVISION_MAP_PATH = "division_map.csv"

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

# ks_stat_1d / wasserstein_1d vem de common.metrics (re-exportados aqui
# para notebooks/tests antigos)

def infer_division_from_log_file(log_file):
    if not log_file:
        return ""
    if re.search(r"(DIV[_ -]?A|Division[-_ ]?A)", log_file, flags=re.IGNORECASE):
        return "A"
    if re.search(r"(DIV[_ -]?B|Division[-_ ]?B)", log_file, flags=re.IGNORECASE):
        return "B"
    return ""

def load_division_map(path=DIVISION_MAP_PATH):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    import csv as _csv
    with open(path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            proc_set_file = (row.get("proc_set_file") or "").strip()
            division = (row.get("division") or "").strip().upper()
            if proc_set_file and division in {"A", "B"}:
                mapping[proc_set_file] = division
    return mapping

# ---------- extraction ----------
def load_proc(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_robot_trajs(proc, include_ball=False, chrono=True):
    """Itera trajetórias dos robôs.
    Se ``chrono=True`` (default), ordena cronologicamente por ``time_c[i][0]``
    intercalando os times. Se ``chrono=False``, mantém o comportamento legado
    (todo o ``blue`` e depois todo o ``yellow``).
    """
    if chrono:
        items = []
        for team in ["blue", "yellow"]:
            if team not in proc:
                continue
            vx_list = proc[team].get("speed", {}).get("x", [])
            vy_list = proc[team].get("speed", {}).get("y", [])
            stop_ids = proc[team].get("stop_id", [])
            time_c   = proc[team].get("time_c", [])
            n = min(len(vx_list), len(vy_list), len(stop_ids))
            for i in range(n):
                vx = np.asarray(vx_list[i], dtype=float)
                vy = np.asarray(vy_list[i], dtype=float)
                if vx.size < 3 or vy.size < 3:
                    continue
                # marca temporal do início da trajetória (fallback p/ índice se faltar)
                if i < len(time_c):
                    tc = time_c[i]
                    try:
                        t0 = float(tc[0]) if hasattr(tc, "__len__") and len(tc) > 0 else float(i)
                    except Exception:
                        t0 = float(i)
                else:
                    t0 = float(i)
                items.append((t0, team, stop_ids[i], vx, vy))
        items.sort(key=lambda r: r[0])
        for t0, team, stop_id, vx, vy in items:
            yield {"source": team, "stop_id": stop_id, "vx": vx, "vy": vy, "t0": t0}
    else:
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

def main_plot_covariate_shift():
    rng = np.random.default_rng(SEED)
    ensure_chapter_dirs()
    os.makedirs(DATA_OUTDIR, exist_ok=True)

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

    # === Mês 1: armazenar trajetórias individuais por arquivo p/ janelas e contexto ===
    by_file_trajs = {}

    for fp, y in zip(files, years):
        proc = load_proc(fp)
        fname = os.path.basename(fp)
        by_file_trajs.setdefault(fname, [])

        for tr in iter_robot_trajs(proc, include_ball=INCLUDE_BALL):
            vx = tr["vx"]
            vy = tr["vy"]

            speed = np.hypot(vx, vy)
            accel = np.hypot(np.diff(vx), np.diff(vy))
            theta = np.arctan2(vy, vx)
            turn = np.abs(wrap_angle(np.diff(theta)))

            by_file_trajs[fname].append({
                "speed": speed,
                "accel": accel,
                "turn": turn,
                "n_frames": int(speed.size),
                "source": tr["source"],
                "stop_id": tr["stop_id"],
                "year": y,
            })

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

    plot_ks_per_file("speed", ks_file_rows, os.path.join(FIGURES_OUTDIR, "ks_per_dataset_speed.png"))
    plot_ks_per_file("accel", ks_file_rows, os.path.join(FIGURES_OUTDIR, "ks_per_dataset_accel.png"))
    plot_ks_per_file("turn", ks_file_rows, os.path.join(FIGURES_OUTDIR, "ks_per_dataset_turn.png"))

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

    write_csv(os.path.join(DATA_OUTDIR, "drift_vs_baseline.csv"), drift_rows_raw,
            header=["year", "feature", "wasserstein", "ks"])

    write_csv(os.path.join(DATA_OUTDIR, "drift_vs_baseline_robust.csv"), drift_rows_robust,
            header=["year", "feature", "wasserstein", "ks", "clip_q", "clip_hi"])

    # write ks+wasserstein per dataset (moved here because write_csv is defined above)
    try:
        write_csv(os.path.join(DATA_OUTDIR, "ks_wd_per_dataset.csv"), ks_wd_rows,
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
        plt.savefig(os.path.join(FIGURES_OUTDIR, f"ecdf_{feat}.png"), dpi=200)
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
    plot_hist_overlay("speed", speed_base, speed_year, years_sorted, FIGURES_OUTDIR, bins=140, min_x=0.5)
    plot_hist_overlay("accel", accel_base, accel_year, years_sorted, FIGURES_OUTDIR, bins=140, min_x=0.05)
    
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
        

    # =====================================================================
    # === MÊS 1 — CSVs novos para o notebook drift_analise.ipynb ===
    # =====================================================================
    proc_to_logfile = {}
    proc_to_teams = {}
    division_map = load_division_map()
    if os.path.exists(DATASET_CSV_PATH):
        try:
            import csv as _csv
            with open(DATASET_CSV_PATH, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    n = row.get("dataset")
                    lf = row.get("log_file")
                    tm = row.get("teams")
                    if not n or not lf:
                        continue
                    key = f"proc_set_{n.strip()}.pkl"
                    proc_to_logfile.setdefault(key, lf)
                    proc_to_teams.setdefault(key, tm or "")
        except Exception as e:
            print(f"[warn] nao consegui ler {DATASET_CSV_PATH}: {e}")

    def _short_match_id(year, fname, log_file, teams):
        import re as _re
        if teams:
            t = teams.replace(" vs ", "_vs_").replace(" ", "")
            t = _re.sub(r"[^A-Za-z0-9_]+", "", t)
            return f"{year}_{t}"
        if log_file:
            base_ = os.path.basename(log_file).replace(".log", "")
            base_ = _re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_", "", base_)
            base_ = _re.sub(r"[^A-Za-z0-9_-]+", "_", base_)
            return f"{year}_{base_}"[:80]
        return f"{year}_{fname.replace('.pkl','')}"

    def _stats_block(arr, prefix):
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {f"{prefix}_{k}": np.nan for k in ["mean", "p50", "p90", "p99", "max"]}
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_p50":  float(np.quantile(x, 0.50)),
            f"{prefix}_p90":  float(np.quantile(x, 0.90)),
            f"{prefix}_p99":  float(np.quantile(x, 0.99)),
            f"{prefix}_max":  float(np.max(x)),
        }

    # ---- per_game_context.csv ----
    ctx_rows = []
    for fname in files_sorted:
        trajs = by_file_trajs.get(fname, [])
        y = by_file[fname].get("year", "")
        if not trajs:
            continue
        speed_all = np.concatenate([t["speed"] for t in trajs])
        accel_all = np.concatenate([t["accel"] for t in trajs])
        turn_all  = np.concatenate([t["turn"]  for t in trajs])

        n_traj = len(trajs)
        unique_stop_ids = len(set((t["source"], t["stop_id"]) for t in trajs))
        # Proxy honesto: avg de trajetórias / stop_id_único. NÃO é nº médio de
        # robôs em campo (esse exigiria metadata oficial). Útil só como
        # ordenação grosseira de "intensidade de play".
        avg_robots_per_stoppage = float(n_traj) / max(1, unique_stop_ids)
        total_frames = int(sum(t["n_frames"] for t in trajs))
        duration_s = total_frames / float(FRAME_RATE_HZ)

        log_file = proc_to_logfile.get(fname, "")
        teams    = proc_to_teams.get(fname, "")
        match_id = _short_match_id(y, fname, log_file, teams)
        division = infer_division_from_log_file(log_file) or division_map.get(fname, "")

        row = {
            "match_id": match_id,
            "log_file": log_file,
            "proc_set_file": fname,
            "year": y,
            "division": division,
            "avg_robots_per_stoppage": round(avg_robots_per_stoppage, 4),
            "n_trajectories": n_traj,
            "n_unique_stop_ids": unique_stop_ids,
            "duration_estimate_s": round(duration_s, 2),
            "frame_rate_hz": FRAME_RATE_HZ,
        }
        row.update(_stats_block(speed_all, "speed"))
        row.update(_stats_block(accel_all, "accel"))
        ts = _stats_block(turn_all, "turn")
        row["turn_mean"] = ts["turn_mean"]
        row["turn_p50"]  = ts["turn_p50"]
        row["turn_p90"]  = ts["turn_p90"]
        ctx_rows.append(row)

    ctx_header = [
        "match_id", "log_file", "proc_set_file", "year", "division",
        "avg_robots_per_stoppage", "n_trajectories", "n_unique_stop_ids",
        "duration_estimate_s", "frame_rate_hz",
        "speed_mean", "speed_p50", "speed_p90", "speed_p99", "speed_max",
        "accel_mean", "accel_p50", "accel_p90", "accel_p99", "accel_max",
        "turn_mean",  "turn_p50",  "turn_p90",
    ]
    write_csv(os.path.join(DATA_OUTDIR, "per_game_context.csv"), ctx_rows, header=ctx_header)
    print("->", os.path.join(DATA_OUTDIR, "per_game_context.csv"))

    # ---- per_window_drift.csv ----
    win_rows = []
    base_clipped = {feat: clip_feat(base[feat], feat) for feat in ["speed", "accel", "turn"]}

    for fname in files_sorted:
        trajs = by_file_trajs.get(fname, [])
        if not trajs:
            continue
        y = by_file[fname].get("year", "")
        log_file = proc_to_logfile.get(fname, "")
        teams    = proc_to_teams.get(fname, "")
        match_id = _short_match_id(y, fname, log_file, teams)

        N = N_TRAJS_PER_WINDOW
        n_total = len(trajs)
        n_win = max(1, int(np.ceil(n_total / N)))

        for w in range(n_win):
            chunk = trajs[w * N:(w + 1) * N]
            if not chunk:
                continue
            speed_w = np.concatenate([t["speed"] for t in chunk])
            accel_w = np.concatenate([t["accel"] for t in chunk])
            turn_w  = np.concatenate([t["turn"]  for t in chunk])
            speed_w = clip_feat(downsample(speed_w, MAX_SAMPLES, rng), "speed")
            accel_w = clip_feat(downsample(accel_w, MAX_SAMPLES, rng), "accel")
            turn_w  = clip_feat(downsample(turn_w,  MAX_SAMPLES, rng), "turn")
            row = {
                "match_id": match_id,
                "log_file": log_file,
                "proc_set_file": fname,
                "year": y,
                "window_id": w,
                "n_trajectories_in_window": len(chunk),
                "ks_speed_window":  ks_stat_1d(base_clipped["speed"], speed_w),
                "ks_accel_window":  ks_stat_1d(base_clipped["accel"], accel_w),
                "ks_turn_window":   ks_stat_1d(base_clipped["turn"],  turn_w),
                "wd_speed_window":  wasserstein_1d(base_clipped["speed"], speed_w),
                "wd_accel_window":  wasserstein_1d(base_clipped["accel"], accel_w),
                "wd_turn_window":   wasserstein_1d(base_clipped["turn"],  turn_w),
            }
            win_rows.append(row)

    win_header = [
        "match_id", "log_file", "proc_set_file", "year", "window_id",
        "n_trajectories_in_window",
        "ks_speed_window", "ks_accel_window", "ks_turn_window",
        "wd_speed_window", "wd_accel_window", "wd_turn_window",
    ]
    write_csv(os.path.join(DATA_OUTDIR, "per_window_drift.csv"), win_rows, header=win_header)
    print("->", os.path.join(DATA_OUTDIR, "per_window_drift.csv"))

    print("OK!")
    print("->", os.path.join(DATA_OUTDIR, "drift_vs_baseline_raw.csv"))
    print("->", os.path.join(DATA_OUTDIR, "drift_vs_baseline_robust.csv"))
    print("-> ks_per_dataset_speed.png / ks_per_dataset_accel.png / ks_per_dataset_turn.png")
    if per_traj_rows:
        print("->", os.path.join(DATA_OUTDIR, "per_trajectory_features.csv"))
    print("-> ecdf_speed.png / ecdf_accel.png / ecdf_turn.png")
    print("-> hist_speed.png / hist_accel.png")


# --- descriptive_figures ---

PLOT_STYLE = make_plot_style(**{
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
})
plt.rcParams.update(PLOT_STYLE)


LANGS = ["pt", "en"]
EXCLUDE_YEARS = {2020, 2021}
N_BOOT = 2000
CI = 95
# ----- from notebook cell 4 -----
# Helpers reutilizados em todo o notebook
# bootstrap_ci_mean / bootstrap_linreg_ci / T vem de common (mesmos defaults)

def save_fig(fig, name):
    """Salva 1 figura em PT e EN (PDF) na OUT_DIR.
    `name` deve ser sem extensão; sufixos `_pt.pdf` e `_en.pdf` são adicionados.
    """
    save_fig_bilingual(fig, name, OUT_DIR)


def run_descriptive_figures() -> None:
    # ----- from notebook cell 6 -----
    # Verifica os arquivos esperados; aviso (não erro fatal) se faltar.
    EXPECTED = {
        "drift_analise/dataset/dataset.csv":            DATASET_DIR / "dataset.csv",
        "covariate_shift_out/ks_wd_per_dataset.csv":    COV_DIR / "ks_wd_per_dataset.csv",
        "covariate_shift_out/per_game_context.csv":     COV_DIR / "per_game_context.csv",
        "covariate_shift_out/per_window_drift.csv":     COV_DIR / "per_window_drift.csv",
        "covariate_shift_out/trajectory_errors_sample": (
            COV_DIR / "trajectory_errors_sample.parquet",
            COV_DIR / "trajectory_errors_sample.csv",
        ),
    }

    print("=" * 64)
    print("Pré-checagem")
    print("=" * 64)
    missing = []
    for label, p in EXPECTED.items():
        if isinstance(p, tuple):
            ok = any(pp.exists() for pp in p)
            shown = next((str(pp) for pp in p if pp.exists()), str(p[0]))
        else:
            ok = p.exists()
            shown = str(p)
        flag = "OK " if ok else "FAL"
        print(f"  [{flag}] {label:50s} -> {shown}")
        if not ok:
            missing.append(label)

    if missing:
        print("\n[!] Arquivos ausentes:")
        for m_ in missing:
            print(f"    - {m_}")
        print("\nRode os scripts antes do notebook:")
        print("    python drift_analise/chapter01_descriptive_pipeline.py covariate        # gera per_game_context.csv e per_window_drift.csv")
        print("    python model_analise/compute_trajectory_errors.py   # gera trajectory_errors_sample.parquet")
    else:
        print("\n[ok] Tudo pronto para análise.")

    # ----- from notebook cell 8 -----
    df = pd.read_csv(DATASET_DIR / "dataset.csv")
    df_ctx = pd.read_csv(COV_DIR / "per_game_context.csv")

    df["year"]  = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    print("dataset.csv         shape:", df.shape, "  jogos únicos:", df["log_file"].nunique())
    print("per_game_context.csv shape:", df_ctx.shape, "  jogos únicos:", df_ctx["log_file"].nunique())
    df_ctx.head(3)

    # ----- from notebook cell 9 -----
    # JOIN por log_file -- mantém todas as linhas do dataset.csv (left join)
    ctx_cols = [c for c in df_ctx.columns if c != "log_file"]
    df_all = df.merge(df_ctx, on="log_file", how="left", suffixes=("", "_ctx"))

    # salva a versão enriquecida (na raiz do projeto, não na pasta de figuras)
    df_all.to_csv(DATASET_DIR / "dataset_enriched.csv", index=False)
    print("dataset_enriched.csv salvo em:", DATASET_DIR / "dataset_enriched.csv")

    # DataFrame filtrado para análises principais (excluir 2020 e 2021)
    df_all_filtered = df_all[~df_all["year"].isin(EXCLUDE_YEARS)].copy()
    print("Anos no df_all:         ", sorted(df_all["year"].dropna().unique().tolist()))
    print("Anos no df_all_filtered:", sorted(df_all_filtered["year"].dropna().unique().tolist()))

    # ----- from notebook cell 10 -----
    # Inspeção rápida das colunas novas vindas do contexto
    new_cols = [c for c in df_ctx.columns if c not in {"log_file"}]
    print("Colunas novas no enriched:", new_cols)
    df_all.head(3)

    # ----- from notebook cell 12 -----
    df_win = pd.read_csv(COV_DIR / "per_window_drift.csv")
    df_win["year"] = pd.to_numeric(df_win["year"], errors="coerce")

    print("per_window_drift.csv  shape:", df_win.shape)
    print("Janelas por ano:")
    print(df_win.groupby("year")["window_id"].count().rename("n_janelas"))

    # ordem temporal global das janelas (ano crescente, depois match_id, depois window_id)
    df_win = df_win.sort_values(["year", "match_id", "window_id"]).reset_index(drop=True)
    df_win["global_idx"] = np.arange(len(df_win))
    df_win.head()

    # ----- from notebook cell 13 -----
    # Mapa de cores estável por ano (usado nas seções 2 e 4)
    years_all = sorted(df_win["year"].dropna().unique().astype(int).tolist())
    cmap = plt.get_cmap("viridis")
    YEAR_COLORS = {int(y): cmap(i / max(1, len(years_all) - 1)) for i, y in enumerate(years_all)}


    def make_plot_window_timeline(metric, lang):
        fig, ax = plt.subplots(figsize=(11, 5))
        for y in years_all:
            sub = df_win[df_win["year"] == y]
            ax.scatter(sub["global_idx"], sub[metric],
                       color=YEAR_COLORS[y], s=18, alpha=0.7, label=str(y))
        ax.set_xlabel(T("Janela (ordem temporal global)", "Window (global temporal order)", lang))
        ax.set_ylabel(metric.replace("_window", "").upper().replace("_", " "))
        ax.set_title(T(f"Drift por janela — {metric}", f"Per-window drift — {metric}", lang))
        ax.legend(title=T("Ano", "Year", lang), ncol=2, fontsize=9, loc="best")
        return fig


    for metric in ["ks_speed_window", "wd_speed_window", "ks_accel_window", "wd_accel_window"]:
        for lang in LANGS:
            fig = make_plot_window_timeline(metric, lang)
            save_fig(fig, f"2a_window_timeline_{metric}")
    print("-> 2a_window_timeline_*.pdf")

    # ----- from notebook cell 14 -----
    def make_plot_window_box(metric, lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [df_win.loc[df_win["year"] == y, metric].dropna().values for y in years_all]
        bp = ax.boxplot(data, labels=[str(y) for y in years_all],
                        patch_artist=True, showfliers=True)
        for patch, y in zip(bp["boxes"], years_all):
            patch.set_facecolor(YEAR_COLORS[y]); patch.set_alpha(0.55)
        ax.set_xlabel(T("Ano", "Year", lang))
        ax.set_ylabel(metric.replace("_window", "").upper().replace("_", " "))
        ax.set_title(T(f"Distribuição intra-ano de {metric}",
                        f"Intra-year distribution of {metric}", lang))
        return fig


    for metric in ["ks_speed_window", "wd_speed_window", "ks_accel_window", "wd_accel_window"]:
        for lang in LANGS:
            fig = make_plot_window_box(metric, lang)
            save_fig(fig, f"2b_window_box_{metric}")
    print("-> 2b_window_box_*.pdf")

    # ----- from notebook cell 15 -----
    # Tabela: variabilidade intra-ano
    agg = (df_win
           .groupby("year")[["ks_speed_window", "wd_speed_window",
                              "ks_accel_window", "wd_accel_window"]]
           .agg([("p50", "median"), ("p90", lambda s: np.nanpercentile(s, 90))])
          )
    agg.columns = ["__".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT_DIR / "window_intra_year_var.csv", index=False)
    print("-> window_intra_year_var.csv")
    agg

    # ----- from notebook cell 17 -----
    parquet_path = COV_DIR / "trajectory_errors_sample.parquet"
    csv_path     = COV_DIR / "trajectory_errors_sample.csv"
    if parquet_path.exists():
        df_traj = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df_traj = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Nem trajectory_errors_sample.parquet nem .csv foram encontrados. "
            "Rode `python model_analise/compute_trajectory_errors.py` antes."
        )

    df_traj["year"] = pd.to_numeric(df_traj["year"], errors="coerce")
    print("Linhas:", len(df_traj))
    print(df_traj.groupby(["year", "model", "horizon"])["traj_id"].count()
          .rename("n_trajs"))
    df_traj.head()

    # ----- from notebook cell 18 -----
    # Histograma com escala log no eixo Y, comparando 2019 vs último ano disponível,
    # por (modelo, horizonte).
    years_in_traj = sorted(df_traj["year"].dropna().unique().astype(int).tolist())
    year_last = max([y for y in years_in_traj if y not in EXCLUDE_YEARS],
                    default=max(years_in_traj))


    def make_plot_ade_traj_hist(model, horizon, lang):
        fig, ax = plt.subplots(figsize=(8, 5))
        base = df_traj[(df_traj["model"] == model) &
                       (df_traj["horizon"] == horizon) &
                       (df_traj["year"] == BASELINE_YEAR)]["ade_traj"].dropna()
        last = df_traj[(df_traj["model"] == model) &
                       (df_traj["horizon"] == horizon) &
                       (df_traj["year"] == year_last)]["ade_traj"].dropna()
        if base.empty and last.empty:
            ax.text(0.5, 0.5, T("Sem dados", "No data", lang), ha="center", va="center")
            return fig

        pool = pd.concat([base, last])
        if pool.empty:
            ax.text(0.5, 0.5, T("Sem dados", "No data", lang), ha="center", va="center")
            return fig
        lo = np.nanpercentile(pool, 0.5)
        hi = np.nanpercentile(pool, 99.5)
        bins = np.linspace(max(0, lo), hi, 60) if hi > lo else 30

        if not base.empty:
            ax.hist(base, bins=bins, alpha=0.55, label=str(BASELINE_YEAR),
                    color=YEAR_COLORS.get(BASELINE_YEAR, "C0"), edgecolor="white")
        if not last.empty:
            ax.hist(last, bins=bins, alpha=0.55, label=str(year_last),
                    color=YEAR_COLORS.get(year_last, "C3"), edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel(T("ADE por trajetória (mm)", "Per-trajectory ADE (mm)", lang))
        ax.set_ylabel(T("Frequência (escala log)", "Frequency (log scale)", lang))
        ax.set_title(T(f"Distribuição de ADE — {model} {horizon}",
                        f"ADE distribution — {model} {horizon}", lang))
        ax.legend()
        return fig


    for model in ["Seq2Seq", "Kalman"]:
        for horizon in ["30→15", "60→30"]:
            for lang in LANGS:
                fig = make_plot_ade_traj_hist(model, horizon, lang)
                save_fig(fig, f"3a_ade_traj_hist_{model}_{horizon.replace('→','_')}")
    print("-> 3a_ade_traj_hist_*.pdf")

    # ----- from notebook cell 19 -----
    # Tabela tail_stats: p50, p90, p99 de ade_traj por (year, model, horizon)
    def _q(p):
        def _f(x): return float(np.nanpercentile(x, p))
        _f.__name__ = f"p{p}"
        return _f

    tail_stats = (df_traj
                  .groupby(["year", "model", "horizon"])["ade_traj"]
                  .agg([_q(50), _q(90), _q(99), "count"])
                  .reset_index()
                  .rename(columns={"count": "n_trajs"})
                 )
    tail_stats.to_csv(OUT_DIR / "tail_stats.csv", index=False)
    print("-> tail_stats.csv")
    tail_stats

    # ----- from notebook cell 20 -----
    def make_plot_p99_p50_ratio(lang):
        ratio = (df_traj
                 .groupby(["year", "model", "horizon"])["ade_traj"]
                 .agg([("p50", lambda s: np.nanpercentile(s, 50)),
                       ("p99", lambda s: np.nanpercentile(s, 99))])
                 .reset_index())
        ratio["p99_p50"] = ratio["p99"] / ratio["p50"]

        fig, ax = plt.subplots(figsize=(9, 5))
        for (model, horizon), sub in ratio.groupby(["model", "horizon"]):
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub["p99_p50"], marker="o",
                    label=f"{model} {horizon}")
        ax.set_xlabel(T("Ano", "Year", lang))
        ax.set_ylabel(T("Razão p99 / p50 (ADE)", "p99 / p50 ratio (ADE)", lang))
        ax.set_title(T("Cauda relativa do erro ao longo dos anos",
                        "Relative error tail across years", lang))
        ax.legend()
        return fig


    for lang in LANGS:
        fig = make_plot_p99_p50_ratio(lang)
        save_fig(fig, "3c_p99_p50_ratio")
    print("-> 3c_p99_p50_ratio_*.pdf")

    # ----- from notebook cell 22 -----
    # DataFrame "1 linha por (log_file, year, model, horizon)" só com ADE,
    # já com colunas enriquecidas do contexto do jogo.
    df_ade_game = (df_all_filtered[
            (df_all_filtered["metric"] == "ADE")
            & (df_all_filtered["model"].isin(["Seq2Seq", "Kalman"]))
        ]
        .groupby(["log_file", "year", "model", "horizon"], as_index=False)
        .agg({
            "value": "mean",
            "ks_speed": "first", "ks_accel": "first", "ks_turn": "first",
            "wd_speed": "first", "wd_accel": "first", "wd_turn": "first",
            "speed_p90": "first", "accel_p90": "first",
            "speed_p99": "first", "accel_p99": "first",
            "match_id":  "first",
        })
        .rename(columns={"value": "ade"})
    )
    print("Linhas df_ade_game:", len(df_ade_game))
    df_ade_game.head()

    # ----- from notebook cell 23 -----
    def make_plot_ade_year_per_game(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        sub_30 = df_ade_game[df_ade_game["horizon"] == "30→15"].copy()

        rng = np.random.default_rng(SEED)
        sub_30["x_jit"] = sub_30["year"] + rng.uniform(-0.18, 0.18, size=len(sub_30))

        model_colors = {"Seq2Seq": "#2A6FDB", "Kalman": "#D1495B"}
        for model, df_m in sub_30.groupby("model"):
            ax.scatter(df_m["x_jit"], df_m["ade"], color=model_colors[model],
                       alpha=0.45, s=28, label=T(f"{model} (jogo)", f"{model} (game)", lang))

            # média anual com IC bootstrap
            for y, df_y in df_m.groupby("year"):
                mu = df_y["ade"].mean()
                lo, hi = bootstrap_ci_mean(df_y["ade"].values)
                ax.errorbar(y, mu, yerr=[[mu - lo], [hi - mu]], fmt="o",
                            color=model_colors[model], ecolor=model_colors[model],
                            markersize=8, markeredgecolor="white", elinewidth=2,
                            capsize=4)

        ax.set_xlabel(T("Ano", "Year", lang))
        ax.set_ylabel(T("ADE — horizonte 30→15 (mm)", "ADE — 30→15 horizon (mm)", lang))
        ax.set_title(T("ADE por jogo — Seq2Seq vs Kalman (IC95% bootstrap nas médias)",
                        "Per-game ADE — Seq2Seq vs Kalman (95% bootstrap CI on means)", lang))
        ax.legend()
        return fig


    for lang in LANGS:
        fig = make_plot_ade_year_per_game(lang)
        save_fig(fig, "4a_ade_year_per_game")
    print("-> 4a_ade_year_per_game_*.pdf")

    # ----- from notebook cell 24 -----
    def make_plot_ade_vs_feature(feature, lang):
        sub = df_ade_game[(df_ade_game["model"] == "Seq2Seq")
                          & (df_ade_game["horizon"] == "30→15")
                          & df_ade_game[feature].notna()
                          & df_ade_game["ade"].notna()]
        if len(sub) < 3:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, T("Dados insuficientes", "Insufficient data", lang),
                    ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(8, 5))
        for y, dfy in sub.groupby("year"):
            ax.scatter(dfy[feature], dfy["ade"], color=YEAR_COLORS.get(int(y), "C0"),
                       s=42, alpha=0.85, label=str(int(y)),
                       edgecolor="white", linewidth=0.6)

        fit = bootstrap_linreg_ci(sub[feature].values, sub["ade"].values)
        if fit is not None:
            ax.plot(fit["x_grid"], fit["mean"], color="black", lw=2,
                    label=T("Regressão (média)", "Regression (mean)", lang))
            ax.fill_between(fit["x_grid"], fit["lo"], fit["hi"],
                            color="black", alpha=0.12,
                            label=T("IC95% bootstrap", "95% bootstrap CI", lang))

        pretty = {
            "speed_p90": T("p90 da velocidade do jogo (mm/s)",
                            "Game speed p90 (mm/s)", lang),
            "accel_p90": T("p90 da aceleração do jogo (mm/s²)",
                            "Game accel p90 (mm/s²)", lang),
            "accel_p99": T("p99 da aceleração do jogo (mm/s²)",
                            "Game accel p99 (mm/s²)", lang),
            "ks_speed":  T("KS de velocidade vs baseline",
                            "Speed KS vs baseline", lang),
            "ks_accel":  T("KS de aceleração vs baseline",
                            "Accel KS vs baseline", lang),
            "wd_speed":  T("Wasserstein de velocidade vs baseline",
                            "Speed Wasserstein vs baseline", lang),
            "wd_accel":  T("Wasserstein de aceleração vs baseline",
                            "Accel Wasserstein vs baseline", lang),
        }.get(feature, feature)

        ax.set_xlabel(pretty)
        ax.set_ylabel(T("ADE Seq2Seq 30→15 (mm)", "ADE Seq2Seq 30→15 (mm)", lang))
        ax.set_title(T(f"ADE vs {pretty}", f"ADE vs {pretty}", lang))
        ax.legend(ncol=2, fontsize=9)
        return fig


    for feature in ["speed_p90", "accel_p90", "accel_p99",
                    "ks_speed", "ks_accel", "wd_speed", "wd_accel"]:
        for lang in LANGS:
            fig = make_plot_ade_vs_feature(feature, lang)
            save_fig(fig, f"4b_ade_vs_{feature}")
    print("-> 4b_ade_vs_*.pdf")

    # ----- from notebook cell 25 -----
    def make_plot_corr_heatmap(lang):
        feats = ["ks_speed", "ks_accel", "ks_turn",
                 "wd_speed", "wd_accel", "wd_turn",
                 "speed_p90", "accel_p90"]
        # reconstrói visão "1 linha por jogo" com ADE de cada modelo
        pivot = (df_ade_game[df_ade_game["horizon"] == "30→15"]
                 .pivot_table(index=["log_file", "year"], columns="model",
                              values="ade", aggfunc="mean")
                 .reset_index()
                 .rename(columns={"Seq2Seq": "ade_seq2seq", "Kalman": "ade_kalman"}))
        ctx = (df_ade_game[df_ade_game["horizon"] == "30→15"]
               .groupby("log_file", as_index=False)[feats].first())
        M = pivot.merge(ctx, on="log_file", how="left")
        M = M[~M["year"].isin(EXCLUDE_YEARS)]
        cols = ["ade_seq2seq", "ade_kalman"] + feats
        corr = M[cols].corr(method="spearman")

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(T("Correlação Spearman (jogos, anos ≠ 2020/2021)",
                        "Spearman correlation (games, years ≠ 2020/2021)", lang))
        return fig


    for lang in LANGS:
        fig = make_plot_corr_heatmap(lang)
        save_fig(fig, "4d_corr_spearman")
    print("-> 4d_corr_spearman_*.pdf")

    # ----- from notebook cell 27 -----
    # Tabela resumo: por ano -> n_jogos, n_trajetorias_total, n_janelas_total
    ctx_g = df_ctx.copy()
    ctx_g["year"] = pd.to_numeric(ctx_g["year"], errors="coerce")

    summary = (ctx_g.groupby("year", as_index=False)
               .agg(n_jogos=("proc_set_file", "nunique"),
                    n_trajetorias_total=("n_trajectories", "sum"))
              )

    win_per_year = (df_win.groupby("year", as_index=False)
                    .agg(n_janelas_total=("window_id", "count")))

    summary = summary.merge(win_per_year, on="year", how="left").fillna({"n_janelas_total": 0})
    summary["n_janelas_total"] = summary["n_janelas_total"].astype(int)
    summary.to_csv(OUT_DIR / "summary_granularity.csv", index=False)
    print("-> summary_granularity.csv")
    summary

    # ----- from notebook cell 28 -----
    # Verificações soft (warnings via print, não levantam exceção)
    print("=" * 64)
    print("Sanidade")
    print("=" * 64)

    low_traj = df_ctx[df_ctx["n_trajectories"] < 10]
    if len(low_traj):
        print(f"[!] {len(low_traj)} jogo(s) com <10 trajetórias:")
        print(low_traj[["proc_set_file", "year", "n_trajectories"]].to_string(index=False))
    else:
        print("[ok] todos os jogos têm >= 10 trajetórias.")

    few_games = summary[summary["n_jogos"] < 2]
    if len(few_games):
        print(f"[!] {len(few_games)} ano(s) com <2 jogos:")
        print(few_games[["year", "n_jogos"]].to_string(index=False))
    else:
        print("[ok] todos os anos têm >= 2 jogos.")

    ks_wd_cols = ["ks_accel", "ks_speed", "ks_turn", "wd_accel", "wd_speed", "wd_turn"]
    nan_count = df_all[ks_wd_cols].isna().sum()
    if nan_count.sum() > 0:
        print("[!] NaN em colunas KS/WD do dataset_enriched:")
        print(nan_count[nan_count > 0])
    else:
        print("[ok] sem NaN em colunas KS/WD do dataset_enriched.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["covariate", "descriptive"])
    args = p.parse_args()
    ensure_chapter_dirs()
    if args.step == "covariate":
        main_plot_covariate_shift()
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        run_descriptive_figures()
