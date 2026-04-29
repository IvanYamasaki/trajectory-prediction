"""
division_validation.py  —  Frente 4 do Mês 3
=============================================

Valida se o drift temporal sobrevive ao corte por divisão SSL.

Pergunta: a diferença de ADE entre anos persiste dentro de uma divisão fixa,
ou desaparece quando se controla por divisão?

Entradas:
  covariate_shift_out/trajectory_errors_sample.parquet (ou .csv)
  drift_analise/division_map.csv
  Relas/results/mes2/drift_decomposition.csv

Saídas:
  Relas/results/mes3/by_division_ade_per_game.pdf
  Relas/results/mes3/by_division_decomposition.csv
  Relas/results/mes3/by_division_2x2_table.csv
  (texto) conclusão sobre sobrevivência do drift ao corte por divisão

Uso:
    MPLBACKEND=Agg python drift_analise/division_validation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

COV_DIR   = Path("covariate_shift_out")
OUT_DIR   = Path("Relas") / "results" / "mes3"
DIV_MAP   = Path("drift_analise") / "division_map.csv"
HORIZON   = "30→15"
MIN_GAMES = 2          # mínimo de proc_sets por divisão/ano para incluir no plot
BASELINE  = 2019


# ─── loaders ─────────────────────────────────────────────────────────────

def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(c)


def load_division_map() -> pd.DataFrame:
    if not DIV_MAP.exists():
        raise FileNotFoundError(f"Arquivo de divisão não encontrado: {DIV_MAP}")
    return pd.read_csv(DIV_MAP)


# ─── bootstrap ───────────────────────────────────────────────────────────

def boot_ci(values: np.ndarray, n_boot=2000, seed=42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(values.mean()), float(np.percentile(means, 97.5))


# ─── análise ─────────────────────────────────────────────────────────────

def build_ade_by_game_division(df: pd.DataFrame, div_map: pd.DataFrame) -> pd.DataFrame:
    """ADE médio por (proc_set_file, year, division)."""
    df_s = df[(df["model"] == "Seq2Seq") & (df["horizon"] == HORIZON)].copy()
    grp = (
        df_s.groupby(["proc_set_file", "year"])["ade_traj"]
            .mean().reset_index()
            .rename(columns={"ade_traj": "ade_mean"})
    )
    merged = grp.merge(div_map, on="proc_set_file", how="left")
    merged["division"] = merged["division"].fillna("Unknown")
    return merged


def ade_ci_by_year_division(df_game: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap CI95 do ADE por (year, division)."""
    rows = []
    for (year, div), grp in df_game.groupby(["year", "division"]):
        vals = grp["ade_mean"].to_numpy()
        n_games = len(vals)
        if n_games < MIN_GAMES:
            rows.append({"year": year, "division": div,
                         "ade_mean": float(np.mean(vals)),
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "n_games": n_games, "insufficient": True})
        else:
            lo, mean, hi = boot_ci(vals)
            rows.append({"year": year, "division": div,
                         "ade_mean": mean, "ci_lo": lo, "ci_hi": hi,
                         "n_games": n_games, "insufficient": False})
    return pd.DataFrame(rows).sort_values(["division", "year"])


# ─── plots ───────────────────────────────────────────────────────────────

COLOR_YEAR = {2019: "#1f77b4", 2021: "#ff7f0e", 2022: "#2ca02c",
              2023: "#d62728", 2024: "#9467bd", 2025: "#8c564b"}


def plot_ade_by_division(df_ci: pd.DataFrame, out_path: Path) -> None:
    divisions = sorted(df_ci["division"].unique())
    n_divs = len(divisions)
    fig, axes = plt.subplots(1, n_divs, figsize=(6 * n_divs, 4.5), sharey=True)
    if n_divs == 1:
        axes = [axes]
    for ax, div in zip(axes, divisions):
        sub = df_ci[df_ci["division"] == div].copy()
        for _, row in sub.iterrows():
            color = COLOR_YEAR.get(int(row["year"]), "gray")
            alpha = 0.4 if row["insufficient"] else 1.0
            ax.errorbar(
                row["year"], row["ade_mean"],
                yerr=[[row["ade_mean"] - row["ci_lo"]],
                       [row["ci_hi"] - row["ade_mean"]]]
                     if not row["insufficient"] else None,
                fmt="o", color=color, alpha=alpha, capsize=4,
            )
        ax.set_title(f"Division {div}", fontsize=12)
        ax.set_xlabel("Ano")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    axes[0].set_ylabel("ADE médio (mm)")
    fig.suptitle("ADE por Ano × Divisão — Seq2Seq 30→15", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ─── tabela 2×2 ──────────────────────────────────────────────────────────

def table_2x2(df_ci: pd.DataFrame, cutoff: int = 2022) -> pd.DataFrame:
    """Tabela ADE médio: (período: antes/depois cutoff) × (divisão)."""
    rows = []
    for div in sorted(df_ci["division"].unique()):
        sub = df_ci[df_ci["division"] == div]
        before = sub[sub["year"] < cutoff]["ade_mean"].mean()
        after  = sub[sub["year"] >= cutoff]["ade_mean"].mean()
        rows.append({"division": div,
                     f"ade_antes_{cutoff}": round(before, 3),
                     f"ade_apos_{cutoff}":  round(after,  3),
                     "delta": round(after - before, 3)})
    return pd.DataFrame(rows)


# ─── decomposição por divisão ─────────────────────────────────────────────

def decomp_by_division(df_ci: pd.DataFrame) -> pd.DataFrame:
    baseline = df_ci[df_ci["year"] == BASELINE].set_index("division")["ade_mean"]
    rows = []
    for div in sorted(df_ci["division"].unique()):
        sub = df_ci[(df_ci["division"] == div) & (df_ci["year"] > BASELINE)]
        ade_2019_div = baseline.get(div, np.nan)
        for _, r in sub.iterrows():
            excess = r["ade_mean"] - ade_2019_div
            rows.append({
                "division":    div,
                "year":        int(r["year"]),
                "ade_2019":    round(ade_2019_div, 4) if pd.notna(ade_2019_div) else np.nan,
                "ade_y":       round(r["ade_mean"], 4),
                "excess_mm":   round(excess, 4) if pd.notna(ade_2019_div) else np.nan,
                "n_games":     int(r["n_games"]),
                "insufficient":bool(r["insufficient"]),
            })
    return pd.DataFrame(rows)


# ─── main ────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_errors()
    div_map = load_division_map()
    print(f"[info] {len(df):,} trajetórias  |  "
          f"divisões mapeadas: {div_map['division'].value_counts().to_dict()}")

    df_game = build_ade_by_game_division(df, div_map)
    df_ci   = ade_ci_by_year_division(df_game)

    # plot ADE por divisão
    plot_ade_by_division(df_ci, OUT_DIR / "by_division_ade_per_game.pdf")

    # decomposição por divisão
    df_decomp = decomp_by_division(df_ci)
    decomp_path = OUT_DIR / "by_division_decomposition.csv"
    df_decomp.to_csv(decomp_path, index=False)
    print(f"[ok] {decomp_path}")
    print(df_decomp.to_string(index=False))

    # tabela 2×2
    df_2x2 = table_2x2(df_ci, cutoff=2022)
    path_2x2 = OUT_DIR / "by_division_2x2_table.csv"
    df_2x2.to_csv(path_2x2, index=False)
    print(f"\n[ok] {path_2x2}")
    print(df_2x2.to_string(index=False))

    # conclusão
    print("\n=== CONCLUSÃO ===")
    divs = sorted(df_ci[~df_ci["insufficient"]]["division"].unique())
    for div in divs:
        sub = df_ci[(df_ci["division"] == div) & ~df_ci["insufficient"]]
        if len(sub) < 2:
            print(f"  Division {div}: amostra insuficiente — não conclusivo.")
            continue
        ade_2019 = sub[sub["year"] == BASELINE]["ade_mean"].values
        ade_post = sub[sub["year"] > BASELINE]["ade_mean"].values
        if len(ade_2019) == 0 or len(ade_post) == 0:
            print(f"  Division {div}: sem baseline ou pós-baseline — não conclusivo.")
            continue
        delta = ade_post.mean() - ade_2019.mean()
        survives = "SIM" if delta > 0.5 else "NÃO" if delta < -0.5 else "PARCIAL/INCERTO"
        print(f"  Division {div}: delta={delta:+.2f} mm → drift sobrevive ao corte? {survives}")

    if len(divs) < 2:
        print("\n[nota] Division B ausente ou insuficiente nos dados — "
              "análise limitada a Division A.")


if __name__ == "__main__":
    main()
