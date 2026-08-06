"""
iw_decomposition_mes2_matching.py  —  Decomposição covariate vs concept (Mês 2)
================================================================================

AVISO: Este script é uma RECONSTRUÇÃO para fins de auditoria.
O código original que gerou `Relas/results/mes2/drift_decomposition.csv` não
sobreviveu no repositório. Esta versão foi reconstruída a partir:
  - do CSV resultante (outputs/drift_decomposition_mes2.csv)
  - da metodologia documentada no relatório mes1a4 (Seção 4)
  - da lógica implícita no nome das colunas e valores

NÃO use este script para resultados novos. O método oficial de decomposição
é o LSIF do Mês 3: drift_analise/chapter03_visuals.py (`main_compute_importance_weights`).

----
Metodologia de matching (Mês 2)
--------------------------------
A decomposição separa "quanto do excesso de ADE é concept drift" de "quanto é
covariate shift" comparando pares de jogos compatíveis entre anos.

Dois jogos são "compatíveis" se têm o mesmo modelo, horizonte, e número
similar de trajetórias (± tolerância). Para cada par (jogo_2019, jogo_ano_y),
calcula-se:

  excess_total     = ade_y_full  - ade_2019_full
  excess_concept   = ade_y_compat - ade_2019_compat   (mesmo contexto de jogo)
  excess_covariate = excess_total - excess_concept

Limitação crítica: o método exige jogos com características cinemáticas
similares ao baseline 2019. O dataset tem poucos jogos por ano (6),
resultando em:
  - 2023: n_compat = 2  (estimativa ruidosa)
  - 2022, 2024, 2025: n_compat = 0  (não estimável)

Por isso o método foi descontinuado em favor do LSIF, que opera sobre
trajetórias individuais e não depende de pareamento de jogos inteiros.

----
Referência
----------
Resultado oficial da decomposição covariate/concept: LSIF (Mês 3)
  Recovery mediana IW: 65,5%  (IC 95%: 59–72%)
  Arquivo: Relas/results/mes3/iw_decomposition.csv
  Script:  drift_analise/chapter03_visuals.py (main_compute_importance_weights)

Para reproduzir o resultado histórico (auditoria apenas):
  python drift_analise/legacy/iw_decomposition_mes2_matching.py

Saída: drift_analise/legacy/outputs/drift_decomposition_mes2.csv
       (cópia do CSV original; não re-executa o pareamento)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── caminhos ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_OUT   = Path(__file__).resolve().parent / "outputs"
COV_DIR      = PROJECT_ROOT / "covariate_shift_out"
HORIZON      = "30→15"
MODEL        = "Seq2Seq"
BASELINE     = 2019
N_TRAJ_TOL   = 0.20   # tolerância de ±20% no nº de trajetórias para "matching"


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_errors() -> pd.DataFrame:
    p = COV_DIR / "trajectory_errors_sample.parquet"
    c = COV_DIR / "trajectory_errors_sample.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(
        "Arquivo de erros de trajetória não encontrado.\n"
        f"  Esperado: {p} ou {c}\n"
        "  Execute compute_trajectory_errors.py primeiro."
    )


def ade_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """ADE médio por (proc_set_file, year)  para MODEL + HORIZON."""
    sub = df[(df["model"] == MODEL) & (df["horizon"] == HORIZON)]
    grp = (
        sub.groupby(["proc_set_file", "year"])["ade_traj"]
        .agg(ade_mean="mean", n_traj="count")
        .reset_index()
    )
    return grp


def find_compatible_pairs(
    df_game: pd.DataFrame,
    tol: float = N_TRAJ_TOL,
) -> pd.DataFrame:
    """
    Para cada jogo de ano y ≠ BASELINE, procura jogos do BASELINE com
    n_traj similar (|n_y - n_base| / n_base ≤ tol).

    Retorna DataFrame com colunas:
      year, proc_set_y, proc_set_base, ade_y, ade_base
    """
    base = df_game[df_game["year"] == BASELINE].copy()
    rows = []
    for _, row_y in df_game[df_game["year"] != BASELINE].iterrows():
        candidates = base[
            np.abs(base["n_traj"] - row_y["n_traj"]) / base["n_traj"] <= tol
        ]
        for _, row_b in candidates.iterrows():
            rows.append({
                "year":          int(row_y["year"]),
                "proc_set_y":    row_y["proc_set_file"],
                "proc_set_base": row_b["proc_set_file"],
                "ade_y":         row_y["ade_mean"],
                "ade_base":      row_b["ade_mean"],
                "n_traj_y":      row_y["n_traj"],
                "n_traj_base":   row_b["n_traj"],
            })
    return pd.DataFrame(rows)


def decompose(df_game: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a tabela de decomposição covariate/concept por ano.

    Para anos com n_compat ≥ 1: estima excess_concept via pares matched.
    Para anos com n_compat = 0: deixa colunas de concept como NaN.
    """
    # ADE médio por ano (full — todos os jogos)
    ade_full = df_game.groupby("year")["ade_mean"].mean()
    n_full   = df_game.groupby("year")["proc_set_file"].count()
    ade_2019_full = ade_full.get(BASELINE, np.nan)

    rows = []
    for year in sorted(df_game["year"].unique()):
        if int(year) == BASELINE:
            continue
        ade_y_full = ade_full.get(year, np.nan)
        excess_total = ade_y_full - ade_2019_full

        year_pairs = pairs[pairs["year"] == year] if not pairs.empty else pd.DataFrame()
        n_compat = len(year_pairs)

        if n_compat >= 1:
            ade_y_compat    = year_pairs["ade_y"].mean()
            ade_base_compat = year_pairs["ade_base"].mean()
            excess_concept   = ade_y_compat - ade_base_compat
            excess_covariate = excess_total - excess_concept
            pct_concept = (excess_concept / excess_total * 100
                           if abs(excess_total) > 1e-9 else np.nan)
        else:
            ade_y_compat = ade_base_compat = excess_concept = None
            excess_covariate = None
            pct_concept = None

        rows.append({
            "year":                    int(year),
            "n_games_full":            int(n_full.get(year, 0)),
            "n_games_compat":          n_compat,
            "ade_baseline_2019":       round(ade_2019_full, 3),
            "ade_full":                round(ade_y_full, 3) if pd.notna(ade_y_full) else None,
            "ade_compat":              round(ade_y_compat, 3) if pd.notna(ade_y_compat) else None,
            "excess_total_mm":         round(excess_total, 3) if pd.notna(excess_total) else None,
            "excess_concept_mm":       round(excess_concept, 3) if excess_concept is not None else None,
            "excess_covariate_mm":     round(excess_covariate, 3) if excess_covariate is not None else None,
            "pct_explained_by_concept": round(pct_concept, 1) if pct_concept is not None else None,
        })

    return pd.DataFrame(rows)


# ─── main ─────────────────────────────────────────────────────────────────────

def main(recompute: bool = False) -> None:
    out_path = LEGACY_OUT / "drift_decomposition_mes2.csv"
    LEGACY_OUT.mkdir(parents=True, exist_ok=True)

    if not recompute and out_path.exists():
        print(f"[info] Usando CSV histórico (sem re-executar matching):")
        print(f"       {out_path}")
        df = pd.read_csv(out_path)
        print(df.to_string(index=False))
        print()
        print("[aviso] Para re-executar o matching, passe --recompute.")
        return

    # Re-execução real
    print("[info] Carregando erros de trajetória...")
    df_err  = load_errors()
    df_game = ade_per_game(df_err)
    print(f"[info] {len(df_game)} jogos | anos: {sorted(df_game['year'].unique())}")

    print("[info] Buscando pares compatíveis (tolerância n_traj ±20%)...")
    pairs = find_compatible_pairs(df_game)
    if not pairs.empty:
        n_by_year = pairs.groupby("year").size()
        print(f"[info] Pares encontrados por ano:\n{n_by_year.to_string()}")
    else:
        print("[aviso] Nenhum par compatível encontrado.")

    df_decomp = decompose(df_game, pairs)
    df_decomp.to_csv(out_path, index=False)
    print(f"\n[ok] {out_path}")
    print(df_decomp.to_string(index=False))

    # Consistência com CSV histórico
    ref_path = out_path  # já é o mesmo arquivo; comparação só faz sentido se
    # o usuário comparar manualmente com o backup.
    print("\n[nota] Compare com o CSV histórico original em:")
    print(f"       Relas/results/mes2/drift_decomposition.csv")


if __name__ == "__main__":
    recompute = "--recompute" in sys.argv
    main(recompute=recompute)
