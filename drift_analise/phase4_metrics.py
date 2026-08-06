"""
drift_analise/phase4_metrics.py
=================================
Fase 4 — Biblioteca de métricas corrigidas.

Substitui SNR_op (sentinela 9999 quando n_2019_alarms=0) por três métricas
que permitem comparação honesta entre detetores.

Métricas exportadas
-------------------
year_coverage(row)                              → int  0–5
snr_smooth(row, a, n_2019_total, n_post_total) → float ≥ 0
detection_efficiency(row)                       → float 0–1
enrich(df, a, n_2019_total, n_post_total)       → DataFrame + 3 colunas novas
rank_detectors(df)                              → DataFrame ordenado
robustness_table(rows, a_vals, n19t, npot)      → DataFrame (ranking × a)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

YEARS_POST: list[int] = [2021, 2022, 2023, 2024, 2025]
LAPLACE_DEFAULT: float = 1.0


# ── métricas elementares ───────────────────────────────────────────────────────

def year_coverage(row: pd.Series) -> int:
    """Nº de anos em {2021..2025} com ≥1 alarme de deteção (0–5)."""
    return int(sum(1 for y in YEARS_POST if float(row.get(f"n_{y}", 0)) > 0))


def snr_smooth(row: pd.Series, a: float,
               n_2019_total: int, n_post_total: int) -> float:
    """SNR com suavização de Laplace — elimina a sentinela 9999.

    Fórmula:
        SNR_smooth = ((n_post + a) × n_2019_total) / ((n_2019_alarms + a) × n_post_total)

    Justificação de a=1:
        Prior simétrico (pseudo-contagem 1 em cada classe). Testado para
        a ∈ {0.5, 1, 2}: o ranking dos detetores é estável (ver robustness_table).

    Exemplos (Seq2Seq, n_2019_total=48000, n_post_total=240000, a=1):
        ph_standalone:    (13×48000)/(1×240000) = 2.60  [era 9999]
        adwin_standalone: (24×48000)/(3×240000) = 1.60  [era 2.30]
        confirmed (AND):  ( 1×48000)/(1×240000) = 0.20  [era 0.00]
    """
    n_post   = float(row.get("n_alarms_post", row.get("n_post", 0)))
    n_2019_a = float(row.get("n_2019", 0))
    denom = (n_2019_a + a) * n_post_total
    if denom == 0:
        return 0.0
    return ((n_post + a) * n_2019_total) / denom


def detection_efficiency(row: pd.Series) -> float:
    """Fração de alarmes que caem no período pós-2019.

    Penaliza ruído em 2019 sem usar sentinela. Um detetor com todos os
    alarmes pós-2019 tem eficiência=1; metade em 2019 tem ~0.5.
    """
    n_total = float(row.get("n_alarms", row.get("n_alarms_total", 0)))
    n_post  = float(row.get("n_alarms_post", row.get("n_post", 0)))
    if n_total == 0:
        return 0.0
    return n_post / n_total


# ── enriquecimento de DataFrame ────────────────────────────────────────────────

def enrich(df: pd.DataFrame,
           a: float = LAPLACE_DEFAULT,
           n_2019_total: int = 1,
           n_post_total: int = 1) -> pd.DataFrame:
    """Aplica year_coverage, snr_smooth, detection_efficiency a cada linha."""
    out = df.copy()
    out["year_coverage"] = out.apply(year_coverage, axis=1)
    out["SNR_smooth"] = out.apply(
        lambda r: round(snr_smooth(r, a, n_2019_total, n_post_total), 4), axis=1)
    out["detection_efficiency"] = out.apply(
        lambda r: round(detection_efficiency(r), 4), axis=1)
    return out


# ── regra de decisão e ranking ─────────────────────────────────────────────────

def rank_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena pela regra de decisão lexicográfica documentada.

    Regra (3 critérios, aplicados por ordem):
      1. year_coverage DESC  — sensibilidade multi-ano (critério primário)
      2. FAR_2019_per1k ASC  — precisão: entre detetores com igual cobertura,
                               prefere-se menor taxa de falsos alarmes
      3. SNR_smooth DESC     — qualidade do sinal (desempate final)

    Justificação: um detetor que cobre 5 anos com FAR=0.04 é operacionalmente
    superior a um que cobre 1 ano com FAR=0, independente do SNR — o segundo
    é inútil como monitor de longo prazo após 2021.
    """
    return df.sort_values(
        ["year_coverage", "FAR_2019_per1k", "SNR_smooth"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


# ── robustez do ranking em relação a a ────────────────────────────────────────

def robustness_table(rows: list[dict],
                     a_values: list[float],
                     n_2019_total: int,
                     n_post_total: int) -> pd.DataFrame:
    """Verifica estabilidade do ranking para a ∈ {0.5, 1, 2}.

    Demonstra que a escolha de a não altera as conclusões: o detetor primário
    recomendado mantém a posição #1 independentemente de a.

    Returns DataFrame com colunas: a, rank, detector, year_coverage,
    FAR_2019_per1k, SNR_smooth.
    """
    records = []
    for a in a_values:
        tmp = pd.DataFrame(rows)
        tmp = enrich(tmp, a=a, n_2019_total=n_2019_total,
                     n_post_total=n_post_total)
        tmp = rank_detectors(tmp)
        for rank_i, (_, row) in enumerate(tmp.iterrows(), start=1):
            det = row.get("detector", row.get("variant", "?"))
            records.append({
                "a":                a,
                "rank":             rank_i,
                "detector":         det,
                "year_coverage":    int(row["year_coverage"]),
                "FAR_2019_per1k":   float(row.get("FAR_2019_per1k", 0)),
                "SNR_smooth":       float(row["SNR_smooth"]),
            })
    return pd.DataFrame(records)


# ── Wilson score interval para FAR=0 ──────────────────────────────────────────

def wilson_ci_per1k(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para proporção k/n, devolvido em alarmes por 1000 trajs."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5
    denom  = 1 + z**2 / n
    lo = max(0.0, (centre - margin) / denom) * 1000
    hi = ((centre + margin) / denom) * 1000
    return (round(lo, 4), round(hi, 4))
