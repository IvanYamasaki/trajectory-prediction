"""
drift_analise/mes10_independence.py
===================================
Mes 10 — Pendencia 1.2: validacao da hipotese de independencia do stream.

A FAR_2019 por trajetoria (e o IC de Wilson) assume trajetorias ~i.i.d.
dentro de 2019. Trajetorias consecutivas do mesmo jogo podem ser
correlacionadas, reduzindo o tamanho amostral efetivo. Tres verificacoes:

  (1) ACF do sinal ADE (bruto e robz_w200) dentro de 2019, com banda de
      Bartlett; tamanho amostral efetivo n_eff = n / (1 + 2*sum(rho_k)).
  (2) IC de Wilson da FAR recalculado com n_eff (k=2 alarmes observados).
  (3) Teste por permutacao (B=50) sobre o bloco 2019, em tres esquemas:
        blocks : permuta a ordem dos 6 jogos (preserva autocorrelacao
                 intra-jogo) — FAR deve ficar ~igual;
        within : embaralha trajetorias dentro de cada jogo (destroi
                 autocorrelacao intra-jogo, preserva composicao);
        full   : embaralha todo o bloco 2019.
      Se a contagem de alarmes cair muito em `within`/`full`, a
      autocorrelacao intra-jogo inflaciona a FAR observada.

Saidas em Relas/results/drift/mes10_pendencias/independence/:
  acf_2019.csv, acf_2019.png
  permutation_far.csv
  INDEPENDENCE_REPORT.md

Uso:
    python drift_analise/mes10_independence.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

from common.constants import ROBZ_W, ADWIN_S2S_LITE as ADWIN_S2S
from common.metrics import wilson_ci
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes10_pendencias" / "independence"

MAX_LAG  = 500
B_PERM   = int(os.environ.get("B_PERM", "200"))
SEED     = 42


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF amostral ate max_lag (metodo direto, numpy puro)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.dot(xc[:-k], xc[k:])) / denom
    return out


def n_eff_from_acf(n: int, rho: np.ndarray) -> tuple[float, int]:
    """n_eff = n / (1 + 2*sum(rho_k ate o primeiro k nao significativo)).

    Trunca a soma no primeiro lag cuja |rho| cai abaixo da banda de
    Bartlett (aprox. 1.96/sqrt(n)) — evita somar ruido de cauda.
    """
    band = 1.96 / np.sqrt(n)
    s, k_cut = 0.0, 0
    for k in range(1, len(rho)):
        if abs(rho[k]) < band:
            k_cut = k
            break
        s += rho[k]
        k_cut = k
    return n / (1.0 + 2.0 * s), k_cut


def count_2019_alarms(raw_2019: np.ndarray) -> int:
    """Pipeline completo (robz + ADWIN config final) sobre um bloco 2019."""
    sig = smooth_robz(raw_2019, ROBZ_W)
    det = ADWINLite(**ADWIN_S2S)
    return len(run_detector(det, sig))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_errors()
    stream = build_stream(df, model="Seq2Seq")
    s19 = stream[stream["year"] == 2019].reset_index(drop=True)
    raw = s19["ade_traj"].astype(float).values
    games = s19["match_id"].values
    n = len(s19)
    game_ids = list(pd.unique(games))
    print(f"[info] 2019: n={n:,} trajetorias em {len(game_ids)} jogos")

    # ── (1) ACF e n_eff ───────────────────────────────────────────────────────
    robz = smooth_robz(raw, ROBZ_W)
    rho_raw  = acf(raw,  MAX_LAG)
    rho_robz = acf(robz, MAX_LAG)
    neff_raw,  kcut_raw  = n_eff_from_acf(n, rho_raw)
    neff_robz, kcut_robz = n_eff_from_acf(n, rho_robz)

    acf_df = pd.DataFrame({
        "lag": np.arange(MAX_LAG + 1),
        "rho_ade_raw": rho_raw,
        "rho_ade_robz": rho_robz,
    })
    acf_df.to_csv(OUT_DIR / "acf_2019.csv", index=False)

    print(f"[ACF raw ] rho1={rho_raw[1]:.3f} rho10={rho_raw[10]:.3f} "
          f"rho100={rho_raw[100]:.3f} | n_eff={neff_raw:,.0f} "
          f"({100*neff_raw/n:.1f}% de n, corte lag {kcut_raw})")
    print(f"[ACF robz] rho1={rho_robz[1]:.3f} rho10={rho_robz[10]:.3f} "
          f"rho100={rho_robz[100]:.3f} | n_eff={neff_robz:,.0f} "
          f"({100*neff_robz/n:.1f}% de n, corte lag {kcut_robz})")

    # figura ACF
    band = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rho, name in [(axes[0], rho_raw, "ADE bruto"),
                          (axes[1], rho_robz, f"ADE robz_w{ROBZ_W}")]:
        ax.stem(np.arange(1, 201), rho[1:201], markerfmt=" ", basefmt="k-")
        ax.axhspan(-band, band, color="orange", alpha=0.25,
                   label="banda Bartlett 95%")
        ax.set_title(f"ACF 2019 — {name}")
        ax.set_xlabel("lag (trajetorias)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rho(k)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acf_2019.png", dpi=150)
    plt.close(fig)

    # ── (2) Wilson com n_eff ─────────────────────────────────────────────────
    K_OBS = 2   # alarmes 2019 do ADWIN_S2S final (phase4_master_table.csv)
    ci_n    = wilson_ci(K_OBS, n)
    ci_neff = wilson_ci(K_OBS, neff_robz)
    far_n    = 1000.0 * K_OBS / n
    far_neff = 1000.0 * K_OBS / neff_robz
    print(f"[Wilson] n={n}:      FAR={far_n:.4f}/1k  IC=[{ci_n[0]:.4f}; {ci_n[1]:.4f}]")
    print(f"[Wilson] n_eff={neff_robz:,.0f}: FAR={far_neff:.4f}/1k  "
          f"IC=[{ci_neff[0]:.4f}; {ci_neff[1]:.4f}]")

    # ── (3) permutacoes ───────────────────────────────────────────────────────
    k_obs_pipeline = count_2019_alarms(raw)
    print(f"[perm] alarmes observados (ordem real, so bloco 2019): {k_obs_pipeline}")

    schemes = {"blocks": [], "within": [], "full": []}
    game_slices = {g: np.where(games == g)[0] for g in game_ids}

    for b in range(B_PERM):
        # blocks: permuta ordem dos jogos
        order = rng.permutation(game_ids)
        idx_blocks = np.concatenate([game_slices[g] for g in order])
        schemes["blocks"].append(count_2019_alarms(raw[idx_blocks]))

        # within: embaralha dentro de cada jogo, ordem dos jogos preservada
        idx_within = np.concatenate([
            rng.permutation(game_slices[g]) for g in game_ids
        ])
        schemes["within"].append(count_2019_alarms(raw[idx_within]))

        # full: embaralha tudo
        schemes["full"].append(count_2019_alarms(raw[rng.permutation(n)]))

        if (b + 1) % 10 == 0:
            print(f"  perm {b+1}/{B_PERM} ...")

    rows = []
    for name, counts in schemes.items():
        c = np.asarray(counts)
        rows.append(dict(
            scheme=name, B=B_PERM,
            k_obs=k_obs_pipeline,
            k_mean=float(c.mean()), k_median=float(np.median(c)),
            k_p5=float(np.percentile(c, 5)), k_p95=float(np.percentile(c, 95)),
            far_mean_per1k=float(1000.0 * c.mean() / n),
            p_geq_obs=float(np.mean(c >= k_obs_pipeline)),
        ))
    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(OUT_DIR / "permutation_far.csv", index=False)
    print(perm_df.to_string(index=False))

    # ── relatorio ────────────────────────────────────────────────────────────
    L = [
        "# Mes 10 — Independencia do stream 2019 (validacao da FAR)",
        "",
        f"n = {n:,} trajetorias (2019, Seq2Seq 30->15), {len(game_ids)} jogos.",
        "",
        "## (1) Autocorrelacao e tamanho amostral efetivo",
        "",
        "| Sinal | rho(1) | rho(10) | rho(100) | n_eff | n_eff/n | corte |",
        "|-------|-------:|--------:|---------:|------:|--------:|------:|",
        f"| ADE bruto | {rho_raw[1]:.3f} | {rho_raw[10]:.3f} | {rho_raw[100]:.3f} "
        f"| {neff_raw:,.0f} | {100*neff_raw/n:.1f}% | lag {kcut_raw} |",
        f"| ADE robz_w{ROBZ_W} | {rho_robz[1]:.3f} | {rho_robz[10]:.3f} | "
        f"{rho_robz[100]:.3f} | {neff_robz:,.0f} | {100*neff_robz/n:.1f}% "
        f"| lag {kcut_robz} |",
        "",
        "## (2) IC de Wilson da FAR (k=2 alarmes) com n vs n_eff",
        "",
        "| Base | FAR/1k | IC Wilson 95%/1k |",
        "|------|-------:|------------------|",
        f"| n = {n:,} | {far_n:.4f} | [{ci_n[0]:.4f}; {ci_n[1]:.4f}] |",
        f"| n_eff = {neff_robz:,.0f} (robz) | {far_neff:.4f} "
        f"| [{ci_neff[0]:.4f}; {ci_neff[1]:.4f}] |",
        "",
        "## (3) Permutacoes do bloco 2019 (B=50, pipeline robz+ADWIN final)",
        "",
        f"Alarmes observados na ordem real: **{k_obs_pipeline}**",
        "",
        "| Esquema | k medio | k mediano | [p5; p95] | FAR media/1k | P(k >= obs) |",
        "|---------|--------:|----------:|-----------|-------------:|------------:|",
    ]
    for _, r in perm_df.iterrows():
        L.append(f"| {r['scheme']} | {r['k_mean']:.2f} | {r['k_median']:.1f} | "
                 f"[{r['k_p5']:.0f}; {r['k_p95']:.0f}] | "
                 f"{r['far_mean_per1k']:.4f} | {r['p_geq_obs']:.2f} |")
    L += [
        "",
        "Leitura: `blocks` preserva a autocorrelacao intra-jogo (mede o efeito",
        "da ordem dos jogos); `within` destroi a autocorrelacao intra-jogo;",
        "`full` destroi toda a estrutura. Se k(within/full) >> k(obs), a",
        "autocorrelacao intra-jogo NAO inflaciona a FAR observada (ao",
        "contrario: a estrutura local reduz falsos alarmes do robz+ADWIN).",
    ]
    (OUT_DIR / "INDEPENDENCE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'INDEPENDENCE_REPORT.md'}")


if __name__ == "__main__":
    main()
