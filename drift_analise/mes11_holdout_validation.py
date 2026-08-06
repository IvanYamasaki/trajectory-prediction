"""
drift_analise/mes11_holdout_validation.py
=========================================
Mes 11 — Parte A1 do plano (PLANO_SECAO_DETECCAO.md): validacao com holdout.

Fecha a fragilidade central do Mes 9: as 174 configs foram selecionadas
otimizando FAR_2019/year_coverage no MESMO stream em que essas metricas sao
reportadas. Protocolo novo:

  SPLIT POR JOGOS. Para cada ano, os jogos (match_id ordenado) de indice PAR
  formam o stream de CALIBRACAO; os de indice IMPAR formam o stream de TESTE.
  2019/2021/2023/2024 tem 6 jogos (3+3); 2022/2025 tem 5 (3 calib + 2 teste).

  CALIBRACAO. Grid da Frente A (96 configs: delta x min_window x cooldown x
  max_window) sobre o sinal robz_w200 do stream de calibracao. Selecao pela
  regra da Fase 4: admissivel (FAR<=0.20/1k) -> year_coverage DESC,
  FAR ASC, SNR_smooth DESC.

  TESTE. A config vencedora e avaliada UMA vez no stream de teste:
  FAR_2019 (IC Wilson 95%), cobertura, alarmes por ano e latencia por ano.
  A config "do paper" (selecionada in-sample no Mes 9) tambem e avaliada no
  teste, como referencia.

Roda para ADWINLite e ADWINExact (river), Seq2Seq e Kalman, e adicionalmente
um braco de controlo sem robz (exact apenas) para checar o valor do
pre-processamento fora da amostra.

Uso (grids paralelizaveis):
    python drift_analise/mes11_holdout_validation.py --grid --variant exact
    python drift_analise/mes11_holdout_validation.py --grid --variant lite --model Seq2Seq
    python drift_analise/mes11_holdout_validation.py --grid --variant lite --model Kalman
    python drift_analise/mes11_holdout_validation.py --consolidate

Saidas em Relas/results/drift/mes11_holdout/:
  holdout_split.csv                     — jogos por split
  partial/grid_{variant}_{model}.csv    — grid completo no stream de calibracao
  holdout_test_results.csv              — vencedores + configs do paper no teste
  holdout_latency_test.csv              — latencia por ano no teste
  MES11_HOLDOUT_REPORT.md
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.bootstrap import init_project
PROJECT_ROOT = init_project()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.constants import ROBZ_W, YEARS_POST, ADWIN_S2S_LITE, ADWIN_KALMAN_LITE
from drift_analise.chapter02_deteccao_pipeline import (
    ADWINLite, ADWINExact, run_detector, load_errors, build_stream,
)
from drift_analise.phase2_grid_search import smooth_robz, GRID_A
from drift_analise.phase4_metrics import wilson_ci_per1k


class ADWINLiteFast:
    """Reimplementacao do ADWINLite com buffer numpy contiguo.

    Numericamente IDENTICA ao ADWINLite (mesma sequencia de operacoes
    float64: mean/var ddof=1 sobre as mesmas metades, mesmo z_crit,
    mesma ordem append->trim->cooldown->teste), apenas sem o custo de
    list.pop(0) e das conversoes list->ndarray por passo. Verificada por
    `--verify` (alarmes identicos aos do ADWINLite em varias configs).
    """

    def __init__(self, delta=1e-5, max_window=2000,
                 min_window=200, cooldown=200):
        from math import sqrt, log
        self.delta = delta
        self.max_w = int(max_window)
        self.min_w = int(min_window)
        self.cooldown = int(cooldown)
        self.z_crit = max(2.0, sqrt(2 * log(2.0 / delta)))
        self._buf = np.empty(self.max_w, dtype=float)
        self._n = 0
        self._cool = 0
        self.drift_detected = False

    def update(self, x: float) -> None:
        if self._n == self.max_w:                    # pop(0)
            self._buf[:-1] = self._buf[1:]
            self._n -= 1
        self._buf[self._n] = float(x)                # append
        self._n += 1
        self.drift_detected = False
        if self._cool > 0:
            self._cool -= 1
            return
        if self._n < self.min_w:
            return
        mid = self._n // 2
        a = self._buf[:mid]
        b = self._buf[mid:self._n]
        mu_a = a.mean()
        mu_b = b.mean()
        sd = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if sd < 1e-9:
            return
        if abs(mu_a - mu_b) / sd > self.z_crit:
            self.drift_detected = True
            nb = self._n - mid                       # buffer = list(b)
            self._buf[:nb] = b
            self._n = nb
            self._cool = self.cooldown

OUT_DIR = PROJECT_ROOT / "Relas" / "results" / "drift" / "mes11_holdout"
PARTIAL = OUT_DIR / "partial"

YEARS_ALL = [2019] + YEARS_POST
FAR_ADMISSIBILITY = 0.20

VARIANT_CLASS = {"lite": ADWINLiteFast, "exact": ADWINExact}


def verify_fast_equivalence() -> None:
    """Compara ADWINLiteFast vs ADWINLite: alarmes devem ser identicos."""
    rng = np.random.default_rng(0)
    sig = np.concatenate([rng.normal(0, 1, 30000),
                          rng.normal(0.8, 1.3, 30000)])
    cfgs = [dict(ADWIN_S2S_LITE),
            dict(delta=1e-4, min_window=200, cooldown=1000, max_window=5000),
            dict(delta=1e-5, min_window=1000, cooldown=500, max_window=2000)]
    for cfg in cfgs:
        a1 = run_detector(ADWINLite(**cfg), sig)
        a2 = run_detector(ADWINLiteFast(**cfg), sig)
        assert a1 == a2, f"divergencia em {cfg}: {a1[:5]}... vs {a2[:5]}..."
        print(f"  [verify ok] {cfg} -> {len(a1)} alarmes identicos")
    print("[verify] ADWINLiteFast == ADWINLite em todas as configs testadas")

# Configs selecionadas in-sample no Mes 9 (referencia no teste)
PAPER_CFG = {
    ("lite", "Seq2Seq"): dict(ADWIN_S2S_LITE),
    ("lite", "Kalman"):  dict(ADWIN_KALMAN_LITE),
    ("exact", "Seq2Seq"): dict(delta=1e-7, min_window=200, cooldown=1000, max_window=2000),
    ("exact", "Kalman"):  dict(delta=1e-7, min_window=1000, cooldown=200, max_window=2000),
}


# ── split ─────────────────────────────────────────────────────────────────────

def split_by_games(stream: pd.DataFrame, fold: str = "A"
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold A: jogos de indice par (match_id ordenado, por ano) -> calib,
    impar -> teste. Fold B: invertido. Mantem a ordem temporal (o stream ja
    vem ordenado por year, match_id, traj_id); reindexa global_idx por split.
    """
    calib_off = 0 if fold == "A" else 1
    calib_mask = np.zeros(len(stream), dtype=bool)
    split_rows = []
    for y in sorted(stream["year"].dropna().unique()):
        games = sorted(stream.loc[stream["year"] == y, "match_id"].unique())
        calib_games = set(games[calib_off::2])
        calib_mask |= (stream["year"] == y).values & \
                      stream["match_id"].isin(calib_games).values
        for g in games:
            split_rows.append(dict(year=int(y), match_id=g, fold=fold,
                                   split="calib" if g in calib_games else "test"))

    def _rebuild(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.reset_index(drop=True).copy()
        sub["global_idx"] = np.arange(len(sub))
        return sub

    return (_rebuild(stream[calib_mask]),
            _rebuild(stream[~calib_mask]),
            pd.DataFrame(split_rows))


# ── metricas ──────────────────────────────────────────────────────────────────

def eval_alarms(alarms: list[int], year_vals: np.ndarray) -> dict:
    n19_total = int((year_vals == 2019).sum())
    npost_total = int((year_vals >= 2021).sum())
    a = np.asarray(sorted(alarms), dtype=int)
    by = {y: int((year_vals[a] == y).sum()) if len(a) else 0 for y in YEARS_ALL}
    n19 = by[2019]
    n_post = sum(by[y] for y in YEARS_POST)
    cov = sum(1 for y in YEARS_POST if by[y] > 0)
    snr_sm = ((n_post + 1.0) * n19_total) / ((n19 + 1.0) * max(1, npost_total))
    return dict(
        n_alarms=len(a), n_2019_alarms=n19, n_post_alarms=n_post,
        FAR_2019_per1k=round(1000.0 * n19 / max(1, n19_total), 4),
        year_coverage=cov, SNR_smooth=round(snr_sm, 3),
        **{f"n_{y}": by[y] for y in YEARS_ALL},
    )


def latency_by_year(alarms: list[int], stream: pd.DataFrame) -> list[dict]:
    year_vals = stream["year"].fillna(0).astype(int).values
    a = np.asarray(sorted(alarms), dtype=int)
    rows = []
    for y in YEARS_POST:
        mask = year_vals == y
        if not mask.any():
            continue
        boundary = int(np.argmax(mask))
        n_year = int(mask.sum())
        n_games = int(stream.loc[mask, "match_id"].nunique())
        trajs_per_game = n_year / max(1, n_games)
        in_year = a[(a >= boundary) & (a < boundary + n_year)]
        if len(in_year):
            delay = int(in_year[0]) - boundary
            rows.append(dict(year=y, delay_trajs=delay,
                             delay_games=round(delay / trajs_per_game, 2),
                             delay_pct_year=round(100.0 * delay / n_year, 1),
                             n_alarms_year=len(in_year)))
        else:
            rows.append(dict(year=y, delay_trajs=-1, delay_games=np.nan,
                             delay_pct_year=np.nan, n_alarms_year=0))
    return rows


# ── grid (calibracao) ─────────────────────────────────────────────────────────

def run_grid(model: str, variant: str, signal: str = "robz",
             deltas: list[float] | None = None, tag: str = "",
             fold: str = "A") -> None:
    PARTIAL.mkdir(parents=True, exist_ok=True)
    df = load_errors()
    stream = build_stream(df, model=model)
    calib, _, split_df = split_by_games(stream, fold)
    split_path = OUT_DIR / f"holdout_split_fold{fold}.csv"
    if not split_path.exists():
        split_df.to_csv(split_path, index=False)

    raw = calib["ade_traj"].astype(float).values
    sig = smooth_robz(raw, ROBZ_W) if signal == "robz" else raw
    year_vals = calib["year"].fillna(0).astype(int).values
    cls = VARIANT_CLASS[variant]

    grid_deltas = deltas if deltas else GRID_A["delta"]
    combos = list(itertools.product(grid_deltas, GRID_A["min_window"],
                                    GRID_A["cooldown"], GRID_A["max_window"]))
    print(f"[grid] fold{fold}/{variant}/{model}/{signal}{tag}: {len(combos)} configs "
          f"sobre {len(calib):,} trajetorias de calibracao", flush=True)
    rows = []
    for i, (delta, mw, cd, xw) in enumerate(combos, 1):
        det = cls(delta=delta, min_window=mw, cooldown=cd, max_window=xw)
        alarms = run_detector(det, sig)
        m = eval_alarms(alarms, year_vals)
        rows.append(dict(fold=fold, variant=variant, model=model, signal=signal,
                         delta=delta, min_window=mw, cooldown=cd,
                         max_window=xw, **m))
        if i % 12 == 0:
            print(f"  {i}/{len(combos)} ...", flush=True)

    fold_tag = "" if fold == "A" else f"_fold{fold}"
    out = PARTIAL / f"grid_{variant}_{model}_{signal}{tag}{fold_tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] {out}", flush=True)


# ── selecao + teste ───────────────────────────────────────────────────────────

def select_winner(grid: pd.DataFrame) -> pd.Series:
    adm = grid[grid["FAR_2019_per1k"] <= FAR_ADMISSIBILITY]
    pool = adm if not adm.empty else grid
    return pool.sort_values(
        ["year_coverage", "FAR_2019_per1k", "SNR_smooth"],
        ascending=[False, True, False]).iloc[0]


def run_config_on(stream: pd.DataFrame, variant: str, cfg: dict,
                  signal: str = "robz") -> list[int]:
    raw = stream["ade_traj"].astype(float).values
    sig = smooth_robz(raw, ROBZ_W) if signal == "robz" else raw
    det = VARIANT_CLASS[variant](**cfg)
    return run_detector(det, sig)


def consolidate(fold: str = "A") -> None:
    df = load_errors()
    streams = {}
    for model in ("Seq2Seq", "Kalman"):
        full = build_stream(df, model=model)
        calib, test, _ = split_by_games(full, fold)
        streams[model] = dict(calib=calib, test=test)

    test_rows, latency_rows = [], []
    winner_alarms: dict[tuple, list[int]] = {}

    partials = sorted(PARTIAL.glob("grid_*.csv"))
    if not partials:
        raise RuntimeError("Nenhum grid parcial em partial/. Rode --grid antes.")
    grids = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
    if "fold" not in grids.columns:
        grids["fold"] = "A"
    grids["fold"] = grids["fold"].fillna("A")
    grids = grids[grids["fold"] == fold]
    if grids.empty:
        raise RuntimeError(f"Nenhum grid parcial do fold {fold}.")

    for (variant, model, signal), grid in grids.groupby(
            ["variant", "model", "signal"]):
        test = streams[model]["test"]
        year_test = test["year"].fillna(0).astype(int).values
        n19_test = int((year_test == 2019).sum())

        win = select_winner(grid)
        win_cfg = dict(delta=float(win["delta"]),
                       min_window=int(win["min_window"]),
                       cooldown=int(win["cooldown"]),
                       max_window=int(win["max_window"]))
        cfg_str = (f"delta={win_cfg['delta']:.0e}, mw={win_cfg['min_window']}, "
                   f"cd={win_cfg['cooldown']}, xw={win_cfg['max_window']}")
        print(f"[{variant}/{model}/{signal}] vencedora calib: {cfg_str} "
              f"(FAR_calib={win['FAR_2019_per1k']}, cov_calib={win['year_coverage']})")

        alarms = run_config_on(test, variant, win_cfg, signal)
        m = eval_alarms(alarms, year_test)
        lo, hi = wilson_ci_per1k(m["n_2019_alarms"], n19_test)
        test_rows.append(dict(
            variant=variant, model=model, signal=signal, role="winner_calib",
            config=cfg_str,
            FAR_calib=float(win["FAR_2019_per1k"]),
            cov_calib=int(win["year_coverage"]),
            wilson_lo_per1k=lo, wilson_hi_per1k=hi, **m))
        if signal == "robz":
            winner_alarms[(variant, model)] = alarms
            for r in latency_by_year(alarms, test):
                latency_rows.append(dict(variant=variant, model=model,
                                         channel="winner", **r))

        # config do paper (selecionada in-sample no Mes 9) no teste
        if signal == "robz" and (variant, model) in PAPER_CFG:
            pcfg = PAPER_CFG[(variant, model)]
            pstr = (f"delta={pcfg['delta']:.0e}, mw={pcfg['min_window']}, "
                    f"cd={pcfg['cooldown']}, xw={pcfg['max_window']}")
            palarms = run_config_on(test, variant, pcfg, signal)
            pm = eval_alarms(palarms, year_test)
            plo, phi = wilson_ci_per1k(pm["n_2019_alarms"], n19_test)
            test_rows.append(dict(
                variant=variant, model=model, signal=signal, role="paper_cfg",
                config=pstr, FAR_calib=np.nan, cov_calib=np.nan,
                wilson_lo_per1k=plo, wilson_hi_per1k=phi, **pm))

    # canal OR por variante (uniao dos vencedores S2S + Kalman; mesmos indices)
    for variant in sorted({v for v, _ in winner_alarms}):
        if (variant, "Seq2Seq") not in winner_alarms or \
           (variant, "Kalman") not in winner_alarms:
            continue
        test = streams["Seq2Seq"]["test"]
        year_test = test["year"].fillna(0).astype(int).values
        n19_test = int((year_test == 2019).sum())
        or_alarms = sorted(set(winner_alarms[(variant, "Seq2Seq")]) |
                           set(winner_alarms[(variant, "Kalman")]))
        m = eval_alarms(or_alarms, year_test)
        lo, hi = wilson_ci_per1k(m["n_2019_alarms"], n19_test)
        test_rows.append(dict(
            variant=variant, model="OR(S2S,Kalman)", signal="robz",
            role="or_channel", config="uniao dos vencedores",
            FAR_calib=np.nan, cov_calib=np.nan,
            wilson_lo_per1k=lo, wilson_hi_per1k=hi, **m))
        for r in latency_by_year(or_alarms, test):
            latency_rows.append(dict(variant=variant, model="OR(S2S,Kalman)",
                                     channel="or", **r))

    fold_tag = "" if fold == "A" else f"_fold{fold}"
    res = pd.DataFrame(test_rows)
    res.insert(0, "fold", fold)
    res.to_csv(OUT_DIR / f"holdout_test_results{fold_tag}.csv", index=False)
    lat = pd.DataFrame(latency_rows)
    lat.insert(0, "fold", fold)
    lat.to_csv(OUT_DIR / f"holdout_latency_test{fold_tag}.csv", index=False)
    print(f"[ok] {OUT_DIR / f'holdout_test_results{fold_tag}.csv'}")
    print(f"[ok] {OUT_DIR / f'holdout_latency_test{fold_tag}.csv'}")

    # ── relatorio ─────────────────────────────────────────────────────────────
    L = [
        f"# Mes 11 — A1: validacao com holdout (split por jogos, fold {fold})",
        "",
        "Fold A: calibracao = jogos pares, teste = impares; fold B: invertido.",
        "A config vencedora do grid de calibracao e avaliada UMA vez no teste.",
        "",
        "## Resultados no stream de TESTE (out-of-sample)",
        "",
        res.drop(columns=[c for c in res.columns
                          if c.startswith("n_2") and c not in
                          ("n_2019_alarms",)]).to_markdown(index=False),
        "",
        "## Alarmes por ano (teste)",
        "",
        res[["variant", "model", "role"] +
            [f"n_{y}" for y in YEARS_ALL]].to_markdown(index=False),
        "",
        "## Latencia por ano (teste)",
        "",
        lat.to_markdown(index=False),
        "",
        "Nota: o pre-processamento robz_w200 foi FIXADO a partir da Fase 1",
        "(selecao in-sample no stream completo); o braco `signal=raw` (exact)",
        "quantifica quanto o robz vale fora da amostra.",
    ]
    md_path = OUT_DIR / f"MES11_HOLDOUT_REPORT{fold_tag}.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {md_path}")


# ── cli ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="checa ADWINLiteFast == ADWINLite")
    ap.add_argument("--variant", choices=["lite", "exact"], default=None)
    ap.add_argument("--model", choices=["Seq2Seq", "Kalman"], default=None)
    ap.add_argument("--signal", choices=["robz", "raw"], default="robz")
    ap.add_argument("--chunk", type=int, choices=[1, 2], default=None,
                    help="1: deltas {1e-4,1e-5}; 2: deltas {1e-6,1e-7}")
    ap.add_argument("--fold", choices=["A", "B"], default="A")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_fast_equivalence()
    if args.grid:
        deltas, tag = None, ""
        if args.chunk == 1:
            deltas, tag = [1e-4, 1e-5], "_c1"
        elif args.chunk == 2:
            deltas, tag = [1e-6, 1e-7], "_c2"
        variants = [args.variant] if args.variant else ["exact", "lite"]
        models = [args.model] if args.model else ["Seq2Seq", "Kalman"]
        for v in variants:
            for m in models:
                run_grid(m, v, args.signal, deltas, tag, args.fold)
    if args.consolidate:
        consolidate(args.fold)
    if not args.grid and not args.consolidate and not args.verify:
        ap.print_help()


if __name__ == "__main__":
    main()
