"""
drift_analise/mes11_final_master_table.py
=========================================
Mes 11 — Partes A2+A4 do plano (PLANO_SECAO_DETECCAO.md).

Constroi a master table FINAL (<=6 linhas) para a nova secao de deteccao:
  - detector primario decidido pelos numeros OUT-OF-SAMPLE do holdout
    (mes11_holdout/holdout_test_results.csv), pela mesma regra lexicografica
    (year_coverage DESC, FAR ASC, SNR_smooth DESC), entre as variantes
    lite/exact do vencedor Seq2Seq;
  - canal OR (S2S + Kalman) out-of-sample;
  - AND de mesma escala (gate unificado, IN-SAMPLE — rotulado como tal);
  - linha-resumo dos resultados negativos (KSWIN, PH_agg, AND entre escalas).

Saidas em Relas/results/drift/mes11_final/:
  final_master_table.csv
  MES11_FINAL_REPORT.md

Uso:
    python drift_analise/mes11_final_master_table.py
"""
from __future__ import annotations

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

DRIFT = PROJECT_ROOT / "Relas" / "results" / "drift"
HOLDOUT = DRIFT / "mes11_holdout" / "holdout_test_results.csv"
LATENCY = DRIFT / "mes11_holdout" / "holdout_latency_test.csv"
SAME_SCALE = DRIFT / "mes11_consistency" / "same_scale_gate_unified.csv"
OUT_DIR = DRIFT / "mes11_final"

FAR_ADMISSIBILITY = 0.20


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(HOLDOUT)
    lat = pd.read_csv(LATENCY)

    rows = []

    # ── candidatos a primario: vencedores S2S robz (lite e exact), OOS ───────
    cand = res[(res["model"] == "Seq2Seq") & (res["role"] == "winner_calib") &
               (res["signal"] == "robz")].copy()
    cand = cand.sort_values(
        ["year_coverage", "FAR_2019_per1k", "SNR_smooth"],
        ascending=[False, True, False])
    primary = cand.iloc[0]
    runner = cand.iloc[1] if len(cand) > 1 else None
    print("[decisao primario] candidatos OOS (S2S, robz):")
    print(cand[["variant", "config", "FAR_2019_per1k", "year_coverage",
                "SNR_smooth"]].to_string(index=False))

    def _lat_str(variant: str, model: str, channel: str) -> str:
        sub = lat[(lat["variant"] == variant) & (lat["model"] == model) &
                  (lat["channel"] == channel)]
        if sub.empty:
            return ""
        vals = sub["delay_games"].dropna()
        return f"{vals.min():.2f}-{vals.max():.2f}" if len(vals) else "sem alarme"

    def _mk_row(r: pd.Series, detector: str, papel: str, escopo: str,
                latencia: str = "") -> dict:
        far = float(r["FAR_2019_per1k"])
        adm = far <= FAR_ADMISSIBILITY
        return dict(
            detector=detector, config=r.get("config", ""),
            escopo=escopo,
            FAR_2019_per1k=far,
            IC_Wilson_per1k=f"[{r['wilson_lo_per1k']}; {r['wilson_hi_per1k']}]"
                            if "wilson_lo_per1k" in r and pd.notna(r.get("wilson_lo_per1k"))
                            else "",
            year_coverage=int(r["year_coverage"]),
            SNR_smooth=float(r["SNR_smooth"]),
            n_post=int(r["n_post_alarms"]) if "n_post_alarms" in r else int(r.get("n_post", 0)),
            latencia_jogos=latencia,
            admissivel=adm,
            papel_final=papel,
        )

    rows.append(_mk_row(
        primary, f"ADWIN_{primary['variant']} (S2S)",
        "Detetor primario recomendado", "teste (out-of-sample)",
        _lat_str(primary["variant"], "Seq2Seq", "winner")))

    if runner is not None:
        rows.append(_mk_row(
            runner, f"ADWIN_{runner['variant']} (S2S)",
            "Robustez de implementacao", "teste (out-of-sample)",
            _lat_str(runner["variant"], "Seq2Seq", "winner")))

    # ── canal OR da variante primaria, OOS ───────────────────────────────────
    orch = res[(res["role"] == "or_channel") &
               (res["variant"] == primary["variant"])]
    if not orch.empty:
        r = orch.iloc[0]
        papel = ("Vigilancia multi-ano / reducao de latencia"
                 if float(r["FAR_2019_per1k"]) <= FAR_ADMISSIBILITY
                 else "OR inadmissivel no teste (FAR > 0,20/1k)")
        rows.append(_mk_row(r, f"OR(S2S, Kalman) {r['variant']}", papel,
                            "teste (out-of-sample)",
                            _lat_str(r["variant"], "OR(S2S,Kalman)", "or")))

    # ── AND mesma escala (gate unificado; in-sample, rotulado) ───────────────
    ss = pd.read_csv(SAME_SCALE)
    ss200 = ss[(ss["gate"] == "phase3_deque") & (ss["W"] == 200)]
    if not ss200.empty:
        r = ss200.iloc[0]
        rows.append(dict(
            detector="AND(ADE, accel_p95) W=200",
            config="dois ADWIN por trajetoria, robz_w200",
            escopo="stream completo (in-sample)",
            FAR_2019_per1k=float(r["FAR_2019_per1k"]),
            IC_Wilson_per1k="",
            year_coverage=int(r["year_coverage"]),
            SNR_smooth=float(r["SNR_smooth"]),
            n_post=int(r["n_post"]),
            latencia_jogos="",
            admissivel=float(r["FAR_2019_per1k"]) <= FAR_ADMISSIBILITY,
            papel_final="Alta especificidade (custo: cobertura)",
        ))

    # ── linha-resumo dos negativos ───────────────────────────────────────────
    rows.append(dict(
        detector="Negativos: KSWIN; PH_agg; AND(ADWIN, PH_agg)",
        config="ver apendice",
        escopo="stream completo (in-sample)",
        FAR_2019_per1k=np.nan, IC_Wilson_per1k="",
        year_coverage=0, SNR_smooth=np.nan, n_post=0, latencia_jogos="",
        admissivel=False,
        papel_final=("KSWIN inadmissivel (0,438/1k); PH_agg so 2021; "
                     "AND entre escalas vazio (Delta_min=214)"),
    ))

    master = pd.DataFrame(rows)
    out_csv = OUT_DIR / "final_master_table.csv"
    master.to_csv(out_csv, index=False)
    print(f"[ok] {out_csv}")
    print(master[["detector", "escopo", "FAR_2019_per1k", "year_coverage",
                  "papel_final"]].to_string(index=False))

    L = [
        "# Mes 11 — Master table final (A2+A4)",
        "",
        "Decisao do primario pelos numeros OUT-OF-SAMPLE do holdout"
        " (regra: cobertura DESC, FAR ASC, SNR_smooth DESC).",
        "",
        master.to_markdown(index=False),
        "",
        "## Candidatos a primario considerados (teste, S2S, robz)",
        "",
        cand[["variant", "config", "FAR_2019_per1k", "wilson_lo_per1k",
              "wilson_hi_per1k", "year_coverage", "SNR_smooth"]]
            .to_markdown(index=False),
    ]
    (OUT_DIR / "MES11_FINAL_REPORT.md").write_text("\n".join(L),
                                                   encoding="utf-8")
    print(f"[ok] {OUT_DIR / 'MES11_FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
