"""
rerun_adapt_no2021.py — Reexecucao do capitulo de Adaptacao ao Drift (Tabela 3)
==============================================================================
Reproduz TODAS as linhas da Tabela 3 (tab:adapt) do artigo, porem sob uma
nova definicao de regime:

    * regime ANTIGO  = apenas 2019            (proc_sets 3-8)
    * dados NOVOS     = 2022, 2023, 2024, 2025 (proc_sets 15-38)
    * 2021 IGNORADO 100%                       (proc_sets 9-14 removidos)

O protocolo (lr=1e-5, encoder congelado, 2 camadas do decoder, 5 epocas,
max_per=500, batch 256, Adam) e todos os estimadores permanecem identicos aos
scripts originais ewc_finetune.py / replay_finetune.py / targeted_finetune.py.
A unica mudanca e o mapeamento PROC_YEAR, monkeypatchado em cada modulo para
excluir 2021 do lado antigo e mante-lo fora do lado novo.

Saida: Relas/results/mes7_no2021/{ewc,replay,targeted}_results.csv

Uso:
    .venv312/Scripts/python.exe model_analise/rerun_adapt_no2021.py
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bootstrap import init_project

PROJECT_ROOT = init_project()

import model_analise.ewc_finetune as ewc
import model_analise.replay_finetune as replay
import model_analise.targeted_finetune as targeted

# ── Novo mapeamento: 2019 antigo | 2022-2025 novos | 2021 REMOVIDO ─────────
# PROC_YEAR original inclui 2021 nos proc_sets 9-14; removemos essas entradas.
NEW_PROC_YEAR = {n: y for n, y in ewc.PROC_YEAR.items() if y != 2021}

OUT_DIR_NEW = PROJECT_ROOT / "Relas" / "results" / "mes7_no2021"
OUT_DIR_NEW.mkdir(parents=True, exist_ok=True)

# Monkeypatch em cada namespace (replay/targeted importaram PROC_YEAR por valor)
for mod in (ewc, replay, targeted):
    mod.PROC_YEAR = NEW_PROC_YEAR
    mod.OUT_DIR = OUT_DIR_NEW

# BP_YEAR continua 2022: index<index(2022) -> so 2019 sobra (2021 removido);
# index>=index(2022) -> {2022,2023,2024,2025}. Confirmamos a particao:
YO = ewc.YEAR_ORDER
bp = YO.index(ewc.BP_YEAR)
before_years = sorted({y for n, y in NEW_PROC_YEAR.items() if YO.index(y) < bp})
after_years = sorted({y for n, y in NEW_PROC_YEAR.items() if YO.index(y) >= bp})
print(f"[split] regime ANTIGO = {before_years}")
print(f"[split] dados NOVOS   = {after_years}")
print(f"[split] 2021 presente? {'sim' if any(y == 2021 for y in NEW_PROC_YEAR.values()) else 'NAO (ignorado 100%)'}")
assert before_years == [2019]
assert after_years == [2022, 2023, 2024, 2025]
assert all(y != 2021 for y in NEW_PROC_YEAR.values())


def run_main(mod, argv: list[str], tag: str) -> None:
    print(f"\n{'#'*70}\n# {tag}\n{'#'*70}")
    old_argv = sys.argv
    sys.argv = [f"{mod.__name__}"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    # 1) FT conservador (r=0) + Replay uniforme (r=0.25,0.5,1.0) num unico sweep
    run_main(
        replay,
        ["--ratios", "0", "0.25", "0.5", "1.0", "--epochs", "5",
         "--out", str(OUT_DIR_NEW / "replay_results.csv")],
        "REPLAY (r=0 controle + r=0.25/0.5/1.0)",
    )

    # 2) EWC: sweep de lambda ao longo do plato reportado [0; 1e7]
    run_main(
        ewc,
        ["--lambdas", "0", "50", "500", "5000", "100000", "1000000",
         "10000000", "--epochs", "5",
         "--out", str(OUT_DIR_NEW / "ewc_results.csv")],
        "EWC (sweep lambda 0..1e7)",
    )

    # 3) Dirigidas pelo diagnostico: replay_iw / target_accel / target+replay
    run_main(
        targeted,
        ["--out", str(OUT_DIR_NEW / "targeted_results.csv")],
        "TARGETED (replay_iw / seletivo accel / seletivo+replay_iw)",
    )

    print(f"\n[ok] CSVs em {OUT_DIR_NEW}")


if __name__ == "__main__":
    main()
