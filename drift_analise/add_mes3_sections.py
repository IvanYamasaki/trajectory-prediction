"""
add_mes3_sections.py
====================
Adiciona as Seções 13-16 (Mês 3) ao notebook drift_analise.ipynb.
Seguro para rodar múltiplas vezes: verifica se as seções já existem.

Uso:
    python drift_analise/add_mes3_sections.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

NOTEBOOK = Path("drift_analise/drift_analise.ipynb")
MARKER = "# Mês 3 — Adaptação ao Data Drift"


def cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def cell_code(source: str) -> dict:
    lines = [ln + "\n" for ln in source.rstrip().split("\n")]
    return {
        "cell_type": "code", "metadata": {},
        "source": lines, "outputs": [], "execution_count": None,
    }


NEW_CELLS = [
    cell_md(
        "# Mês 3 — Adaptação ao Data Drift\n\n---\n\n"
        "# 13. Importance Weighting (LSIF) — Frente 2\n\n"
        "Estimamos pesos $w_i = p_{2019}(x_i)/p_y(x_i)$ (LSIF) para cada trajetória de ano $y$.\n"
        "O ADE-IW mede quanto do excesso de ADE é explicado por covariate shift.\n\n"
        "**recovery_pct** = (ADE_y − ADE_IW) / (ADE_y − ADE_2019) × 100  \n"
        "Se recovery ≈ 100%: todo o excesso é covariate shift.  \n"
        "Se recovery ≈ 0%: é concept drift puro.\n"
        "\n**ESS_ratio** = (Σw)² / (n · Σw²) — deve ser ≥ 0.3 para pesos confiáveis.\n"
    ),
    cell_code(
        "# --- Seção 13: importance weighting ---\n"
        "iw_path = RES_MES3 / 'iw_decomposition.csv'\n"
        "if iw_path.exists():\n"
        "    df_iw = pd.read_csv(iw_path)\n"
        "    display(df_iw.round(3))\n"
        "    print()\n"
        "    for _, row in df_iw.iterrows():\n"
        "        ess_ok = row['ess_stable']\n"
        "        rec = row['recovery_pct']\n"
        "        print(f\"  {int(row['year'])}: ESS={row['ess_ratio']:.2f} \"\n"
        "              f\"{'ok' if ess_ok else 'INSTAVEL'} | \"\n"
        "              f\"ADE_y={row['ade_y']:.2f} ADE_IW={row['ade_iw']:.2f} \"\n"
        "              f\"recovery={rec:.1f}%\")\n"
        "else:\n"
        "    print('[warn] Execute: python drift_analise/compute_importance_weights.py')"
    ),
    cell_code(
        "# --- Seção 13: plots IW ---\n"
        "for fname in ['iw_ade_comparison.pdf', 'iw_recovery.pdf', 'iw_weights_dist.pdf']:\n"
        "    p = RES_MES3 / fname\n"
        "    if p.exists():\n"
        "        print(f'  OK: {p}')\n"
        "    else:\n"
        "        print(f'  MISSING: {p}')\n"
        "        print('  Execute: MPLBACKEND=Agg python drift_analise/plot_iw_results.py')"
    ),
    cell_md(
        "# 14. Retreino Seletivo nos Breakpoints — Frente 3\n\n"
        "Fine-tuning leve da última Dense do Seq2Seq (encoder congelado),\n"
        "usando dados antes do breakpoint Pelt para treino e dados depois para avaliação.\n\n"
        "**Breakpoint:** transição 2019+2021 → 2022-2025 (detectado por Pelt RBF, pen=1).\n\n"
        "**Critério de aceite:** recovery > 20% e n_degraded/n_improved < 0.3."
    ),
    cell_code(
        "# --- Seção 14: resultados do retreino seletivo ---\n"
        "retrain_path = RES_MES3 / 'retrain_results.csv'\n"
        "if retrain_path.exists():\n"
        "    df_ret = pd.read_csv(retrain_path)\n"
        "    display(df_ret.round(3))\n"
        "    for _, row in df_ret.iterrows():\n"
        "        ok_rec = row['recovery_pct'] > 20 if pd.notna(row['recovery_pct']) else False\n"
        "        ok_cat = (row['n_traj_degraded'] / max(row['n_traj_improved'], 1)) < 0.3\n"
        "        status = 'ACEITE' if (ok_rec and ok_cat) else 'nao atingido'\n"
        "        print(f\"  {row['breakpoint_label']}: {status} \"\n"
        "              f\"recovery={row['recovery_pct']:.1f}% \"\n"
        "              f\"cat_ratio={row['catastrophic_ratio']:.2f}\")\n"
        "else:\n"
        "    print('[warn] Execute: python model_analise/retrain_at_breakpoints.py --penalty 1')"
    ),
    cell_md(
        "# 15. Validação por Divisão — Frente 4\n\n"
        "Verifica se o drift temporal sobrevive ao corte por divisão SSL (A/B).\n\n"
        "**Nota:** apenas Division A está mapeada nos dados disponíveis;\n"
        "jogos de 2021 aparecem como 'Unknown' porque a fonte oficial marca participação 'Both'."
    ),
    cell_code(
        "# --- Seção 15: validação por divisão ---\n"
        "dec_div_path = RES_MES3 / 'by_division_decomposition.csv'\n"
        "tab_2x2_path = RES_MES3 / 'by_division_2x2_table.csv'\n"
        "if dec_div_path.exists():\n"
        "    print('=== Decomposicao por divisao ===')\n"
        "    display(pd.read_csv(dec_div_path).round(3))\n"
        "if tab_2x2_path.exists():\n"
        "    print()\n"
        "    print('=== Tabela 2x2 (antes/depois 2022) x divisao ===')\n"
        "    df2x2 = pd.read_csv(tab_2x2_path)\n"
        "    display(df2x2.round(3))\n"
        "    for _, row in df2x2.iterrows():\n"
        "        delta = row.get('delta', float('nan'))\n"
        "        if abs(delta) > 0.5:\n"
        "            verdict = 'SIM — drift sobrevive ao corte'\n"
        "        elif abs(delta) < 0.5:\n"
        "            verdict = 'NAO — drift desaparece ao fixar divisao'\n"
        "        else:\n"
        "            verdict = 'PARCIAL/INCERTO'\n"
        "        print(f\"  Division {row['division']}: delta={delta:+.3f} mm -> {verdict}\")\n"
        "if not dec_div_path.exists():\n"
        "    print('[warn] Execute: python drift_analise/division_validation.py')"
    ),
    cell_md(
        "# 16. Sensibilidade Amostral (n=3 vs n=6) — Frente 1\n\n"
        "Compara ADE médio por ano (com CI95 bootstrap) ao duplicar o número de jogos amostrados.\n"
        "Se delta < 1 mm: amostra n=3 é suficiente; resultados do Mês 2 são robustos."
    ),
    cell_code(
        "# --- Seção 16: sensibilidade amostral ---\n"
        "sens_path = RES_MES3 / 'sample_size_sensitivity.csv'\n"
        "if sens_path.exists():\n"
        "    df_sens = pd.read_csv(sens_path)\n"
        "    pivot = df_sens.pivot_table(\n"
        "        index=['year', 'model', 'horizon'],\n"
        "        columns='n_per_year', values='ade_mean'\n"
        "    ).round(2)\n"
        "    display(pivot)\n"
        "    if 3 in pivot.columns and 6 in pivot.columns:\n"
        "        diff = (pivot[6] - pivot[3]).abs()\n"
        "        flag = diff[diff > 1.0]\n"
        "        if flag.empty:\n"
        "            print('[OK] ADE estavel entre n=3 e n=6 (delta < 1 mm)')\n"
        "        else:\n"
        "            print('[AVISO] instabilidade amostral em:')\n"
        "            print(flag)\n"
        "else:\n"
        "    print('[info] Execute apos n=6 run:')\n"
        "    print('  python model_analise/sample_size_sensitivity.py')"
    ),
    cell_md(
        "## Conclusão — Mês 3\n\n"
        "Respostas às três perguntas centrais:\n\n"
        "1. **Quanto do erro adicional pós-2019 é covariate shift?**  \n"
        "   Ver `iw_decomposition.csv` — coluna `recovery_pct` por ano.\n\n"
        "2. **Fine-tuning leve recupera quantos % do excesso?**  \n"
        "   Ver `retrain_results.csv` — coluna `recovery_pct` e critério de aceite.\n\n"
        "3. **Drift entre anos sobrevive ao corte por divisão?**  \n"
        "   Ver `by_division_2x2_table.csv` — delta ADE antes/depois de 2022 por divisão.\n"
    ),
]


def main() -> None:
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    # check if already added
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if MARKER in src:
            print(f"[info] Seções do Mês 3 já existem no notebook. Pulando.")
            return

    # check RES_MES3 variable exists in notebook (needed by the new cells)
    res_var_ok = any(
        "RES_MES3" in "".join(c.get("source", []))
        for c in nb["cells"]
    )
    if not res_var_ok:
        print("[warn] Variável RES_MES3 não encontrada no notebook.")
        print("  Adicionando definição antes das novas seções...")
        NEW_CELLS.insert(0, cell_code(
            "# Caminhos Mês 3\n"
            "RES_MES3 = ROOT / 'Relas' / 'results' / 'mes3'\n"
            "RES_MES3.mkdir(parents=True, exist_ok=True)"
        ))

    original_n = len(nb["cells"])
    nb["cells"].extend(NEW_CELLS)

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"[ok] Notebook atualizado: {original_n} → {len(nb['cells'])} células.")


if __name__ == "__main__":
    main()
