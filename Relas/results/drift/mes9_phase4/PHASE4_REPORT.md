    # Fase 4 — Consolidação Científica e Resultado Final

    > Esta fase **não re-executa grids**. Re-tabula os finalistas das Fases 1–3
    > com métricas corrigidas, caracteriza formalmente a inviabilidade do AND, e
    > produz os artefactos finais para o paper.

    ## Objectivo

    Corrigir a métrica `SNR_op` (sentinela 9999), consolidar os candidatos finais
    numa única tabela ranqueada, e produzir a figura e o texto que vão ao paper.
    Dataset: Seq2Seq, horizonte 30→15 min, n_2019_total=48,000, n_post_total=240,000.

    ---
    ## Filtro de Elegibilidade (Fase 4.1)

    Antes da regra de ordenação, aplica-se o critério de admissibilidade:

    > **FAR\_2019\_per1k ≤ 0.2/1k** — candidatos com FAR superior são
    > classificados como **Inadmissíveis** e não concorrem ao posto de detetor primário.

    Justificação: um detetor com FAR=0.4375/1k produz ~21 falsos alarmes por cada
    48 000 trajetórias de linha de base — especificidade insuficiente para re-treino
    automático. O limiar de 0.2/1k admite até ~10 alarmes/48 000 trajs,
    compatível com revisão humana esporádica.

    ---
    ## Métricas Novas (substituem uso conclusivo de SNR_op)

    | Métrica | Fórmula | Direcção |
    |---------|---------|---------|
    | `year_coverage` | nº anos em {2021..2025} com ≥1 alarme (0–5) | **acima melhor** |
    | `SNR_smooth` | `((n_post+a)×n_2019_total) / ((n_2019_alarms+a)×n_post_total)` com a=1 | **acima melhor** |
    | `detection_efficiency` | `n_alarms_post / n_alarms_total` | **acima melhor** |
    | `FAR_2019_per1k` | mantida sem alteração | **abaixo melhor** |

    **Justificação do SNR_smooth:** a pseudo-contagem de Laplace `a=1` elimina a sentinela 9999
    (disparada quando n_2019_alarms=0). Ranking estável para a ∈ {0.5, 1, 2}.

    **Regra de decisão (sobre elegíveis):**
    1. `year_coverage` DESC — sensibilidade multi-ano é o critério primário
    2. `FAR_2019_per1k` ASC — menor ruído de linha de base (desempate)
    3. `SNR_smooth` DESC — qualidade do sinal (desempate final)

    ---
    ## Tabela-Mestre Final (phase4_master_table.csv)

| detector | FAR_2019_per1k | year_coverage | SNR_smooth | detection_efficiency | n_alarms_post | n_2019 | papel_final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADWIN_S2S | 0.0417 | 5 | 1.6000 | 0.9200 | 23 | 2 | Detetor primário recomendado |
| ADWIN_Kalman | 0.1042 | 5 | 1.5333 | 0.9000 | 45 | 5 | Corroborador OR |
| OR(ADWIN_S2S, ADWIN_Kalman) | 0.1458 | 5 | 1.7250 | 0.9067 | 68 | 7 | Vigilância multi-ano |
| KSWIN | 0.4375 | 5 | 0.7455 | 0.7941 | 81 | 21 | Inadmissível — FAR > 0.20/1k (especificidade insuficiente) |
| AND(ADWIN_S2S, PH_agg) | 0.0000 | 0 | 0.2000 | 0.0000 | 0 | 0 | Resultado negativo — AND inviável |
| PH_agg | 0.0000 | 1 | 2.6000 | 1.0000 | 12 | 0 | Resultado negativo — cobertura 1/5 anos |

    *Critério de elegibilidade: FAR\_2019\_per1k ≤ 0.2/1k aplicado antes da ordenação.
    Inadmissíveis e resultados negativos aparecem após os elegíveis ranqueados.*

    ---
    ## Correção Metodológica — SNR_op → SNR_smooth

    O `SNR_op` atribui sentinela 9999 sempre que `n_2019_alarms == 0`,
    independentemente do número de deteções pós-2019.

    | Detetor | SNR_op | SNR_smooth | year_coverage | Artefacto? |
    |---------|--------|------------|---------------|------------|
    | ADWIN_S2S | 2.30 | 1.60 | 5/5 | Não |
    | ADWIN_Kalman | 1.80 | 1.53 | 5/5 | Não |
    | OR(ADWIN_S2S, Kalman) | 1.94 | 1.73 | 5/5 | Não |
    | KSWIN | 0.77 | 0.75 | 5/5 | Não |
    | PH_agg | **9999** | **2.60** | 1/5 | **Sim** — sentinela 9999 (n_2019_alarms=0) |
    | AND(ADWIN_S2S, PH_agg) | 0.00 | **0.20** | 0/5 | Não (0 deteções) |

    **Efeito:** `ph_standalone` desce de 9999 para 2.60, eliminando a explosão
    numérica da sentinela. Importa notar que 2.60 é o valor **mais alto** da
    tabela-mestre — o SNR_smooth sozinho não desmascara o PH_agg, pois ainda premeia
    especificidade (n_2019_alarms=0). O que revela a sua inutilidade é o year_coverage=1/5
    (deteta só 2021), não o SNR_smooth. A regra de decisão usa year_coverage como critério
    **primário** e o SNR_smooth apenas como desempate final. Limitação reconhecida: o
    SNR_smooth resolve o problema numérico da sentinela mas herda o viés pró-especificidade;
    mitigamo-lo subordinando-o ao year_coverage na regra de decisão.

    ---
    ## Inviabilidade Estrutural do AND Temporal

    > **O ensemble AND temporal foi testado, caracterizado e rejeitado com base em evidência.**

    ### Factor 1 — Desalinhamento Temporal

    | Evento | PH_agg (janelas N=50) | ADWIN_S2S (por-trajetória) |
    |--------|----------------------|---------------------------|
    | Primeiro alarme em 2021 | índice **~49,275** | índice **~67,585** |
    | Δ₁ (mesmo evento) | — | **18,310 trajetórias** |
    | Par mais próximo | PH@81,125 | ADWIN@80,911 |
    | Δ_min (par mais próximo) | — | **214 trajetórias** |

    ### Factor 2 — Limitação de Escopo do PH_agg

    | Canal | n_2021 | n_2022 | n_2023 | n_2024 | n_2025 | year_coverage |
    |-------|--------|--------|--------|--------|--------|---------------|
    | PH_agg standalone | 12 | 0 | 0 | 0 | 0 | **1/5** |
    | ADWIN_S2S standalone | 11 | 1 | 5 | 4 | 2 | **5/5** |

    ### Sweep W Estendido

| W_coincidence | W_approx_months | n_confirmed | FAR_2019_per1k | year_coverage_confirmed | note |
| --- | --- | --- | --- | --- | --- |
| 50 | 0.0 | 0 | 0.000 | 0 | Δ_min=214 > W=50: sem coincidências |
| 100 | 0.0 | 0 | 0.000 | 0 | Δ_min=214 > W=100: sem coincidências |
| 200 | 0.1 | 0 | 0.000 | 0 | Δ_min=214 > W=200: sem coincidências |
| 500 | 0.1 | 1 | 0.000 | 1 | year_coverage_confirmed=1 (só 2021) — PH_agg nunca dispara em 2022-2025 |
| 1000 | 0.2 | 1 | 0.000 | 1 | year_coverage_confirmed=1 (só 2021) — PH_agg nunca dispara em 2022-2025 |
| 2000 | 0.5 | 1 | 0.000 | 1 | year_coverage_confirmed=1 (só 2021) — PH_agg nunca dispara em 2022-2025 |
| 5000 | 1.2 | 2 | 0.000 | 1 | year_coverage_confirmed=1 (só 2021) — PH_agg nunca dispara em 2022-2025 |
| 10000 | 2.5 | 4 | 0.000 | 2 | year_coverage_confirmed=2 via pairing cross-ano: ADWIN early-2022 dentro de W=10000 (~2.5 meses) do ultimo PH late-2021 — W operacionalmente absurdo |
| 20000 | 5.0 | 4 | 0.000 | 2 | year_coverage_confirmed=2 via pairing cross-ano: ADWIN early-2022 dentro de W=20000 (~5.0 meses) do ultimo PH late-2021 — W operacionalmente absurdo |

    **Achado crítico:** year_coverage_confirmed=2 exige W≥10000
    (~2.5 meses). Uma janela de "coincidência temporal" de 2.5 meses é operacionalmente
    absurda. Este resultado reforça a inviabilidade estrutural.

    ---
    ## Intervalos de Confiança de Wilson (FAR=0)

    | Detetor | k (2019) | n (2019 trajs) | FAR obs./1k | IC Wilson 95% [lower, upper]/1k |
    |---------|----------|----------------|------------|----------------------------------|
    | ADWIN_S2S | 2 | 48,000 | 0.0417 | [0.0114, 0.1519] |
    | PH_agg standalone | 0 | 48,000 | 0.0000 | [0.0000, 0.0800] |
    | AND CONFIRMED (W=200) | 0 | 48,000 | 0.0000 | [0.0000, 0.0800] |

    ---
    ## Recomendação Operacional Final

    **Detetor primário:** `ADWIN_S2S standalone`
    **Critério de re-treino:** alarme ADWIN com cooldown=200 trajs.
    **Camada de vigilância:** canal OR(ADWIN_S2S, ADWIN_Kalman).

    | Parâmetro | Detetor Primário (ADWIN_S2S) | Canal OR (Kalman) |
    |-----------|------------------------------|-------------------|
    | Model | Seq2Seq | Kalman |
    | detector | ADWINLite + robz_w200 | ADWINLite + robz_w200 |
    | delta | 1e-7 | 1e-7 |
    | min_window | 600 | 200 |
    | cooldown | 200 | 1000 |
    | max_window | 2000 | 5000 |
    | FAR_2019 | 0.0417/1k | 0.1042/1k |
    | year_coverage | 5/5 | 5/5 |
    | n_post | 23 | 45 |

    **Ensemble AND(ADWIN_S2S, PH_agg): testado, caracterizado e rejeitado.**
    Não aparece como recomendação em nenhum output da Fase 4.

    ---
    ## Artefactos Gerados

    | Ficheiro | Conteúdo |
    |----------|---------|
    | `phase4_master_table.csv` | 6 candidatos finais ranqueados com métricas corrigidas |
    | `W_infeasibility_sweep.csv` | W ∈ {50..20000} com year_coverage_confirmed e nota |
    | `images/and_infeasibility.png` | Figura 300 dpi: desalinhamento ADWIN vs PH (Δ₁=18,310, Δ_min=214) |
    | `PHASE4_FINAL.md` | Texto completo para o paper (Q1–Q4, IC Wilson, ameaças à validade) |
    | `PHASE4_REPORT.md` | Este ficheiro — resumo estruturado no formato das fases anteriores |
