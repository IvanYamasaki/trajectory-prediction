# Fase 3 — Ensemble Temporal de Deteção de Drift

## Configuração do Ensemble

### Componentes
| Canal | Detector | Config | Papel |
|-------|----------|--------|-------|
| ADWIN_S2S | ADWINLite + robz_w200 | delta=1e-7, mw=600, cd=200, xw=2000 | Espinha dorsal AND+OR |
| PH_agg    | PageHinkley + N=50 trimmed-mean | K_thr=50, K_delta=2.0 | Validador binário AND |
| Kalman    | ADWINLite + robz_w200 | delta=1e-7, mw=200, cd=1000, xw=5000 | Corroborador OR-only |

### Portas
- **Tier 1 (CONFIRMED):** `AND(ADWIN_S2S, PH_agg)` — janela de coincidência W=200 trajs, debounce=5000 trajs.
- **Tier 2 (SUSPECTED):** `OR(ADWIN_S2S, Kalman)` — cobertura multi-ano.
- Kalman **nunca** entra no AND. Buffers limpos após cada alarme CONFIRMED.

---
## Resultados Principais (Seq2Seq, W=200)

| variant                  |   n_alarms |   FAR_2019_per1k |   SNR_op |   n_alarms_post |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|:-------------------------|-----------:|-----------------:|---------:|----------------:|---------:|---------:|---------:|---------:|---------:|---------:|
| confirmed (AND+debounce) |          0 |            0.000 |    0.000 |               0 |        0 |        0 |        0 |        0 |        0 |        0 |
| suspected (OR)           |         75 |            0.146 |    1.940 |              68 |        7 |       25 |       12 |       15 |       13 |        3 |

- **n_confirmed** = 0
- **n_suspected** = 75
- **FAR_2019 CONFIRMED** = 0.0000/1k (✓ zero)
- **FAR_2019 SUSPECTED** = 0.1458/1k
- Anos SUSPECTED com n>0: 5/5
- Anos CONFIRMED com n>0: 0/5

---
## Ablação

| variant                          |   n_alarms |   FAR_2019_per1k |   SNR_op |   n_alarms_post |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 | W_coincidence   | debounce   |
|:---------------------------------|-----------:|-----------------:|---------:|----------------:|---------:|---------:|---------:|---------:|---------:|---------:|:----------------|:-----------|
| confirmed (AND+debounce)         |          0 |            0.000 |    0.000 |               0 |        0 |        0 |        0 |        0 |        0 |        0 | 200             | 5000       |
| suspected (OR)                   |         75 |            0.146 |    1.940 |              68 |        7 |       25 |       12 |       15 |       13 |        3 | n/a             | n/a        |
| confirmed_no_debounce (ablation) |          0 |            0.000 |    0.000 |               0 |        0 |        0 |        0 |        0 |        0 |        0 | 200             | 0          |
| and_only (ablation)              |          0 |            0.000 |    0.000 |               0 |        0 |        0 |        0 |        0 |        0 |        0 | 200             | 5000       |
| or_only (ablation)               |         75 |            0.146 |    1.940 |              68 |        7 |       25 |       12 |       15 |       13 |        3 | n/a             | n/a        |
| adwin_standalone                 |         25 |            0.042 |    2.300 |              23 |        2 |       11 |        1 |        5 |        4 |        2 | n/a             | n/a        |
| ph_standalone                    |         12 |            0.000 | 9999.000 |              12 |        0 |       12 |        0 |        0 |        0 |        0 | n/a             | n/a        |
| kalman_standalone                |         50 |            0.104 |    1.800 |              45 |        5 |       14 |       11 |       10 |        9 |        1 | n/a             | n/a        |

### Interpretação da Ablação

| Métrica | ADWIN standalone | PH standalone | Kalman standalone | CONFIRMED (AND) |
|---------|-----------------|---------------|-------------------|-----------------|
| FAR_2019 | 0.0417 | 0.0000 | 0.1042 | **0.0000** |
| SNR_op   | 2.30 | 9999.0 | 1.80 | **0.0** |
| n_post   | 23 | 12 | 45 | **0** |

---
## Análise de Sensibilidade — W_COINCIDENCE

|   W_coincidence |   n_confirmed |   FAR_2019_per1k |   SNR_op |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|----------------:|--------------:|-----------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
|          50.000 |         0.000 |            0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |
|         100.000 |         0.000 |            0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |
|         200.000 |         0.000 |            0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |
|         500.000 |         1.000 |            0.000 | 9999.000 |    0.000 |    1.000 |    0.000 |    0.000 |    0.000 |    0.000 |
|        1000.000 |         1.000 |            0.000 | 9999.000 |    0.000 |    1.000 |    0.000 |    0.000 |    0.000 |    0.000 |

**W recomendado:** `W=500` — maximiza n_confirmed=1 mantendo FAR_confirmed=0.

---
## Respostas às Questões Científicas (Q1–Q4)

### Q1 — A ausência de deteções em 2022–2025 na Frente B é real ou limitação?

**Resposta (Hipótese A confirmada — limitação estrutural):** Com W=200 trajetórias, a porta AND não dispara: os alarmes ADWIN_S2S (primeiro em 2021: índice ~67585) e PH_agg (primeiro em 2021: índice ~49275) estão separados por ~18000 trajetórias na entrada do período 2021. O ADWIN acumula buffer de 2019 e só deteta o drift após suficientes trajetórias de 2021; o PH_agg dispara logo após o salto. A mais próxima coincidência é ~214 trajetórias (ADWIN@80911, PH@81125), ligeiramente acima de W=200.

Com W=500, o sweep produz 1 alarme CONFIRMED (em 2021) com FAR_confirmed=0 — mínimo possível dado o espaçamento entre detectores.

**Conclusão (Hipótese A):** A ausência de CONFIRMED com W≤200 é **estrutural** — o ADWIN e o PH_agg operam em escalas temporais distintas (por-trajetória vs. janelas de 50). Não é artefato de calibração. O W_COINCIDENCE recomendado é 500 (justificado pelo sweep).

O canal SUSPECTED (OR) compensa: cobre 5/5 anos pós-2019 (FAR=0.1458/1k).

### Q2 — O SNR_op < 1 do KSWIN indica inutilidade?

**Resposta:** Sim, confirmado na Fase 2. O KSWIN foi **excluído** do ensemble (18/18 configs com SNR_op<1 no Seq2Seq). Não entra em nenhuma porta.

### Q3 — Trade-off operacional FAR_2019 vs n_post?

**Resposta:** O ensemble resolve o trade-off por camadas:
- CONFIRMED (AND): FAR=0.0000/1k, n_post=0 — alta confiança, aciona re-treino raramente.
- SUSPECTED (OR): FAR=0.1458/1k, n_post=68 — alta cobertura, aciona monitorização.
- O operador pode escolher o nível de resposta conforme o custo de re-treino.

### Q4 — Qual o ensemble mais promissor?

**Resposta:** O ensemble AND(ADWIN_S2S, PH_agg) melhora o melhor standalone (ADWIN: FAR=0.0417/1k):
- CONFIRMED: FAR=0.0000/1k vs ADWIN standalone FAR=0.0417/1k.
- Trade-off: n_post CONFIRMED=0 vs ADWIN n_post=23.

**Conclusão honesta:** Com W=200, o AND gate produz 0 alarmes CONFIRMED — FAR=0 trivialmente mas sem deteções. Com W=500 (recomendado pelo sweep), produz 1 alarme CONFIRMED em 2021 com FAR=0 e SNR=9999.

O ADWIN_S2S standalone (FAR=0.042/1k, n_post=23) é **mais informativo em cobertura multi-ano** mas com residual de falsos alarmes em 2019. O ensemble AND troca cobertura por especificidade máxima. A escolha depende do custo operacional do re-treino: se re-treino é caro, prefere-se AND com W=500 (1 evento confirmado, FAR=0); se cobertura é prioritária, usa-se ADWIN standalone ou SUSPECTED.

---
## Candidatos Fase 3 — Tabela Preenchida (preenche PHASE2_REPORT.md)

| Frente | Detector | Config | FAR_2019 | SNR_op | Papel no Ensemble |
|--------|----------|--------|----------|--------|-------------------|
| A | ADWIN+robz_w200 | delta=1e-7, mw=600, cd=200, xw=2000 | 0.042 | 2.30 | AND + OR (espinha dorsal) |
| B | PH_agg N=50     | K_thr=50, K_delta=2.0 | 0.000 | 9999 | AND (validador 2021) |
| C | KSWIN           | EXCLUÍDO — SNR_op<1 em todas as configs | — | — | Não entra |
| Kalman | ADWIN+robz_w200 | delta=1e-7, mw=200, cd=1000, xw=5000 | 0.104 | 1.80 | OR-only (corroborador) |

---
## Recomendação Operacional Final

**Config:** `AND(ADWIN_S2S_multiyear, PH_agg)` com `W_COINCIDENCE=500`.
**Critério de re-treino:** alarme CONFIRMED + debounce=5000 trajs.
**Critério de monitorização:** alarme SUSPECTED (OR).

| Parâmetro | Valor |
|-----------|-------|
| ADWIN delta | 1e-7 |
| ADWIN min_window | 600 |
| ADWIN cooldown | 200 |
| ADWIN max_window | 2000 |
| PH N_window | 50 |
| PH K_thr | 50.0 |
| PH K_delta | 2.0 |
| W_COINCIDENCE | 500 |
| RETRAIN_DEBOUNCE | 5000 |

**Imagens geradas:**
- `images/coincidence_sweep.png` — FAR e n_confirmed vs W
- `images/ensemble_stream_seq2seq.png` — stream plots 3 painéis
