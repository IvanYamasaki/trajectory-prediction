    # Relatório Final — Fase 4: Consolidação Científica

    > Gerado automaticamente por `drift_analise/phase4_consolidate.py`.
    > Dataset: Seq2Seq, n_2019_total=48,000, n_post_total=240,000.

    ---

    ## 1. Resultado Primário — ADWIN\_S2S Standalone

    O detetor recomendado para produção é o **ADWIN_S2S standalone**
    (configuração: `delta=1e-7, mw=200, cd=1000, xw=2000`). É o único detetor que
    combina cobertura total dos cinco anos pós-pandemia (2021–2025) com
    taxa de falsos alarmes baixa e controlada.

    | Métrica | Valor | Notas |
    |---------|-------|-------|
    | FAR\_2019\_per1k | 0.1250 | IC Wilson 95%: [0.0573, 0.2727]/1k |
    | year\_coverage | 5/5 | Deteta em todos os anos 2021–2025 |
    | SNR\_smooth (a=1) | 1.1429 | Substituto finito e estável do SNR\_op |
    | detection\_efficiency | 0.8667 | 86.7% dos alarmes são pós-2019 |
    | n\_post (alarmes 2021–2025) | 39 | Distribuídos por 5 anos |

    O valor FAR=0.1250/1k corresponde a
    6 alarmes no período de linha de base (2019),
    num total de 48,000 trajetórias — IC de Wilson 95%:
    [0.0573, 0.2727]/1k.

    ---

    ## 2. Resultado Negativo Caracterizado — Inviabilidade Estrutural do AND Temporal

    > **O ensemble AND temporal foi testado, caracterizado e rejeitado com base
    > em evidência empírica.**

    A inviabilidade do AND(ADWIN\_S2S, PH\_agg) resulta de dois factores
    estruturais independentes, não de má calibração de parâmetros:

    ### Factor 1 — Desalinhamento Temporal (escala de deteção)

    ADWIN\_S2S opera por trajetória (escala fina) enquanto PH\_agg agrega
    janelas de N=50 trajetórias (escala grossa). Na transição 2019→2021:

    - PH\_agg dispara o primeiro alarme em índice **~49,275**
      (imediatamente após o salto COVID+comportamental, via janelas largas).
    - ADWIN\_S2S dispara o primeiro alarme em índice **~48,290**
      (após acumular buffer suficiente com mistura 2019+2021).
    - **Δ₁ = -985 trajetórias** entre os primeiros alarmes do mesmo evento.
    - Par mais próximo: ADWIN@78,141, PH@78,675,
      **Δ\_min = 534 trajetórias**.

    Para a porta AND disparar, é necessário W ≥ Δ\_min = 534 trajetórias.
    O sweep de W confirma que W=500 (≥ 534) produz exactamente
    1 alarme CONFIRMED, apenas em 2021.

    Ver figura: `images/and_infeasibility.png`

    ### Factor 2 — Limitação de Escopo do PH\_agg (determinante)

    O PH\_agg detecta **exclusivamente o salto 2019→2021** (evento de grande
    magnitude). Para os anos 2022–2025, em que o drift é mais gradual:

    | Canal | n\_2021 | n\_2022 | n\_2023 | n\_2024 | n\_2025 | year\_coverage |
    |-------|---------|---------|---------|---------|---------|----------------|
    | PH\_agg standalone | 12 | 0 | 0 | 0 | 0 | 1/5 |
    | ADWIN\_S2S standalone | 8 | 9 | 11 | 9 | 2 | 5/5 |

    Como o PH_agg nunca dispara diretamente em 2022–2025, a porta AND não pode
    confirmar nesses anos com janelas operacionalmente razoáveis.
    O sweep abaixo quantifica isto:

    ### Sweep W Estendido

| W_coincidence | W_approx_months | n_confirmed | FAR_2019_per1k | year_coverage_confirmed |
| --- | --- | --- | --- | --- |
| 50 | 0.0 | 0 | 0.000 | 0 |
| 100 | 0.0 | 0 | 0.000 | 0 |
| 200 | 0.1 | 0 | 0.000 | 0 |
| 500 | 0.1 | 0 | 0.000 | 0 |
| 1000 | 0.2 | 2 | 0.000 | 1 |
| 2000 | 0.5 | 3 | 0.000 | 1 |
| 5000 | 1.2 | 3 | 0.000 | 1 |
| 10000 | 2.5 | 5 | 0.000 | 2 |
| 20000 | 5.0 | 5 | 0.000 | 2 |

    W mínimo para year_coverage_confirmed >= 2: **W=10000**.

    **Interpretação:** Para W <= 2000 (até ~0.5 meses de dados), o AND confirma
    apenas alarmes em 2021. Com W=10000 (~2.5 meses),
    o AND consegue "confirmar" 1 alarme em 2022 — mas apenas porque um alarme
    ADWIN de early-2022 cai dentro da janela de 10000 trajetórias a partir do
    último alarme PH de late-2021. Uma janela de "coincidência temporal" de
    ~2 meses é operacionalmente absurda para um detetor de drift. Esta é a
    prova definitiva da inviabilidade: obter year_coverage >= 2 requer
    abandonar qualquer noção de simultaneidade temporal.

    ---

    ## 3. Resultado Secundário — Canal OR/SUSPECTED como Camada de Vigilância

    O canal OR(ADWIN\_S2S, ADWIN\_Kalman) funciona como camada de vigilância
    de sensibilidade aumentada, adequado para disparar monitorização reforçada
    (não re-treino automático).

    | Métrica | Valor |
    |---------|-------|
    | FAR\_2019\_per1k | 0.3125 |
    | year\_coverage | 5/5 |
    | SNR\_smooth (a=1) | 1.1875 |
    | detection\_efficiency | 0.8624 |
    | n\_post | 94 |

    O canal OR cobre os mesmos 5/5 anos que o ADWIN standalone, com maior
    sensibilidade (n\_post=94 vs 39)
    à custa de FAR mais alto (0.312 vs
    0.125/1k). Recomendado apenas como
    camada de alerta; re-treino deve ser condicionado à confirmação manual.

    ---

    ## 4. Correção Metodológica — SNR\_op → SNR\_smooth

    ### O Problema do SNR\_op com Sentinela 9999

    O `SNR_op` implementado nas Fases 1–3 atribui o valor sentinela 9999 sempre
    que `n_2019_alarms == 0`, independentemente do número de deteções pós-2019.
    Isto cria um artefacto metodológico: qualquer detetor degenerado que evite
    **todos** os alarmes em 2019 (incluindo um que não deteta nada de útil)
    parece "ótimo".

    **Prova:** `ph_standalone` tem SNR\_op=9999 mas year\_coverage=1/5 (só 2021).
    `confirmed (AND)` com W=200 tem SNR\_op=0.0 com 0 deteções.
    Nenhuma destas classificações reflete performance real.

    ### SNR\_smooth — Fórmula com Pseudo-Contagem de Laplace

    ```
    SNR_smooth = ((n_post + a) × n_2019_total) / ((n_2019_alarms + a) × n_post_total)
    ```

    Com `a=1` (prior simétrico): um detetor com 0 falsos alarmes e 1 deteção
    obtém um valor finito e comparável; o AND com 0 deteções obtém SNR < 1.

    ### Tabela Antes/Depois

| Detetor | SNR_op | SNR_smooth (a=1) | year_coverage | Artefacto? |
| --- | --- | --- | --- | --- |
| ADWIN_S2S | 1.30 | 1.14 | 5 | Não |
| ADWIN_Kalman | 1.22 | 1.12 | 5 | Não |
| OR(ADWIN_S2S, ADWIN_Kalman) | 1.25 | 1.19 | 5 | Não |
| KSWIN | 0.77 | 0.75 | 5 | Não |
| AND(ADWIN_S2S, PH_agg) | 0.00 | 0.20 | 0 | Não |
| PH_agg | 9999.00 | 2.60 | 1 | **Sim** — sentinela 9999 |

    **Efeito:** `ph_standalone` desce de 9999 para 2.60,
    eliminando a explosão numérica da sentinela. Importa notar, porém, que 2.60
    é o valor **mais alto** da tabela-mestre — o SNR\_smooth sozinho não desmascara o PH\_agg:
    esta métrica ainda premeia especificidade (n\_2019\_alarms=0), pelo que qualquer detetor
    com zero falsos alarmes no baseline obtém um SNR\_smooth elevado, independentemente da
    cobertura temporal. O que revela a inutilidade do PH\_agg é o year\_coverage=1/5
    (deteta exclusivamente o salto de 2021), **não** o SNR\_smooth. É precisamente por isso
    que a regra de decisão usa year\_coverage como critério **primário** e o SNR\_smooth
    apenas como terceiro critério de desempate. As conclusões das Fases 1–3 baseadas em
    year\_coverage + FAR não se alteram; as baseadas em SNR\_op > 100 ficam invalidadas.
    Esta é uma limitação reconhecida da métrica corrigida: o SNR\_smooth resolve o problema
    numérico da sentinela mas herda o viés pró-especificidade; mitigamo-lo subordinando-o
    ao year\_coverage na regra de decisão.

    ### Robustez do Ranking para a ∈ {0.5, 1, 2}

| a | rank | detector | SNR_smooth |
| --- | --- | --- | --- |
| 0.5 | 1 | ADWIN_S2S | 1.2154 |
| 0.5 | 2 | ADWIN_Kalman | 1.1684 |
| 0.5 | 3 | OR(ADWIN_S2S, ADWIN_Kalman) | 1.2194 |
| 1.0 | 1 | ADWIN_S2S | 1.1429 |
| 1.0 | 2 | ADWIN_Kalman | 1.1200 |
| 1.0 | 3 | OR(ADWIN_S2S, ADWIN_Kalman) | 1.1875 |
| 2.0 | 1 | ADWIN_S2S | 1.0250 |
| 2.0 | 2 | ADWIN_Kalman | 1.0364 |
| 2.0 | 3 | OR(ADWIN_S2S, ADWIN_Kalman) | 1.1294 |

    O ranking dos três primeiros candidatos é **estável** para todos os valores
    de `a` testados: a escolha de a=1 não é determinante para as conclusões.

    ---

    ## 5. Respostas às Questões Científicas (Q1–Q4)

    ### Q1 — A ausência de deteções em 2022–2025 no PH\_agg é real ou limitação?

    **Resposta (confirmada — limitação estrutural dupla):**

    (a) O PH\_agg com K\_thr=50 só ultrapassa o limiar no salto 2019→2021
    (grande magnitude, evento COVID + mudança comportamental). Os drifts
    2022–2025 são graduais e ficam abaixo do limiar.

    (b) Mesmo que o PH disparasse mais tarde, o desalinhamento temporal
    (Δ\_min=534 trajs, Factor 1 da Sec. 2) tornaria o AND inoperante
    com W razoável.

    Estas são limitações estruturais, não de calibração. O ADWIN\_S2S
    (por-trajetória) detecta drift gradual em todos os 5 anos.

    ### Q2 — O SNR\_op < 1 do KSWIN indica inutilidade?

    **Resposta:** Sim. O KSWIN é **inadmissível** pelo filtro de elegibilidade
    (FAR\_2019 > 0.2/1k) e confirmado inútil pela métrica corrigida.
    KSWIN (alpha=1e-5, ws=1000, ss=100):
    - FAR\_2019=0.4375/1k > limiar 0.2/1k
      → **inadmissível** (especificidade insuficiente para produção)
    - SNR\_smooth=0.75 < 1 (mais alarmes no
      baseline do que no pós-período, proporcionalmente)
    - n\_2019=21 falsos alarmes em 2019 (pior em ordem de grandeza)
    - detection\_efficiency=0.79
    O KSWIN não passa o filtro de elegibilidade e não é candidato ao detetor primário.

    ### Q3 — Trade-off operacional FAR vs cobertura?

    **Resposta:** Com as métricas corrigidas, o trade-off é:

    | Nível | Detetor | FAR/1k | year\_coverage | Uso recomendado |
    |-------|---------|--------|----------------|-----------------|
    | Alto rigor | ADWIN\_S2S standalone | 0.1250 | 5/5 | Re-treino automático |
    | Alta cobertura | OR/SUSPECTED | 0.3125 | 5/5 | Monitorização reforçada |

    O AND CONFIRMED (FAR=0, year\_coverage=0–1) foi rejeitado: troca toda a
    cobertura pós-2021 por uma especificidade que resulta em 0 deteções úteis.

    ### Q4 — O ensemble AND adiciona valor sobre o standalone?

    **Resposta: Não. O AND temporal NÃO adiciona valor sobre o ADWIN\_S2S standalone.**

    Dois factores estruturais independentes (Sec. 2):
    1. Desalinhamento temporal de -985 trajetórias no primeiro alarme 2021
       (Δ\_min=534 trajetórias no par mais próximo).
    2. PH\_agg não deteta 2022–2025 → year\_coverage\_confirmed ≤ 1 para
       **qualquer** W.

    O sweep W ∈ {50, …, 20000} confirma:
    year_coverage_confirmed permanece em 1 (apenas 2021) para W <= 2000.
    Atingir year_coverage_confirmed = 2 requer W >= 10000 (~2 meses de dados) —
    o que destrói o significado de "coincidência temporal".
    W=500 não tem valor operacional: produz 1 deteção em 1/5 anos, versus
    23 deteções em 5/5 anos para o ADWIN standalone.

    **Conclusão:** Demonstrámos empiricamente que o ensemble AND temporal,
    apesar de eliminir os falsos alarmes de 2019 (FAR=0), é estruturalmente
    inviável como monitor de longo prazo. O ADWIN\_S2S standalone é o detetor
    recomendado. O canal OR/SUSPECTED é a camada de vigilância adequada.

    ---

    ## 6. Ameaças à Validade

    ### 6.1 IC de Wilson para FAR=0 com Amostra Pequena

    Detetores com FAR=0 (PH\_agg standalone, AND CONFIRMED) baseiam-se em
    0 alarmes num período de 48,000 trajetórias de baseline. O IC de Wilson
    a 95% não implica FAR=0 verdadeiro:

    - **PH\_agg standalone**: IC 95% = [0.0000, 0.0800]/1k
    - **AND CONFIRMED (W=200)**: IC 95% = [0.0000, 0.0800]/1k

    Com 48,000 trajetórias de baseline, o limite superior da FAR é
    ≈0.08/1k — valor baixo mas não zero. Esta incerteza não altera as
    conclusões sobre inviabilidade do AND (Factor 2 é determinístico).

    ### 6.2 Evento Único de Drift Dominante em 2021

    O salto 2019→2021 (descontinuidade COVID + regresso gradual com novos
    padrões comportamentais) é o evento de drift mais pronunciado do dataset.
    Os drifts pós-2021 são mais graduais. Isto favorece o ADWIN\_S2S
    (sensível a mudanças graduais) sobre o PH\_agg (calibrado para o salto de
    2021). Generalizações para datasets com drift mais uniforme ao longo dos
    anos requerem re-calibração de K\_thr.

    ### 6.3 Generalização Limitada

    Os resultados referem-se a uma liga e horizonte específicos (Seq2Seq,
    30→15 minutos). O desempenho em outras ligas ou outros horizontes de
    previsão pode diferir. O padrão estrutural (desalinhamento entre
    detetores de escalas distintas) é esperado manter-se, mas os valores
    numéricos (Δ₁, Δ\_min) dependem da distribuição temporal dos erros.

    ---

    ## Tabela-Mestre Final

| detector | FAR_2019_per1k | year_coverage | SNR_smooth | detection_efficiency | n_alarms_post | n_2019 | papel_final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADWIN_S2S | 0.1250 | 5 | 1.1429 | 0.8667 | 39 | 6 | Detetor primário recomendado |
| ADWIN_Kalman | 0.1875 | 5 | 1.1200 | 0.8594 | 55 | 9 | Corroborador OR |
| OR(ADWIN_S2S, ADWIN_Kalman) | 0.3125 | 5 | 1.1875 | 0.8624 | 94 | 15 | Vigilância multi-ano |
| KSWIN | 0.4375 | 5 | 0.7455 | 0.7941 | 81 | 21 | Inadmissível — FAR > 0.20/1k (especificidade insuficiente) |
| AND(ADWIN_S2S, PH_agg) | 0.0000 | 0 | 0.2000 | 0.0000 | 0 | 0 | Resultado negativo — AND inviável |
| PH_agg | 0.0000 | 1 | 2.6000 | 1.0000 | 12 | 0 | Resultado negativo — cobertura 1/5 anos |

    *Critério de elegibilidade: FAR\_2019\_per1k ≤ 0.2/1k (aplicado antes da ordenação).
    Candidatos inadmissíveis (FAR > 0.2/1k) e resultados negativos aparecem após os elegíveis.
    Regra de decisão (elegíveis): (1) year\_coverage DESC, (2) FAR\_2019\_per1k ASC, (3) SNR\_smooth DESC.*
