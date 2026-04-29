# Review do Mês 3 — Análise crítica dos resultados

> Lidos: `sample_size_sensitivity.csv`, `iw_decomposition.csv`,
> `retrain_results.csv`, `by_division_decomposition.csv`,
> `by_division_2x2_table.csv`. Esta é a leitura honesta antes do Mês 4
> — sem maquiagem.

---

## Resumo executivo

O Mês 3 entregou tudo que estava no plano (sensibilidade amostral, IW
via LSIF, retreino seletivo, corte por divisão), **mas a leitura dos
números expõe três problemas que precisam ser resolvidos antes de
qualquer escrita formal**:

1. O baseline 2019 está subamostrado — ADE de 2019 cai de **4.92 mm**
   (n=3) para **4.53 mm** (n=6). Os 3 jogos de 2019 sortidos parecem ser
   *mais difíceis* que a média do ano.
2. Como consequência, **ADE de 2022 (4.29 mm) e 2024 (4.72 mm) ficam
   ABAIXO de 2019 (4.92 mm)** com n=3. Toda a "decomposição covariate
   vs concept" fica viciada nesse ponto.
3. **Quando cortamos por Division A**, o drift entre 2019 e pós-2022
   praticamente desaparece (`delta = -0.05 mm`). Isso é potencialmente o
   achado mais forte do Mês 3, mas precisa ser confirmado com cobertura
   maior antes de virar afirmação de paper.

---

## Pontos fortes

### Engenharia / pipeline

- **Pipeline completo e modular.** Cada frente do plano virou um script
  isolado e idempotente. Reorganização em pastas (`drift_analise/`,
  `model_analise/`, `model/`, `weights/`, `dataset/`) reduziu o
  acoplamento.
- **Scripts versionados em vez de células.** `compute_importance_weights.py`,
  `retrain_at_breakpoints.py`, `division_validation.py`,
  `sample_size_sensitivity.py` são arquivos rastreáveis em git e
  reaproveitáveis.
- **CSVs e PDFs padronizados.** Toda saída do Mês 3 está em
  `Relas/results/mes3/` no mesmo padrão dos meses anteriores.

### Resultados que sobrevivem ao escrutínio

- **Sensibilidade amostral foi medida.** `sample_size_sensitivity.csv`
  permite quantificar a instabilidade — não precisamos *adivinhar*
  quanto a amostra de 3 jogos influencia.
- **LSIF convergiu.** ESS ratio entre **0.42 e 0.85** em todos os pares
  ano-baseline; nenhum caso de `ess_stable=False`. O estimador de
  density-ratio está saudável; o que é frágil é a *interpretação*.
- **Decomposição por divisão é cientificamente sólida.** Mesmo com
  amostra pequena, ver `delta_A = -0.05 mm` antes/depois de 2022 é uma
  evidência *forte* de que o drift agregado é **confundimento entre
  divisões A e B**, não evolução temporal. Se isso se mantiver com mais
  jogos, vira a tese central da IC.
- **Retreino mostrou que fine-tuning leve não basta.** Resultado
  negativo, mas valioso: `recovery_pct = -4.9%` e
  `catastrophic_ratio = 0.83`. Diz "este caminho não é a saída" — fecha
  uma porta de design.

---

## Pontos fracos

### Subamostragem do baseline 2019

A tabela abaixo é direta. ADE Seq2Seq 30→15 por amostra:

| Ano  | n_per_year=3 | n_per_year=6 | Δ (n=6 − n=3) |
|------|--------------|--------------|---------------|
| 2019 | **4.92**     | **4.53**     | −0.39         |
| 2021 | 6.82         | 7.58         | +0.76         |
| 2022 | 4.29         | 4.75         | +0.46         |
| 2023 | 5.21         | 5.21         |  0.00         |
| 2024 | 4.72         | 4.50         | −0.22         |
| 2025 | 5.28         | 5.26         | −0.02         |

Diagnóstico: **2019 e 2021 são os mais sensíveis ao tamanho da
amostra**. O baseline (2019) muda 8% só por dobrar a amostra. Toda
afirmação que o use como referência herda essa instabilidade.

**Consequência direta na Seção 11 do notebook (decomposição):** com
ADE_2019 = 4.92, anos como 2022 (4.29) e 2024 (4.72) ficam *abaixo* do
baseline. A decomposição passa a reportar `excess_concept_mm` e
`excess_covariate_mm` negativos, que **não têm interpretação útil** no
arcabouço covariate/concept. Os mesmos cálculos com ADE_2019 = 4.53
mudariam o sinal de várias linhas.

### IW com `recovery_pct` fora de [0, 100]

Olhando `iw_decomposition.csv`:

| Ano  | ADE_y | ADE_2019 | ADE_IW | recovery_pct | Leitura          |
|------|-------|----------|--------|--------------|------------------|
| 2021 | 6.82  | 4.92     | 5.49   | **69.7%**    | razoável          |
| 2022 | 4.29  | 4.92     | 4.23   | **−9.4%**    | sem sentido (ADE_y < baseline) |
| 2023 | 5.21  | 4.92     | 4.79   | **145.6%**   | overshoot          |
| 2024 | 4.72  | 4.92     | 4.58   | **−69.2%**   | sem sentido        |
| 2025 | 5.28  | 4.92     | 4.91   | **101.5%**   | overshoot marginal |

Três problemas distintos:

1. **Anos com ADE_y < ADE_2019** (2022, 2024) não cabem no modelo
   conceitual do IW. A fórmula `(ade_y − ade_iw) / (ade_y − ade_2019)`
   tem denominador negativo nesses casos.
2. **Recovery >100%** (2023 com 146%, 2025 com 102%) significa que o
   modelo *ponderado por inputs ano-y* fica MELHOR que o ADE original
   de 2019. Isso só faz sentido se o subconjunto reponderado seja
   sistematicamente "mais fácil" que 2019 — sintoma da subamostragem
   discutida acima.
3. **2021 é o único ano que rende uma afirmação publicável**: 70% do
   excesso é explicado por covariate shift. Mas com IC bootstrap (não
   reportado em `iw_decomposition.csv`) provavelmente abre largo.

### Retreino: net positivo pequeno, dano colateral grande

`retrain_results.csv` diz:
- Apenas **um** breakpoint detectado (`antes_2022`).
- Treino: 60 mil trajetórias, teste: 120 mil.
- `ade_before = 4.92`, `ade_after = 4.67` → ganho de 4.9% no agregado.
- **`catastrophic_ratio = 0.832`** — quase metade das trajetórias
  pioraram (54.498 trajetórias degraded vs 65.502 improved).

Diagnóstico: o fine-tuning está *redistribuindo* o erro, não
*reduzindo*. A média cai mas a variância cresce. Em produção isso é
ruim — preferimos um modelo com erro consistente sobre um que melhora 1
mm em metade dos casos e piora 0.8 mm na outra metade.

### Cobertura por divisão insuficiente

`by_division_decomposition.csv`:
- 2021 está como `division="Unknown"` (Codex marcou "Both" no map e o
  script não mapeou para A nem B).
- Division A só tem 3 jogos por ano (corte de `n_per_year=3`).
- Division B **não aparece em momento nenhum** — provavelmente é toda a
  população "Unknown" ou nenhum dos jogos amostrados é B.

Resultado: a única conclusão que sobrevive (`delta_A = −0.05` antes/após
2022) está sustentada em 6 jogos no total — frugal demais para pôr em
paper.

### Resultado mais discreto: nenhuma latência real foi medida

A Seção 10.2 do notebook (curva de latência) tem o código pronto, mas
os números atuais saem do *stub* (sandbox sem `river`/`ruptures` ao
montar o notebook). Quando você rodou local, a Seção 10 funcionou, mas
o `latency_curve.csv` ainda é placeholder se o notebook não foi
re-executado em sequência. Vale conferir.

### Pendência operacional

Há ainda dois scripts apontando `STATS_PATH` para a raiz em vez de
`model/`:
- `drift_analise/compute_importance_weights.py:54`
- `model_analise/retrain_at_breakpoints.py:47`

Se rodaram com sucesso, é porque há cópia da `.pkl` na raiz ainda —
verificar e deduplicar.

---

## O que fazer antes do Mês 4

### Bloco A — consertos não-negociáveis (pré-publicação)

1. **Re-rodar TUDO do Mês 3 com `--n_per_year 6`.** O resultado de n=3
   já está medido como instável; manter n=3 nas conclusões finais é
   metodologicamente fraco. Concretamente:
   ```bash
   python model_analise/compute_trajectory_errors.py --n_per_year 6 --seed 42
   python drift_analise/compute_importance_weights.py
   python model_analise/retrain_at_breakpoints.py
   python drift_analise/division_validation.py
   ```
   E re-executar o notebook.

2. **Decidir o que fazer com baseline 2019.** Opções:
   - (a) Usar **todos os jogos de 2019** disponíveis para o baseline
     (não amostrar). Hoje são 6 proc_sets disponíveis (proc_set_3..8);
     usar todos. Reportar `n_baseline = 6` em vez de 3.
   - (b) Manter amostragem mas com `n_per_year >= 6` para todos os anos
     e reportar IC bootstrap em todas as decomposições.

   Recomendo (a): o baseline é uma única referência, não precisa ser
   amostrada.

3. **Mapear os jogos de 2021** em `division_map.csv`. Hoje saem como
   `Unknown` e contaminam a Frente 4. Se forem mesmo "Both" (Division A
   e B coexistindo no torneio de 2021), declarar e excluir 2021 da
   análise por divisão — não silenciar como `Unknown`.

4. **Tornar a decomposição IW robusta a `ADE_y < ADE_2019`.** O reporte
   atual de `recovery_pct = −9%` é ruído cosmético. Substituir por:
   - Caso `ADE_y > ADE_2019`: reportar `recovery_pct` normal, *clipado
     em [0, 100]* (com flag `overshoot=True` se passar de 100).
   - Caso `ADE_y ≤ ADE_2019`: reportar `excess_total = 0.0`,
     `recovery_pct = NaN`, e `note = "no excess to recover"`. Não tentar
     interpretar.

5. **Adicionar IC bootstrap em todas as % da Tabela 11**. O plano
   original previa, mas o CSV atual não tem. Sem CI, nenhuma fração
   acima é defensável.

### Bloco B — análises adicionais que ficaram abertas (escolher 2-3)

6. **Decompor IW por feature.** Hoje o vetor de features do LSIF tem 8
   dimensões (speed/accel/turn × mean/p90/p99). Saber *qual* dessas
   dimensões está dirigindo o weighting reforça interpretação. Repetir
   IW marginal por feature.

7. **Latência real medida.** Confirmar que `latency_curve.csv` foi
   gerado com a execução real do `river` (não stub). Adicionar comparação
   ADWIN vs Page-Hinkley em uma única figura legível.

8. **Retreino com hiperparâmetros diferentes.** `recovery_pct = −4.9%` e
   `catastrophic_ratio = 0.83` é evidência de que **fine-tuning de uma
   camada não basta**. Antes de declarar "retreino leve não resolve",
   testar:
   - Descongelar últimas 2 camadas
   - lr ainda menor (1e-5) com mais épocas (20)
   - Early stopping por catastrophic_ratio (parar quando passar de 0.5)
   Se mesmo assim não fechar, a conclusão "leve não resolve" fica
   honesta.

9. **Bootstrap de breakpoints do Pelt.** Hoje detectou 1 breakpoint só
   (`antes_2022`). Rodar Pelt em B amostras bootstrap dos jogos e ver
   quais breakpoints são estáveis. Se o breakpoint `antes_2022` aparece
   em >80% dos bootstraps, é robusto; senão é coincidência.

### Bloco C — só faça depois dos blocos A e B

10. **Análise da hipótese "tudo é divisão"**. Se `delta_A` continuar
    perto de zero com n=6 e `delta_B` for grande (quando houver
    cobertura), o paper vira: *"o drift reportado por estudos anteriores
    em SSL é majoritariamente confundimento de divisão, não evolução
    temporal."* É uma tese forte, contraintuitiva, e bem alinhada com
    metodologia de IC.

11. **Documentação e escrita do Mês 4.** Aí sim, com os números
    corrigidos, atualizar `checkpoint.tex` e começar o relatório final.

---

## Sugestão de escopo do Mês 4

Pensando no que faz sentido depois desse review:

**Cenário 1 — paper de "drift confundido com divisão"**
Foco: solidificar a descoberta da Frente 4 e construir o argumento de
que o drift agregado é parcialmente artefato de mistura A/B. Demanda:
mais cobertura por divisão, possivelmente baixar para Division B (se
houver pkls disponíveis) e contraste explícito.

**Cenário 2 — paper de "detecção formal funciona, adaptação leve não"**
Foco: usar o que o Mês 2 entregou (detectores rodam, têm sinal acima do
null, latência mensurável) e contrastar com Mês 3 (IW e fine-tuning
recuperam pouco do erro). Mensagem: "detectar é fácil, adaptar não".
Demanda: melhor calibração das %, IC em tudo, e pelo menos uma rodada
adicional de fine-tuning honesto.

**Cenário 3 — escrita pura, sem novos experimentos**
Se a janela de tempo é apertada, aceitar os resultados como estão e
focar em redação. Mas aí o paper terá várias passagens "isto é uma
limitação", "isto não é estatisticamente significativo", etc. Funciona
para IC, é fraco para publicação externa.

**Recomendo Cenário 1 ou 2** dependendo de qual descoberta sobreviver
ao re-cálculo do Bloco A. Se com n=6 a Division A continuar plana, vai
para Cenário 1. Se a Division A também mostrar drift leve mas
significativo, Cenário 2 é mais defensável.

---

## Lista checada

| Item                                                      | Estado |
|----------------------------------------------------------|--------|
| Sample size sensitivity rodado e analisado               | OK     |
| LSIF/IW rodando com ESS estável                          | OK     |
| Retreino seletivo rodado (1 breakpoint)                  | OK     |
| Corte por divisão rodado                                 | OK (cobertura baixa) |
| Re-rodar com n=6                                         | **TODO** |
| Baseline 2019 = todos os jogos                           | **TODO** |
| Mapear 2021 em division_map.csv                          | **TODO** |
| IC bootstrap em todas as % da decomposição               | **TODO** |
| Tratar `recovery_pct` fora de [0,100]                    | **TODO** |
| Latência real (não stub) confirmada                      | **TODO** |
| Decidir cenário do Mês 4                                 | **TODO** |
| Atualizar checkpoint.tex com Mês 3                       | aguarda Bloco A |
