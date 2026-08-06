# Briefing para Análise dos Resultados — Fases 1 e 2 (Mês 9)

Este documento contextualiza um agente ou investigador externo para analisar os resultados
experimentais de deteção de drift numa stream de ADE (Average Displacement Error) de trajetórias
de robôs RoboCup SSL.

---

## 1. Contexto do Projeto

### O sistema
- **Tarefa**: previsão de trajetórias de robôs em partidas de RoboCup SSL (Small Size League).
- **Modelos**: dois modelos de previsão treinados em dados de 2019 — **Seq2Seq** e **Kalman**.
- **Horizonte de previsão**: 30 → 15 frames (redução durante o projeto).
- **Métrica de erro**: ADE (Average Displacement Error, em mm) por trajetória.

### O problema de drift
Os modelos foram treinados em 2019. Dados de partidas de 2021–2025 podem mostrar **drift de conceito** — mudanças na distribuição do erro que indicam que o modelo degradou. O objetivo é detetar esses momentos com **alta especificidade** (poucos falsos alarmes em 2019) e **boa sensibilidade** (detetar mudanças reais em 2021–2025).

### Estrutura da stream de dados
- **Ficheiro fonte**: `data/preprocessed/errors_*.parquet` (carregado via `load_errors()`)
- **Total**: 1 152 000 linhas × 10 colunas (4 modelos × 288 000 trajetórias cada)
- **Colunas-chave**: `model`, `year`, `match_id`, `traj_id`, `ade_traj`, `global_idx`
- **Ordenação**: `(year, match_id, traj_id)` — stream temporal contínua por modelo
- **Distribuição por ano**: 2019 = 48 000 trajs (baseline), 2021–2025 = 240 000 trajs (48 000/ano)

### Pipeline de deteção (código em `drift_analise/chapter02_deteccao_pipeline.py`)
- **ADWINLite**: z-test entre duas metades de uma janela deslizante (não o ADWIN exato do River).
  Parâmetros: `delta` (limiar z), `min_window`, `cooldown`, `max_window`.
- **PageHinkley**: calibrado via `from_baseline(b2019, K_thr, K_delta)` onde o limiar =
  `K_thr × MAD(valores 2019)`. Parâmetros: `K_thr`, `K_delta`, `min_instances`, `cooldown`.
- **`run_detector(detector, signal)`**: itera sobre a stream e devolve lista de índices de alarme.

---

## 2. Métricas Utilizadas

| Métrica | Fórmula | Direção |
|---------|---------|---------|
| `FAR_2019_per1k` | `1000 × n_alarms_2019 / n_2019_total` | **abaixo melhor** — mede falsos alarmes no período de treino |
| `n_alarms_post` | soma de alarmes em 2021–2025 | **acima melhor** — mede deteções reais |
| `SNR_op` | `(n_post × n_2019_total) / (n_2019_alarms × n_post_total)` | **acima melhor** — razão sinal/ruído operacional |

**SNR_op = 9999** quando `n_2019_alarms = 0` e `n_post > 0` (perfeito).
**SNR_op < 1** significa que a taxa de alarmes por trajetória é *maior* em 2019 do que em 2021–2025 — pior que aleatório.

**Objetivo original do plano**: atingir `FAR_2019_per1k ≤ 0.5/1k` com `SNR_op ≥ 2.0`.

---

## 3. Fase 1 — Ensaio de Suavização do Sinal

### Script e outputs
- Script: `drift_analise/phase1_smoothing.py`
- Resultados: `mes9_phase1/phase1_results.csv` (88 linhas: 22 exp × 2 modelos × 2 detetores)
- Relatório completo: `mes9_phase1/PHASE1_REPORT.md`
- Imagens (3 painéis por exp): `mes9_phase1/images/<exp>/<exp>_<modelo>.png`

### Design experimental
22 técnicas de suavização pré-processamento × 2 modelos (Seq2Seq, Kalman) × 2 detetores (ADWIN, PH) = 88 runs.
Técnicas: `ctrl` (sem suavização), SMA (w=50/100/200/500), Mediana (w=50/100/200), EWMA (α=0.01–0.30),
Holt (α/β combinações), Savitzky-Golay, Robust-z (w=200/500).

### Descobertas principais

**1. Suavização convencional (SMA/Mediana/EWMA) piora dramaticamente o ADWIN:**
- `ctrl`: FAR_2019 = 1.23/1k → `sma_w500`: FAR_2019 = 9.21/1k (↑↑↑)
- Causa: suavização introduz autocorrelação → viola a suposição i.i.d. do z-test do ADWINLite → dispara em diferenças mínimas entre as duas metades da janela.

**2. Única técnica que ajuda o ADWIN: Robust-z com janela w=200:**
- `robz_w200`: FAR_2019 = 0.60/1k (↓ 51% vs ctrl=1.23/1k)
- Mecanismo: normaliza localmente `(x - median_rolling) / MAD_rolling` → o ADWIN só vê variações relativas à distribuição local, não a tendência absoluta.

**3. PH melhora modestamente com Holt:**
- `holt_a005_b001`: FAR_2019 = 2.52/1k (↓ 49% vs ctrl=4.98/1k)

**4. Nenhuma técnica da Fase 1 atingiu FAR_2019 ≤ 0.5/1k para Seq2Seq.**

### Implicação estratégica (pivot para Fase 2)
- Suavização convencional descartada para ADWIN (confirmado ineficaz).
- Frente A da Fase 2: usar `robz_w200` como pré-processamento fixo + grid de hiperparâmetros ADWIN.
- Frente B: mudar paradigma — agregar N trajetórias em janelas → média por janela tem distribuição
  aproximadamente i.i.d. (Teorema do Limite Central) → usar PH sobre médias de janela.
- Frente C: introduzir KSWIN (Kolmogorov-Smirnov windowed) — teste distribucional que não assume i.i.d. linear.

---

## 4. Fase 2 — Grid Search de Hiperparâmetros

### Script e outputs
- Script: `drift_analise/phase2_grid_search.py`
- Resultados: `mes9_phase2/phase2_results.csv` (348 linhas: 174 configs × 2 modelos)
- Relatório: `mes9_phase2/PHASE2_REPORT.md`
- Imagens: `mes9_phase2/images/frente_A/`, `frente_B/`, `frente_C/`
  - Por frente: heatmap FAR_2019, heatmap SNR_op, scatter FAR×n_post, top-5 stream plots

### Grids testados

**Frente A — ADWIN + robz_w200 pré-fixo (96 configs)**
```
delta      = [1e-4, 1e-5, 1e-6, 1e-7]
min_window = [200, 400, 600, 1000]
cooldown   = [200, 500, 1000]
max_window = [2000, 5000]
```

**Frente B — PH com agregação por janela (60 configs)**
```
N_window = [50, 100, 200, 500]   # trajetórias por janela → TLC garante ≈ i.i.d.
K_thr    = [5, 10, 20, 30, 50]   # limiar = K_thr × MAD(2019)
K_delta  = [0.5, 1.0, 2.0]       # sensibilidade PH
```

**Frente C — KSWIN puro scipy (18 configs válidas)**
```
alpha       = [0.001, 1e-4, 1e-5]     # nível de significância KS
window_size = [200, 500, 1000]        # janela de referência
stat_size   = [50, 100]               # janela de teste (stat_size < window_size)
```
*Nota: KSWINDetector implementado no próprio script (sem dependência river) — checa a cada `stat_size` observações via `scipy.stats.ks_2samp`.*

### Resultados — Seq2Seq

#### Frente A (ADWIN + robz_w200)
| Config | FAR_2019 | n_2019 | n_2021 | n_2022 | n_2023 | n_2024 | n_2025 | n_post | SNR_op |
|--------|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| d=1e-7, mw=600, cd=200, xw=5000 | **0.021** | 1 | 8 | 1 | 3 | 0 | 0 | 12 | **2.40** |
| d=1e-7, mw=1000, cd=200, xw=5000 | 0.021 | 1 | 8 | 1 | 3 | 0 | 0 | 12 | 2.40 |
| d=1e-7, mw=600, cd=500, xw=5000 | 0.021 | 1 | 6 | 1 | 3 | 0 | 0 | 10 | 2.00 |
| d=1e-7, mw=600, cd=200, xw=2000 | 0.042 | 2 | 11 | 1 | 5 | 4 | 2 | 23 | 2.30 |

Mínimo FAR_2019: **0.021/1k**. Padrão: delta=1e-7 domina; max_window=5000 prefere.

#### Frente B (PH + janela)
| Config | FAR_2019 | n_2021 | n_2022 | n_2023 | n_2024 | n_2025 | n_post | SNR_op |
|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| N=50, K=50, δ=2.0 | **0.000** | 13 | 0 | 0 | 0 | 0 | 13 | 9999 |
| N=100, K=50, δ=2.0 | 0.000 | 11 | 0 | 0 | 0 | 0 | 11 | 9999 |
| N=200, K=30, δ=2.0 | 0.000 | 11 | 0 | 0 | 0 | 0 | 11 | 9999 |
| N=200, K=50, δ=1.0 | 0.000 | 10 | 0 | 0 | 0 | 0 | 10 | 9999 |
| N=500, K=20, δ=2.0 | 0.000 | 13 | 0 | 0 | 0 | 0 | 13 | 9999 |

Mínimo FAR_2019: **0.000/1k**. **Atenção**: todas as configs com FAR=0 detetam *apenas* em 2021.

#### Frente C (KSWIN)
| Config | FAR_2019 | n_post | SNR_op |
|--------|----------|--------|--------|
| α=1e-5, ws=1000, ss=100 | **0.438** | 81 | 0.77 |
| α=1e-5, ws=1000, ss=50 | 0.521 | 100 | 0.80 |
| α=1e-5, ws=500, ss=100 | 0.562 | 125 | 0.93 |

Mínimo FAR_2019: **0.438/1k**. SNR_op < 1 em todas as configs — taxa de alarme proporcional em 2019 > pós-2019.

### Resultados — Kalman

| Frente | Melhor FAR_2019 | Melhor config | n_post | SNR_op |
|--------|-----------------|---------------|--------|--------|
| A | 0.104/1k | d=1e-7, mw=400, cd=1000, xw=5000 | 45 | 1.80 |
| B | **0.000/1k** | N=500, K=20, δ=2.0 | 9 | 9999 |
| C | 0.583/1k | α=1e-5, ws=1000, ss=100 | 105 | 0.75 |

---

## 5. Questões Científicas em Aberto

As seguintes questões **devem ser respondidas pela análise das imagens e CSV** antes de avançar para a Fase 3 (ensemble):

### Q1 — A ausência de deteções em 2022–2025 na Frente B é real ou é limitação do detector?

**Hipótese A (drift real termina em 2021):** O salto 2019→2021 (gap COVID + novos comportamentos/hardware RoboCup) é a única mudança distribucional significativa. De 2022 em diante o comportamento estabilizou (relativo a 2021), pelo que os erros ADE pós-2021 não excedem o limiar K_thr=K×MAD(2019).

**Hipótese B (drift gradual não detetado):** Há drift em 2022–2025 mas a agregação por janela com K_thr alto é insensível a mudanças graduais. O ADWIN (Frente A) apanha 2022 e 2023 porque opera trajetória-a-trajetória.

**Como verificar:** Comparar as stream plots de `frente_A/` vs `frente_B/`. Se o sinal ADE raw mostra um patamar visualmente diferente em 2022–2025 vs 2021, a Hipótese B é mais provável.

### Q2 — O SNR_op < 1 do KSWIN indica inutilidade ou apenas calibração errada?

KSWIN com as configurações testadas ativa mais vezes por trajetória em 2019 do que no período pós. Isto pode indicar:
- O KSWIN deteta variabilidade interna de 2019 (variância alta dentro do ano de treino).
- As janelas testadas (ws=200–1000) são pequenas demais para o nível de ruído da stream.
- Aumentar ws ou stat_size poderia mudar o comportamento (não testado neste grid).

**Como verificar:** Analisar `frente_C/heatmap_FAR_2019_per1k_*.png` — ver se há cliff edge nos parâmetros ou se é monotonicamente crescente com janela.

### Q3 — Qual é o trade-off operacional aceitável entre FAR_2019 e n_post?

Frente A com `cd=200, xw=5000` tem FAR=0.021/1k mas n_post=12 (só 3 anos cobertos).
Frente A com `cd=200, xw=2000` tem FAR=0.042/1k mas n_post=23 (mais anos cobertos, 2021–2025).
A decisão depende da aplicação: notificar re-treino uma vez (drift 2021) ou monitorizar continuamente?

### Q4 — Para Fase 3, qual é o ensemble mais promissor?

Opções a avaliar:
- **AND(A, B)**: elimina o 1 alarme restante de 2019 em A → FAR=0, mas n_post reduz para interseção (provável só 2021).
- **OR(A, C)**: mantém cobertura multi-ano de KSWIN mas adiciona FAR de C → FAR ≈ 0.021 + 0.438 (antes de descontar redundância).
- **A standalone**: já quase perfeito (FAR=0.021, SNR=2.40) — questionar se ensemble adiciona valor.
- **AND(A, C) com janela temporal**: alarme só se A e C concordam num intervalo de 200 trajs → elimina FAR residual de ambos.

---

## 6. Ficheiros de Referência

```
Relas/results/drift/
├── mes9_phase1/
│   ├── PHASE1_REPORT.md          # relatório completo fase 1
│   ├── phase1_results.csv        # 88 linhas (22 exp × 2 modelos × 2 detetores)
│   └── images/<exp>/             # 3 painéis: sinal | raster ADWIN | raster PH
│
├── mes9_phase2/
│   ├── PHASE2_REPORT.md          # relatório completo fase 2
│   ├── phase2_results.csv        # 348 linhas (174 configs × 2 modelos)
│   └── images/
│       ├── frente_A/             # heatmaps + scatter + top-5 stream plots (ADWIN)
│       ├── frente_B/             # heatmaps + scatter + top-5 stream plots (PH)
│       └── frente_C/             # heatmaps + scatter + top-5 stream plots (KSWIN)
│
└── CONTEXT_ANALISE_AGENTE.md     # este documento

drift_analise/
├── chapter02_deteccao_pipeline.py  # ADWINLite, PageHinkley, run_detector, load_errors, build_stream
├── phase1_smoothing.py             # script fase 1
└── phase2_grid_search.py           # script fase 2 (inclui KSWINDetector, aggregate_stream)
```

### Colunas do CSV phase2_results.csv
`frente, model, detector, exp, [params específicos da frente], pre_proc, n_2019_total, n_post_total, n_alarms_total, FAR_2019_per1k, n_alarms_post, SNR_op, n_2019, n_2021, n_2022, n_2023, n_2024, n_2025`

---

## 7. Tarefa para o Agente Analista

1. **Ler `mes9_phase2/phase2_results.csv`** e calcular estatísticas descritivas por frente e modelo.
2. **Analisar as imagens** de stream (top-5 configs) para cada frente — verificar visualmente se os alarmes coincidem com transições de ano.
3. **Responder às Q1–Q4** acima com base nos dados.
4. **Propor os candidatos finais para Fase 3**, preenchendo a tabela em `PHASE2_REPORT.md` (secção "Candidatos para Fase 3").
5. **Decidir se KSWIN deve entrar no ensemble** ou se A+B é suficiente.
6. **Redigir uma recomendação de Fase 3** (estratégia de ensemble, janela de concordância, critério de alarme final).
