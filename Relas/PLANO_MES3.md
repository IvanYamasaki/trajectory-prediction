# Plano de Mês 3 — Adaptação ao Data Drift

> Status na entrada: Mês 1 (granularidade fina) e Mês 2 (detectores formais)
> entregues e validados com dados reais. O notebook `drift_analise.ipynb`
> tem 55 células e produz 39 figuras + 12 CSVs. O modelo Seq2Seq está
> configurado para `target="dv"` e os pesos `robot_*_t.weights.h5` são
> usados sem modificação.

---

## Objetivo do Mês 3

Sair do modo *diagnóstico* (Mês 1 e 2 mediram **que** há drift) e entrar
no modo *adaptação* (Mês 3 mede **o que pode ser feito** sobre ele).
O escopo é responder, com números defensáveis em banca, três perguntas:

1. **Quanto do erro adicional pós-2019 é covariate shift?**
   Operacionalmente: o ADE em ano $y$ ponderado por
   $w(x) = p_{2019}(x) / p_y(x)$ se aproxima do ADE de 2019?
   Se sim, o problema é mostly de distribuição de input.
2. **Quanto do erro adicional é concept drift recuperável por retreino
   leve?** Operacionalmente: fine-tuning da última camada do Seq2Seq
   nos breakpoints do Pelt recupera quantos % do excesso?
3. **Drift "entre anos" sobrevive a corte por divisão?** A diferença
   entre Division A (6+6 robôs) e Division B (introduzida em 2018) pode
   estar inflando a aparência de drift temporal. Cortar por divisão
   antes de comparar anos.

---

## Frentes de trabalho

### Frente 1 — Validação dos resultados do Mês 2 (~2 dias)

**Por quê:** as Seções 10 e 11 do notebook foram desenhadas com amostra de
3 jogos por ano. Isso é frugal. Antes de adaptar, conferir robustez.

**Tarefas:**

- Rodar `compute_trajectory_errors.py --n_per_year 6 --seed 42` para
  duplicar a amostra. Se o disco aguentar, ir até 10/ano.
- Re-executar o notebook e comparar:
  - `drift_decomposition.csv` antes (n=3) e depois (n=6). Se o
    `pct_explained_by_concept` muda mais de 15 pontos, há instabilidade
    amostral — relatar e seguir com o n maior.
  - `null_distribution.csv`: o p-value dos detectores deve ficar mais
    estável com mais amostras. Se ficar erráticio, o detector está
    captando ruído.
- Adicionar bootstrap CI95 nas % da Seção 11. Padrão do projeto:
  `n_boot=2000, seed=42`.

**Critério de aceite:**
- Tabela `Relas/results/mes3/sample_size_sensitivity.csv` com
  `(n_per_year, year, pct_concept, ci_lo, ci_hi)`.
- Decisão registrada: amostra final do Mês 3 (qual `n_per_year` usar daqui
  pra frente).

---

### Frente 2 — Importance weighting (KLIEP) (~4 dias)

**Por quê:** importance weighting é o teste cego para covariate shift. Se
$\text{ADE}^{\text{IW}}_y \approx \text{ADE}_{2019}$, então **todo** o
drift de erro vem do input ter mudado, não do modelo. Em produção isso é
infinitamente mais barato que retreinar.

**Implementação** (criar `compute_importance_weights.py`):

1. **Espaço de features para o weighting.** Vetor por trajetória:
   `[speed_mean, speed_p90, speed_p99, accel_mean, accel_p90, accel_p99,
   turn_mean, turn_p90]`. Usar log + clipping em p99.5 do baseline para
   estabilizar.
2. **Estimador de razão de densidade.** Começar com KLIEP via biblioteca
   `densratio` (`pip install densratio`). Backup: implementação manual
   com kernel gaussiano e validação cruzada para o bandwidth.
3. **Diagnóstico do weight.** Computar effective sample size
   $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$.
   Se $\text{ESS}/N < 0.3$, os pesos são instáveis: aumentar
   regularização ou trocar para RuLSIF.
4. **ADE ponderado.** Para cada ano $y$:
   $$\text{ADE}^{\text{IW}}_y = \frac{\sum_{i \in \text{2019}} w_i \cdot \text{ade}_i}{\sum_i w_i}$$
   onde $w_i = p_y(x_i) / p_{2019}(x_i)$. Compara-se com
   $\text{ADE}_y$ direto: se forem próximos, drift é covariate.

**Saídas:**

- `covariate_shift_out/importance_weights.parquet` (uma linha por
  trajetória de 2019, com `w_2019_to_y` para cada $y \in \{2021, 2022,
  2023, 2024, 2025\}$).
- Seção nova no notebook (12 ou 13): `Relas/results/mes3/iw_*.pdf`
  e `iw_decomposition.csv` com colunas:
  `year, ade_y, ade_2019, ade_iw, ess_ratio, recovery_pct`.

**Critério de aceite:**
- ESS ratio reportado para cada par de anos. Anos com ESS < 0.3 marcados
  como "weighting não confiável".
- Plot comparativo: ADE direto vs ADE-IW por ano, com IC95 bootstrap.

---

### Frente 3 — Retreino seletivo nos breakpoints (~6 dias)

**Por quê:** mesmo que weighting recupere parte, é provável que sobre
concept drift residual. Retreinar do zero é caro (horas em GPU); o
compromisso é fine-tuning leve nos pontos onde o regime muda — que são
exatamente os **breakpoints do Pelt** (Seção 8).

**Implementação** (criar `retrain_at_breakpoints.py`):

1. **Ler breakpoints** de `alarms_consolidated.csv` filtrando
   `method=Pelt_*`. Para cada breakpoint $t^*$:
   - Treino = todos os jogos cronologicamente antes de $t^*$
   - Teste = todos os jogos depois
2. **Fine-tuning leve.** Carregar `robot_30_15_t.weights.h5`,
   **congelar encoder**, descongelar apenas a `Dense` final do decoder.
   Treinar por **5–10 épocas** com `lr=1e-4` (1/10 do treino original).
   Salvar como `robot_30_15_t_finetuned_bp{idx}.weights.h5`.
3. **Avaliar.** Computar ADE no conjunto de teste antes e depois do
   fine-tuning. Reportar:
   - `ade_before, ade_after, delta_mm, delta_pct`
   - Per-trajetória: histograma do delta (ADE_after - ADE_before).
     Idealmente concentrado em torno de zero ou negativo (melhora). Se
     houver cauda direita pesada, há trajetórias que **pioraram**
     (catastrophic interference) — sinal de alerta.
4. **Comparação com baseline.** Para cada breakpoint, também rodar uma
   "estratégia ingênua": treinar com **todos** os dados pré-$t^*$ no
   modelo original e ver se ele já estava overfitado a 2019 ou não.

**Saídas:**

- `Relas/results/mes3/retrain_results.csv` com colunas
  `breakpoint_year, breakpoint_idx, n_train, n_test, ade_before,
  ade_after, recovery_pct, n_traj_improved, n_traj_degraded`.
- Plot: ADE pré vs pós para cada breakpoint, com IC bootstrap.
- Histograma de delta_per_trajectory para o breakpoint mais informativo.

**Critério de aceite:**
- Pelo menos um breakpoint com `recovery_pct > 20%` e
  `n_traj_degraded / n_traj_improved < 0.3`.
- Se nenhum critério é atingido em todos os breakpoints, o relatório
  conclui honestamente que **fine-tuning leve não resolve** e o problema
  exige retreino mais profundo (escopo de Mês 4).

---

### Frente 4 — Validação por divisão (~2 dias)

**Por quê:** Division A e B coexistem na liga desde 2018. Cada uma tem
número de robôs e dinâmica diferentes. Drift "entre anos" pode ser
parcialmente confundido com "mistura entre divisões" se a proporção A/B
mudou no período. `division_map.csv` já existe — só falta usar.

**Tarefas:**

1. Cortar `dataset_enriched.csv` por divisão e refazer:
   - Plot 4a (ADE por jogo) com facet por divisão
   - Tabela de decomposição (Seção 11) separada por A vs B
2. Verificar se há "drift dentro de uma divisão fixa". Se sim, a
   conclusão de drift é robusta. Se o drift desaparece quando se fixa a
   divisão, a maior parte do efeito é confundimento.
3. Se ano $y$ tem $\leq 2$ jogos numa divisão, marcar como "amostra
   insuficiente" e excluir do plot — não falsear cobertura.

**Saídas:**

- `Relas/results/mes3/by_division_*.pdf` (4 figuras: ADE por jogo,
  decomposição, ks_speed timeline, alarmes ADWIN — todas facetadas A/B).
- Linha extra na conclusão: "drift sobrevive ao corte por divisão? Sim/Não/Parcial".

**Critério de aceite:**
- Tabela 2x2 (ano antes/depois 2022, divisão A/B) com ADE médio em cada
  célula. Bem documentada na Seção 14 do notebook.

---

### Frente 5 — Síntese e atualização do checkpoint (~1 dia)

**Tarefas:**

- Adicionar Seções 13 (importance weighting), 14 (retreino seletivo) e
  15 (validação por divisão) ao notebook.
- Atualizar Seção 12 (discussão) com os números reais dessas seções.
- Atualizar `Relas/checkpoint.tex` com Mês 3 — manter mesmo formato
  (Linha do tempo + Mês 3 detalhado + Análise atualizada + Próximos
  passos para Mês 4 / fim da IC).
- Atualizar `Relas/PIPELINE.md` com os novos CSVs/figuras do Mês 3.

---

## Riscos e contingências

| Risco                                    | Sintoma                          | Plano B                                 |
|------------------------------------------|----------------------------------|-----------------------------------------|
| KLIEP não converge / ESS muito baixo     | `ess_ratio < 0.2` em vários anos | Trocar para RuLSIF; reduzir features    |
| Fine-tuning piora o modelo               | `delta_pct < 0` em todos os bps  | Reduzir lr, congelar mais camadas       |
| Não há jogos suficientes pós-bp          | `n_test < 5`                     | Fundir breakpoints próximos             |
| ESS alta mas IW não muda ADE             | `ade_iw ≈ ade_y`                 | Drift é dominantemente concept; ir direto p/ Frente 3 |
| Divisão B tem cobertura muito baixa      | `n_jogos_B < 5`                  | Reportar como limitação, não cortar     |

---

## Cronograma sugerido (15–18 dias úteis)

```
Semana 1:  Frente 1 (validação)  +  início Frente 2 (KLIEP setup)
Semana 2:  Frente 2 (KLIEP rodando)  +  início Frente 3 (retreino)
Semana 3:  Frente 3 (avaliação retreino)  +  Frente 4 (divisão)
Semana 4:  Frente 5 (síntese + checkpoint atualizado)
```

Adapta conforme o resultado da Frente 1: se a sensibilidade amostral for
grande, gastar mais tempo em estabilizar antes de seguir.

---

## Critérios de sucesso do Mês 3

Mês 3 está fechado quando o notebook responder, com número e IC, as três
perguntas iniciais. Em texto:

> "Em 2025, importance weighting recupera X% (CI95 [a, b]) do excesso de
> ADE vs 2019. Fine-tuning seletivo no breakpoint de 2022 (Pelt)
> recupera Y% adicional (CI95 [c, d]). O efeito sobrevive ao corte por
> divisão na Division A (n=Z jogos), mas é insuficiente em Division B
> (n=W). Conclusão: o regime atual exige (weighting | retreino |
> ambos)."

---

## O que NÃO fazer no Mês 3

- Treinar Seq2Seq do zero (escopo de mestrado, não de IC).
- Mudar arquitetura (encoder/decoder/attention).
- Adicionar features novas ao input do modelo (vai pra Mês 4 se houver).
- Refazer Mês 1 ou 2. Se algo lá precisa ser corrigido, abrir issue
  separada e seguir o cronograma.
- Inventar números ou interpolar resultados ausentes — se um breakpoint
  não rodou, reportar como "não disponível".

---

## Referências para citar no relatório final

- KLIEP / RuLSIF: Sugiyama et al. 2008; Yamada et al. 2013.
- Importance weighting + covariate shift: Sugiyama & Kawanabe 2012 (livro).
- Catastrophic forgetting em fine-tuning: Goodfellow et al. 2013.
- Continual learning para robotics: Lesort et al. 2020 survey.
