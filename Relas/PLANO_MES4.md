# Plano de Mês 4 — Consolidação e Redação

> Estado na entrada: Mês 1, 2 e 3 implementados. `REVIEW_MES3.md`
> identificou três problemas críticos (subamostragem do baseline 2019,
> `recovery_pct` fora de [0,100], cobertura por divisão insuficiente).
> O **Bloco A** do review já foi codificado (Mês 4 abre com a re-execução
> dos scripts) e parte do **Bloco B** também (IW por feature, retreino
> expandido).

---

## Objetivo do Mês 4

Sair do modo *exploratório* (Mês 1-3 montou e exercitou o pipeline) e
entrar no modo *publicação*. O Mês 4 não introduz métodos novos; ele
**fecha resultados, escolhe a tese central, e produz o relatório
final**.

Três entregáveis concretos:

1. **Resultados re-rodados sem os vícios do Mês 3** (Bloco A do review).
2. **Tese central decidida** (Cenário 1, 2 ou 3 do review) com base nos
   números corrigidos.
3. **Documento final**: `Relas/main_pt.tex` atualizado, abstract para
   workshop/relatório de IC, e o `checkpoint.tex` com seção de Mês 3 e
   conclusões.

---

## Frentes de trabalho

### Frente 1 — Re-execução com correções (~2 dias)

**Por quê:** os scripts já estão atualizados (Bloco A do review). Falta
rodar e analisar. Ordem importa: erro → IW → retreino.

```bash
# 1. Erros por trajetória — agora com baseline 2019 completo + n=6 nos demais
python model_analise/compute_trajectory_errors.py --n_per_year 6 --seed 42

# 2. Importance weighting com recovery_pct robusto + IW por feature
python drift_analise/compute_importance_weights.py --per_feature

# 3. Retreino expandido — 2 camadas descongeladas, lr=1e-5, 20 epochs,
#    early-stop por catastrophic_ratio
python model_analise/retrain_at_breakpoints.py

# 4. Validação por divisão (com 2021 mapeado como Both)
python drift_analise/division_validation.py

# 5. Notebook end-to-end
jupyter nbconvert --to notebook --execute drift_analise/drift_analise.ipynb \
    --output drift_analise.executed.ipynb \
    --output-dir drift_analise \
    --ExecutePreprocessor.timeout=600
```

**Critério de aceite:**

- `iw_decomposition.csv` tem coluna `recovery_pct ∈ [0,100]` ou `NaN`,
  com flag `overshoot` documentada
- `iw_per_feature.csv` existe e ranqueia features por `recovery_pct`
- `retrain_results.csv` tem `n_unfreeze=2` e `epochs_run` (early-stop ou
  budget total)
- `retrain_epoch_log_bp{idx}.csv` existe e tem `cat_ratio` por época
- `by_division_decomposition.csv` tem coluna para 2021 com
  `division="Both"` (não mais `"Unknown"`)

**Plano B se algo quebrar:**

- Se `n_per_year=6` estourar memória/tempo: baixar para 4. Reportar.
- Se `--per_feature` der ESS instável em todas as 8 dimensões: usar
  apenas as 3 dimensões `*_p99` (caudas) e deixar nota.

### Frente 2 — Análise dos resultados corrigidos (~1 dia)

Reler `REVIEW_MES3.md` lado a lado com os novos CSVs. Atualizar a tabela
"O que sobrevive ao escrutínio".

**Perguntas a responder explicitamente, com número:**

1. Com baseline 2019 completo (n_baseline = 6 ou mais), quanto é o
   **excess médio** de cada ano? Quais anos têm `excess > 0` agora?
2. Para os anos com excess positivo, qual o **recovery_pct** com IC95?
   Quantos % do drift são covariate?
3. **Qual feature do per-feature LSIF** explica mais do drift?
   Speed_p99? Accel_p90? Esse ranking estabiliza entre anos?
4. Retreino com 2 camadas + lr menor: o `catastrophic_ratio` baixou?
   `recovery_pct` melhorou?
5. **Division A continua plana?** Com 2021 incluído como "Both", o
   delta antes/após-2022 muda?

### Frente 3 — Decisão da tese central (~meio dia)

A escolha do cenário (1, 2 ou 3 do `REVIEW_MES3.md`) deve ser
**determinada pelos números**, não pela conveniência. Use esta árvore
de decisão:

```
delta_A < 0.3 mm e delta_B (ou Both) > 1.5 mm
    -> Cenário 1: "drift é confundimento de divisão"
delta_A > 0.5 mm e recovery_pct mediano < 50%
    -> Cenário 2: "detectar é fácil, adaptar não"
nenhum dos acima decisivo
    -> Cenário 3: redação descritiva sem afirmação forte
```

**Documente a decisão** em commit message ou nota no `checkpoint.tex`.
Se Cenário 2, o "fine-tuning não basta" tem que estar suportado pelos
**três** experimentos: leve (1 camada), médio (2 camadas), com early-stop.

### Frente 4 — Atualização do checkpoint.tex (~1 dia)

Adicionar duas seções novas ao `checkpoint.tex`:

**Seção "Mês 3 — Adaptação ao drift"** (dois subseções):

- *3.1 Importance weighting (LSIF)* — método, fórmula da decomposição
  IW, ESS ratio reportado, tabela `iw_decomposition.csv` resumida,
  decomposição por feature.
- *3.2 Retreino seletivo* — pipeline (Pelt → split → fine-tuning),
  `retrain_results.csv`, discussão de catastrophic interference.

**Seção "Mês 3 — Validação por divisão"** — apresentação do
`by_division_*.csv` e a leitura de tese: o drift entre anos sobrevive
quando se controla por divisão?

**Seção "Análise crítica e tese central"** — explicita o cenário
escolhido na Frente 3 e o suporte numérico.

**Atualizar:**

- Resumo (1º parágrafo) — incluir Mês 3 e Mês 4.
- Linha do tempo — entrada nova: "Mês 4 — consolidação e redação".
- Bibliografia — adicionar Sugiyama 2008 e 2012, Truong 2020 (já estão
  como `\bibitem` mas verificar `\cite`s).

### Frente 5 — Relatório final / paper (~3 dias)

Decidir o formato:

- **IC formal**: relatório de iniciação científica em `Relas/main_pt.tex`
  no formato exigido pela universidade. Ver `main_en.tex` se for
  internacional.
- **Workshop/short paper**: 6-8 páginas, ICRA workshop, RoboCup symposium
  (CBR), abstract ICRA SSL.

Se for IC, **trabalhar diretamente no `main_pt.tex`** (já existe
esqueleto). Se for paper, criar `Relas/paper_robocup.tex` separado.

**Estrutura sugerida do paper (qualquer formato):**

1. Introdução — drift em SSL, gap de literatura.
2. Setup — modelo Seq2Seq, dataset, granularidades introduzidas no Mês 1.
3. Detecção formal — Mês 2 (ADWIN, PH, KSWIN, Pelt + null + latência).
4. Adaptação — Mês 3 (IW e retreino).
5. Tese central — Cenário escolhido.
6. Limitações e trabalho futuro.

**Aproveitar TODAS as figuras já produzidas em `Relas/results/`**.
Cada PDF (PT/EN) é um candidato à paper figure.

### Frente 6 — Reprodutibilidade e housekeeping (~meio dia)

Antes de submeter / arquivar:

1. `git status` deve ficar limpo. Commit dos resultados regerados.
2. `tests/test_drift_metrics.py` rodando (`pytest -q tests/`).
3. `requirements.txt` reflete o que foi efetivamente usado (ver se
   `densratio` foi adicionado — se não foi, o LSIF é manual e tudo OK).
4. README.md no topo do repo: parágrafo curto explicando o pipeline e
   apontando para `Relas/PIPELINE.md`.
5. Tag git: `git tag -a v1.0-ic -m "Iniciacao Cientifica - submissao"`.

---

## Riscos e contingências

| Risco                                              | Sintoma                          | Plano B                              |
|----------------------------------------------------|----------------------------------|--------------------------------------|
| Re-rodar com n=6 não muda muito o quadro           | excess de 2025 ainda baixo       | Cenário 1 fica improvável; ir para 2 |
| `--per_feature` instável                           | ESS variável entre features      | Reportar só features estáveis        |
| Retreino expandido ainda dá `cat_ratio > 0.5`      | early-stop em 2-3 épocas         | Conclusão fica "fine-tuning não basta" — Cenário 2 |
| Division B sem cobertura (todos `Both` ou `A`)     | só A e Both no CSV               | Reportar como limitação no Cenário 1 |
| Prazo apertado para escrita                        | < 1 semana antes da entrega      | Cenário 3: descritivo + roadmap      |
| Notebook quebra com novos paths                    | `KeyError`/`FileNotFoundError`   | Recompilar com `build_notebook_v2.py` |

---

## Cronograma sugerido (~7-8 dias úteis)

```
Dia 1:    Frente 1 — re-execução
Dia 2:    Frente 1 (validação) + Frente 2 (análise)
Dia 3:    Frente 3 (decisão de tese) + Frente 4 (checkpoint.tex)
Dia 4-6:  Frente 5 (escrita do relatório/paper)
Dia 7:    Frente 6 (housekeeping) + Frente 5 (revisão)
Dia 8:    Buffer / submissão
```

Adapta para:
- Se decidiu Cenário 1 cedo, Frente 5 fica mais curta (tese clara).
- Se Cenário 2, Frente 4 demora mais (mais experimentos para
  documentar).
- Se Cenário 3, Frente 5 vira "documentar limitações" e ganha mais
  tempo.

---

## Critérios de fechamento do Mês 4

A IC está pronta para entrega quando:

- [ ] Os 4 scripts do Mês 3 rodaram com as correções e os CSVs novos
      estão em `Relas/results/mes3/`
- [ ] `iw_decomposition.csv` tem `recovery_pct` clipado e CI95
- [ ] `iw_per_feature.csv` ranqueia as 8 features
- [ ] `retrain_results.csv` reflete `n_unfreeze=2` e `epochs_run`
- [ ] `by_division_decomposition.csv` cobre 2021 (Both)
- [ ] `checkpoint.tex` compila e tem 3 seções novas (3.1, 3.2,
      validação por divisão, tese central)
- [ ] Notebook re-executado sem erros
- [ ] `git status` limpo, tag aplicada
- [ ] Tese central definida com suporte numérico explícito
- [ ] Relatório final / paper em `Relas/main_pt.tex` (ou `paper_*.tex`)
      revisado

---

## O que NÃO fazer no Mês 4

- Não introduzir métodos novos. Toda a análise está calibrada para os
  detectores e métodos do Mês 1-3.
- Não retreinar do zero. Se o `cat_ratio` ainda for alto com 2 camadas,
  documentar e seguir.
- Não rodar n_per_year > 6 só para "ver mais dados". Cada incremento
  exige re-rodar tudo; é caro e ganho marginal.
- Não tentar publicar antes de fechar o cenário. Paper sem tese clara
  não convence revisor.
- Não atualizar o `checkpoint.tex` antes da Frente 1 estar 100%
  rodada. Os números têm que ser os finais.

---

## Antecipando o pós-Mês 4

Coisas que ficam fora do escopo da IC mas são naturais para mestrado /
trabalho de continuação:

- **Retreino do encoder** (não só decoder).
- **Online retraining** com gatilho dos detectores do Mês 2 — fazer o
  loop fechar.
- **Multi-task com features de divisão** — incluir `division` como
  input do modelo.
- **Importance weighting via redes neurais** — substituir LSIF por
  classifier 2-sample (DNN density-ratio).
- **Estender para outros tipos de drift** — drift em features de
  robôs específicos, drift em estratégia de jogo (não só cinemática).
- **Validação cross-team** — treinar em N times, testar em time excluído
  do treino.

Esses pontos podem virar a seção "Trabalho futuro" do paper.
