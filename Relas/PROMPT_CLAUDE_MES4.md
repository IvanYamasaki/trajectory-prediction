# Prompt para o Claude no VS Code executar — Mês 4

> Cole o conteúdo abaixo da linha `===` no chat do Claude do VS Code.
> Cada bloco é uma tarefa; ele deve seguir em ordem e parar pra confirmar
> antes de mudar de seção principal. Os arquivos referenciados existem.

```
===

Você está no projeto `trajectory-prediction` (IC sobre data drift em Seq2Seq
de predição de trajetórias no RoboCup SSL). Os Mês 1, 2 e 3 já foram
executados. Agora você vai conduzir o Mês 4: consolidação, decisão de tese
central, e redação do relatório final.

LEIA PRIMEIRO (em ordem):
1. `Relas/PLANO_MES4.md` — plano oficial. Você está executando as Frentes
   2 a 6.
2. `Relas/REVIEW_MES3.md` — análise crítica que motivou os ajustes do Bloco
   A. Use como contexto para entender o que cada CSV significa.
3. `Relas/checkpoint.tex` — relatório de checkpoint atual (cobre Mês 1 e 2).
   Você vai estendê-lo nas Frentes 4 e 5.
4. `Relas/results/mes3/*.csv` — resultados re-rodados pós-Bloco A. São as
   evidências numéricas para tudo no Mês 4.

REGRAS GERAIS
- Não retreine modelo. Não introduza método novo.
- Não modifique `dataset.csv` nem `dataset/proc_set_*.pkl`.
- Toda figura nova vai em `Relas/results/mes4/` em PT e EN.
- Toda afirmação numérica precisa de IC bootstrap (n_boot=2000, seed=42).
- Reporte cada tarefa concluída com [OK]/[FAIL] e os arquivos modificados.

ESTADO ATUAL (já feito antes de você começar):
- Frente 1 (re-execução com correções) está completa. Os CSVs em
  `Relas/results/mes3/` refletem: baseline 2019 com 6 proc_sets,
  recovery_pct clipado, IC bootstrap, IW por feature, retreino com
  n_unfreeze=2 e lr=1e-5.
- Os números-chave: IW recupera ~60-72% do excess em anos com excess>0;
  retreino leve recupera 33% mas com catastrophic_ratio=0.82.

═════════════════════════════════════════════════════════════════
TAREFA 1 — Frente 2 do PLANO_MES4: análise dos resultados corrigidos
═════════════════════════════════════════════════════════════════

Crie `Relas/ANALISE_MES4.md` (markdown estruturado). Para cada
pergunta abaixo, responda em <=2 parágrafos com NÚMERO + CSV de
origem. Não invente, leia os CSVs.

Q1. Com baseline 2019 completo, qual o excess médio por ano? Quais
    anos têm excess > 0? (Fonte: `iw_decomposition.csv`)
Q2. Para os anos com excess positivo, qual o recovery_pct e CI95?
    Quanto do drift é covariate? (mesma fonte)
Q3. Qual feature do `iw_per_feature.csv` explica mais drift? O ranking
    estabiliza entre anos? Reporte o top-3 por ano. (Fonte:
    `iw_per_feature.csv`)
Q4. Retreino com 2 camadas + lr=1e-5: catastrophic_ratio baixou ou
    subiu vs Mês 3 inicial (que era 0.832)? Recovery melhorou? Os logs
    por época mostram convergência ou early-stop? (Fontes:
    `retrain_results.csv`, `retrain_epoch_log_bp*.csv`)
Q5. Division A continua plana após o ajuste de baseline? E 2021 (Both)?
    (Fonte: `by_division_decomposition.csv`)

ACEITE: arquivo existe, 5 perguntas respondidas com número e fonte
explícita, sem afirmação não suportada.

═════════════════════════════════════════════════════════════════
TAREFA 2 — Frente 3 do PLANO_MES4: decidir cenário central
═════════════════════════════════════════════════════════════════

Aplique a árvore de decisão do `PLANO_MES4.md`:

  delta_A < 0.3 mm e delta_B (ou Both) > 1.5 mm
      -> Cenário 1: "drift é confundimento de divisão"
  delta_A > 0.5 mm e recovery_pct mediano < 50%
      -> Cenário 2: "detectar é fácil, adaptar leve não"
  nenhum dos acima decisivo
      -> Cenário 3: descritivo sem afirmação forte

Os números atuais (compare com `Relas/results/mes3/`):
  Division A delta médio (vs 2019) é 0.39 mm (média dos 4 anos), mas
  com 2 anos (2023, 2025) acima de 0.65 mm.
  Recovery mediano IW = ~65%.
  Retreino: recovery 33%, cat_ratio 0.82.

Adicione ao final do `Relas/ANALISE_MES4.md`:
- "Cenário escolhido: ___ — justificativa em 1 parágrafo"
- "Suporte numérico" — bullet com cada número que sustenta a decisão.

ACEITE: cenário declarado, justificativa quantitativa, decisão única
(não "1 ou 2").

═════════════════════════════════════════════════════════════════
TAREFA 3 — Frente 4 do PLANO_MES4: atualizar checkpoint.tex
═════════════════════════════════════════════════════════════════

`Relas/checkpoint.tex` tem hoje seções de Mês 1 e Mês 2. Adicione
seções novas no LUGAR CERTO (antes da seção "Lista de artefatos
atuais" e da bibliografia):

3.1. \section{M\^es 3 --- Adapta\c{c}\~ao ao drift}
     subsec: Importance weighting (LSIF) — fórmula da decomposição,
       ESS, tabela de iw_decomposition.csv resumida (top-3 anos),
       parágrafo sobre per-feature (qual domina).
     subsec: Retreino seletivo — Pelt detection, fine-tuning leve,
       reportar recovery 33% e catastrophic_ratio 0.82, mostrar epoch
       log do bp0.

3.2. \section{M\^es 3 --- Valida\c{c}\~ao por divis\~ao}
     - Mostrar by_division_decomposition.csv como tabela.
     - Discutir: o drift agregado sobrevive ao corte por divisão?
       Em A, sim ou não? (Resposta numérica.)
     - 2021 (Both) é tratado separadamente.

3.3. \section{An\'alise cr\'itica e tese central}
     - Declarar o cenário (resultado da Tarefa 2).
     - Limitações honestas: tamanho amostral, ausência de Division B,
       calibração genérica de detectores.

ATUALIZAR também:
- Resumo (1º parágrafo): incluir Mês 3 e 4 brevemente.
- Linha do tempo: bullet novo "Mês 4 — consolidação e redação".
- Bibliografia: garantir que Sugiyama 2008/2012 e Truong 2020 são
  citados onde apropriado.

ACEITE: `pdflatex Relas/checkpoint.tex` compila sem erro
(2 passes), página final aumenta em pelo menos 3 páginas, citações
funcionam.

═════════════════════════════════════════════════════════════════
TAREFA 4 — Frente 5 do PLANO_MES4: relatório final / paper
═════════════════════════════════════════════════════════════════

PERGUNTA AO USUÁRIO ANTES DE COMEÇAR: o relatório final é (a)
relatório de IC formal em `Relas/main_pt.tex` ou (b) paper de
workshop separado em `Relas/paper_robocup.tex`? Espere a resposta.

Se (a): trabalhe em `Relas/main_pt.tex`. Aproveite o esqueleto
existente. Vai precisar de seções:
  1. Introdução (drift em SSL, gap)
  2. Dataset e modelo
  3. Granularidades introduzidas (Mês 1)
  4. Detecção formal (Mês 2)
  5. Decomposição e adaptação (Mês 3)
  6. Tese central + limitações
  7. Trabalho futuro

Se (b): crie `Relas/paper_robocup.tex` (6-8 páginas). Mesma
estrutura, mais comprimido.

Aproveite TODAS as figuras já produzidas em `Relas/results/`. Cada
PDF é candidato a paper figure — cite com `\includegraphics`.

ACEITE: arquivo `.tex` compila, tem 6-15 páginas, todas as 4 perguntas
de pesquisa do `ANALISE_MES4.md` aparecem como afirmações no texto
com citação ao CSV.

═════════════════════════════════════════════════════════════════
TAREFA 5 — Frente 6 do PLANO_MES4: housekeeping
═════════════════════════════════════════════════════════════════

5.1. Rode `pytest -q tests/` e cole a saída. Se algo falha, conserte.
5.2. Verifique `requirements.txt` — `river`, `ruptures`, `densratio`
     (se foi usado) presentes. Se não, atualize.
5.3. Crie ou atualize `README.md` no topo do repo. 1 parágrafo:
     "Projeto de IC sobre drift em Seq2Seq de predição no SSL.
     Pipeline em `Relas/PIPELINE.md`. Resultados em
     `Relas/results/mes{1,2,3,4}/`."
5.4. `git status` — me mostre. Se houver pendências, sugira commits
     atômicos.
5.5. Crie tag `v1.0-ic` apontando para o último commit:
     `git tag -a v1.0-ic -m "Iniciacao Cientifica - submissao"`

ACEITE: pytest passa, requirements completo, README criado, tag
existe, working tree limpo.

═════════════════════════════════════════════════════════════════
COMO REPORTAR
═════════════════════════════════════════════════════════════════

A cada tarefa concluída:
  [OK] Tarefa N — <descrição curta>
       Arquivos criados/modificados: <lista>
       Próximo: Tarefa N+1

A cada problema:
  [FAIL] Tarefa N — <descrição>
         Erro: <stack ou descrição>
         Hipótese: <o que pode ser>
         Pergunta: <específica para humano>

Não pule pra próxima tarefa com a anterior em [FAIL] sem confirmação.
Não invente número que não está em CSV ou figura.

Comece pela Tarefa 1.
===
```

---

Este prompt cobre Frentes 2 a 6 do `PLANO_MES4.md`. A Tarefa 4 tem um ponto de bifurcação onde o Claude vai te perguntar qual formato (relatório de IC vs paper de workshop) — responda quando ele perguntar.

A Tarefa 1 (análise dos resultados) é a mais importante; o resto deriva dela. Se em algum ponto o Claude começar a inventar números ou pular fontes, interrompe e pede pra ele citar o CSV de origem.
