# Revisão aplicada — mapa comentário → alteração

Dois arquivos, mesmo conteúdo base:

| Arquivo | O que é |
|---|---|
| `main.tex` → `main.pdf` (5 pág.) | versão **antiga**, anotada: grifo amarelo = texto que saiu ou foi reescrito; marcas `[+ ...]` = onde entrou material novo |
| `main_corrigido.tex` → `main_corrigido.pdf` (6 pág.) | versão **final**, sem nenhuma marcação, já com a revisão de estilo |

O `main_corrigido.tex` não tem mais `\chg`, `soul` nem `xcolor` — é o arquivo pronto
para submissão. Para ver o que mudou, compare com `main.pdf`.

---

## Comentários no PDF da SBR (`main (3)_revMaximo.pdf`, 24 anotações)

| # | Comentário | O que foi feito |
|---|---|---|
| 1, 14, 19 | "world-cup" → RoboCup (confusão com a Copa do Mundo) | trocado nos 3 pontos: abstract, introdução e Seção III |
| 2, 13 | "audits" soa estranho; usar "contributes by verifying / evaluates" | abstract usa "measures how ... ages"; introdução abre com "This work contributes by measuring..." |
| 3, 15 | Kalman benchmark parece padrão da literatura, mas é baseline seu — definir | abstract: "a baseline built on a Kalman filter for tracking and prediction"; introdução define formalmente o termo *Kalman-filter baseline*; o termo passou a ser usado de forma consistente no texto todo (inclusive na Tabela IV) |
| 4 | "+0.73 mm em 2022–2025 em cima de +3.05 mm" ficou confuso | reescrito na Seção IV separando explicitamente os dois efeitos: o salto pontual de 2021 e a tendência gradual de 2022–2025, ambos medidos contra a mesma baseline de 2019 — "not one on top of the other" |
| 5 | LSIF não definido no abstract | sigla removida do abstract (agora "importance weighting") e definida por extenso em Related Work |
| 6 | métrica 0.82 → 0.46 sem definição | números tirados do abstract; `cat_ratio` só aparece na Seção VI, onde já está definido |
| 7 | "Fisher term" sem definição | tirado do abstract; na Seção VI cada símbolo de EWC ($\theta_i$, $\theta^*_i$, $F_i$) passou a ser definido |
| 8 | falso alarme medido em relação a quê? | abstract: "false alarms **on the drift-free training season**"; Seção VII: "where no drift exists **and every alarm is therefore false**" |
| 9 | abstract ruim, termos e números jogados | **abstract inteiro reescrito** em nível mais alto: sem siglas não definidas, mantendo só 59–72 % e 0.13/1k, e fechando com a mensagem prática ("component with a shelf life") |
| 10 | falta outline do artigo | parágrafo "The remainder of this paper is organized as follows..." no fim da introdução |
| 11 | citar Steuernagel ao falar de seq2seq | citação adicionada na frase de seq2seq da introdução |
| 12 | "remains the reference architecture" é claim forte demais | frase removida |
| 16, 17, 18 | Related Work curto, seco, e fala pouco de drift em si | seção passou de ~18 para ~45 linhas (≈1 coluna). Cada trabalho ganhou uma frase explicando o que ele faz; a subseção virou "**Data drift and its detection**", com um trecho novo sobre por que a distinção virtual/real importa na prática e sobre o trade-off estabilidade–plasticidade; covariate shift ganhou a explicação do que é importance weighting e por que estimar a razão direto |
| 20 | 2020 não teve competição; 2021 é que foi remoto | corrigido: "2020, when the tournament was not held at all because of the COVID-19 pandemic" |
| 21 | Division A: 6 partidas por ano ou amostra? Explicar; citar Division A na introdução | introdução ganhou uma frase dizendo o que é a Division A e por que ela importa; Seção III explicita "the same number in every year, so that no season is over-represented" |
| 22 | 2021 foi **durante** a pandemia, não depois | corrigido, e acrescentado que rodou no grSim (sugestão do outro PDF) |
| 23 | "simulation-to-reality gap" está invertido | trocado por "the gap between simulation and reality" nos 2 pontos (Seção III e limitações) |
| 24 | "League composition is constant..." incompreensível | reescrito: "Because every log comes from the same division, the growing error cannot be explained by a change in the competitive level of the sample" |

## Comentários do PDF longo que valem aqui (`main_en_revMaximo.pdf`)

| Tema | O que foi feito |
|---|---|
| "Kalman" sozinho → "Kalman filter" (8 anotações) | termo único *Kalman-filter baseline* em todo o texto, tabela inclusive |
| filtro de Kalman não é treinado, então "degrada" é estranho | Seções III e IV: explicitado que ele não tem distribuição de treino da qual se afastar, e que o erro cresce porque os robôs deixam de seguir o modelo de velocidade constante |
| declarar a contribuição explicitamente | "This work contributes by measuring... by identifying... and by turning that mechanism into..." |
| não minimizar a contribuição | "This paper does not propose a new architecture" foi removido |
| não falar em replicar o artigo do Steuernagel | "The replicated architecture follows [2]" → "The predictor follows the architecture of [2]" |
| motivar o drift (liga/hardware/estratégia evoluem) | parágrafo novo na introdução |
| definir todo símbolo matemático | definidos: $x,y$/$v_x,v_y$ no plano horizontal, $\psi$ como ângulo de guinada, $\hat p_{i,t}$/$p_{i,t}$, $e_i$, $\mathrm{ADE}_y$, $\boldsymbol\alpha$, $\theta_i$, $\theta^*_i$, $F_i$ |
| notação $30 \rightarrow 15$ inexplicada | "$a \rightarrow b$ denotes observing $a$ past frames and predicting the next $b$ (0.5 s de histórico para 0.25 s de futuro a 60 Hz)" |
| "Bahdanau attention" com 3 autores | "additive attention as proposed by Bahdanau et al. [18]" |
| explicar o que olhar nos gráficos de KS/WD | parágrafo da Seção IV explica as duas distâncias e como lê-las; legenda da Fig. 2 ampliada |
| escopo vs. Steuernagel pertence à introdução | parágrafo "Relative to [2], the contribution is a shift of question" **movido** da conclusão para a introdução |
| "Conclusion" → "Conclusions and Future Work" | título trocado |
| ≥ 10 citações | são 19 |

---

## Dois pontos que dependem de você

1. **Normalização das features** (comentário: "Como a normalização é feita? StdScaler ou MinMaxScaler?").
   Nem o `main_pt.tex` nem o `main_en` dizem qual é. Escrevi uma formulação verdadeira mas
   genérica — que ela segue a implementação de referência, é ajustada em 2019 e aplicada
   sem refit aos anos seguintes (o que é o fato relevante para o drift). **Troque por
   o nome do scaler** quando confirmar no código: Seção III, parágrafo *Model and benchmark*.

2. **As seis partidas da Division A.** Escrevi "the same number in every year, so that no
   season is over-represented", que responde ao *porquê* sem afirmar se são todas as
   partidas disponíveis ou uma amostra. Se for amostragem aleatória, vale dizer.

---

## Revisão de estilo (passe de "humanização")

Feita depois de remover as marcações, sobre o texto todo. Nenhum número, tabela,
figura ou afirmação técnica foi alterado — só a redação.

- **Voz.** Introduzido o "we" nas decisões de método ("we compared four families",
  "we fit the normalization once", "we judge a detector by"). O texto era 100 %
  impessoal, o que por si só soa artificial num paper de conferência.
- **Ritmo.** Frases longas quebradas em curtas, alternando com as que ficaram longas.
  O texto final tem média de 20 palavras por frase com desvio-padrão de 16 — a
  variação alta é justamente o que falta em texto gerado, que tende ao uniforme.
- **Construções removidas** (as marcas típicas de texto gerado):
  - pseudo-clivadas: "*What* these evaluations share *is*...", "...*is what* makes
    the detector viable", "...*is what* decides what to do next" → 2 restantes, ambas naturais
  - antíteses "not X but Y": 4 ocorrências → 0
  - o fecho grandiloquente do abstract ("...can be measured, explained and monitored")
  - "The distinction is practical rather than academic", "The right reading is not
    that...", "The finding here is that..."
  - travessões: 10 → 7, todos em aposto legítimo
- **Léxico.** Zero ocorrências de "delve", "leverage", "showcase", "crucial",
  "pivotal", "Moreover", "Furthermore", "In conclusion".
- **Variação.** Repetições de estrutura entre parágrafos vizinhos desfeitas (vários
  parágrafos abriam com "X é Y: explicação").

## Estado da compilação

- `main.pdf`: 5 páginas · `main_corrigido.pdf`: **6 páginas** (limite da SBR é 6)
- zero `Overfull \hbox`, zero citação/referência indefinida em ambos
- double-blind mantido: sem autores no corpo, metadados do PDF limpos (`Author:` vazio,
  `Title: SBR 2026 Submission`)
- o `check_double_blind.ps1` só acusa comentários LaTeX (linhas `%`, não renderizadas)
