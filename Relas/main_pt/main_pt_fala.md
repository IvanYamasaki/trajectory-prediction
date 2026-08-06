# Roteiro de fala — *Data Drift in a Seq2Seq Model for SSL Robot Trajectory Prediction*
**RoboCup Symposium 2026 — Ivan G. Yamasaki, Marcos R. O. A. Máximo, Paulo M. Tasinaffo (ITA)**

> Duração-alvo: ~13–15 min + perguntas. Tempo sugerido por slide entre colchetes.
> Marque em **negrito** os números que você quer garantir que saem na fala.

---

## Slide 1 — Título [~0:40]

Bom dia a todos. Meu nome é Ivan Yamasaki, sou do Instituto Tecnológico de Aeronáutica, e este é um trabalho conjunto com os professores Marcos Máximo e Paulo Tasinaffo.

O título é "Data Drift em um modelo Seq2Seq para predição de trajetórias na Small Size League". A pergunta que motiva o trabalho é simples de enunciar: um modelo de aprendizado treinado numa temporada continua confiável nas temporadas seguintes? Vou mostrar que a resposta é "não sem cuidado" — e, mais importante, vou quantificar *quanto*, explicar *por quê*, e propor *o que fazer a respeito*.

---

## Slide 2 — Motivação [~1:00]

Na Small Size League, prever a trajetória dos robôs em tempo real alimenta planejamento e decisão. O benchmark clássico é um filtro de Kalman de velocidade constante: barato, mas com hipóteses cinemáticas restritivas.

Um modelo seq2seq aprende padrões muito mais ricos — só que aprende do período de treino. E aqui está o ponto: a liga muda entre temporadas. Estratégias, aceleração dos robôs, dinâmica de jogo evoluem. Se a distribuição dos dados muda, o modelo envelhece.

A nossa pergunta central tem três partes: **quanto** o modelo degrada, **por quê**, e **o que fazer** — adaptar, detectar, ou os dois.

---

## Slide 3 — Trabalhos relacionados e a lacuna [~0:50]

O trabalho conversa com três literaturas. Predição de trajetórias — Social-LSTM, Trajectron++, AgentFormer, MID — mas quase sempre em recortes de dados fixos. Detecção de drift — ADWIN, Page-Hinkley, KSWIN — onde Lu e colegas apontam a taxa de falsos alarmes calibrada como requisito central. E quantificação de covariate shift — LSIF, RuLSIF, o classifier two-sample test.

A lacuna que a gente ataca: a **estabilidade temporal entre temporadas** de um preditor de trajetórias é pouco explorada. É exatamente esse buraco que o artigo preenche.

---

## Slide 4 — Quatro perguntas [~0:50]

O trabalho não propõe uma nova arquitetura. Ele responde quatro perguntas sobre um modelo de referência.

Primeira: como o erro evolui por temporada, comparado ao Kalman? Segunda: que fração da degradação é covariate shift, e qual característica dos dados domina? Terceira: adaptação leve — fine-tuning, EWC, replay — recupera desempenho sem esquecer o passado? E quarta: dá para operar um detector online de drift com falsos alarmes, cobertura e latência aceitáveis?

Cada resultado que vou mostrar responde a uma dessas.

---

## Slide 5 — Dados e protocolo [~1:00]

Os dados são os logs oficiais dos mundiais de **2019 a 2025**, sem 2020, que não teve mundial presencial. São **6 jogos por ano, 36 jogos**, a 60 Hz, todos da Divisão A. Isso vira um stream ordenado de **288 mil trajetórias**, cerca de 48 mil por ano.

A métrica é o ADE — erro médio de deslocamento — em milímetros, nos horizontes 30 para 15 e 60 para 30 passos.

Dois cuidados importantes. O baseline é 2019 com os seis jogos: se a gente usasse só um jogo, o ADE varia de 4,5 a 6,2 mm e inverteria a ordem em 4 dos 5 anos — ou seja, a análise por jogo único seria enganosa. E 2021 foi uma edição **virtual, em simulação**: o salto de erro dela mistura a lacuna simulação-real com o drift sazonal, então ela é sempre sinalizada à parte.

---

## Slide 6 — Modelo e benchmark [~0:50]

O modelo é o de referência, sem modificações — para uma comparação limpa. Encoder LSTM de 128 unidades, atenção de Bahdanau, decoder auto-regressivo sem teacher forcing. A entrada são seis variáveis normalizadas: posição, velocidade e orientação. 

Um detalhe que importa: o alvo é em velocidades incrementais, o *dv*. Isso torna o modelo naturalmente sensível a drift de velocidade e aceleração — que, vou adiantar, é justamente onde a mudança acontece.

O benchmark de Kalman prevê *dv* nulo: é determinístico, fixo, e idêntico em todos os anos — a régua neutra contra a qual medimos.

---

## Slide 7 — Resultados (divisória) [~0:10]

Com o cenário montado, vamos aos resultados.

---

## Slide 8 — O drift é real e afeta a cauda primeiro [~1:10]

Primeira pergunta, respondida nesta figura. No painel (a), o ADE do seq2seq por ano: ele cresce depois de 2019, com excesso crescente de 2022 a 2025, e um salto adicional em 2021 — a edição virtual. Mas repare: mesmo degradado, o seq2seq fica muito abaixo do Kalman o tempo todo.

O painel (b) mostra o aumento percentual em relação a 2019 — chega a **+67% em 2021**. E o (c) é a leitura mais fina: a razão p99 sobre p50 do erro. Em 2021, o salto é de cauda — a razão sobe de 4,4 para 4,9. Já de 2022 a 2025, a razão cai abaixo do baseline, enquanto a **mediana sobe cerca de 25%**. 

A mensagem: o regime novo entra primeiro como cauda, em 2021, e depois se incorpora ao corpo da distribuição. O drift é real, é gradual, e entra pela cauda.

---

## Slide 9 — O que mudou nos dados: aceleração lidera [~0:50]

Se o erro muda, o que mudou nos dados? Aqui comparo cada ano contra 2019 com duas estatísticas de distância entre distribuições: KS no painel (a), Wasserstein relativa no (b). As duas concordam: **velocidade e aceleração mudam bem mais que o giro**, e a aceleração tem o maior deslocamento.

E o painel (c) fecha o argumento: o ADE médio por jogo contra a aceleração de pico do jogo, o accel_p99. A associação é forte — **Spearman 0,86** — e consistente entre anos. Ou seja: os jogos com mais aceleração de cauda são os jogos onde o modelo mais erra.

---

## Slide 10 — Quantificação: importance weighting triangulado [~1:00]

Correlação não é decomposição. Para medir *quanto* do excesso de erro é covariate shift, uso importance weighting.

A ideia: a densidade conjunta é p(y|x) vezes p(x). Covariate shift é uma mudança só em p(x) — e isso é corrigível por reponderação, sem retreinar. Eu estimo a razão de densidades w(x) — probabilidade no ano-alvo sobre probabilidade em 2019 — com LSIF, usando kernels gaussianos. Reponderar o erro do ano-alvo para a distribuição de 2019 me diz quanto do erro *some* quando neutralizo a mudança de p(x).

Para não depender de um único estimador, eu triangulo: RuLSIF como variante robusta a caudas, e o classifier two-sample test como árbitro independente, via AUC. E todos os intervalos de confiança usam block bootstrap, com o bloco sendo o jogo, 2 mil réplicas.

A interpretação da recovery: 100% significa drift puramente covariate; 0% significa concept drift puro.

---

## Slide 11 — Decomposição do excesso por ano [~1:20]

Esta é a tabela central do artigo. Cada linha é um ano. Vou ler as colunas: o ADE observado, o ADE reponderado para 2019, a recovery do LSIF em porcentagem com intervalo, a correção absoluta em milímetros, e as duas colunas do classifier — AUC e cobertura.

O que ler aqui: a recovery do LSIF fica entre **59 e 72%** nos anos com excesso. A coluna de correção mostra que o limite inferior do intervalo é **positivo em todos os anos com excesso** — então a correção é estatisticamente real, não ruído. Em 2021, por exemplo, dos 3 mm de excesso, quase 1,9 mm é explicado por covariate shift.

Três notas de rigor. 2024 tem excesso negativo — o modelo até melhora — então fica fora do cálculo de recovery. 2022 tem um excesso muito pequeno, 0,23 mm, então a leitura percentual dele pede cautela — é o daga. E o classifier confirma tudo com p menor que 10⁻³ por permutação.

A conclusão: **a maior parte do excesso é covariate shift**. Sobra uma fração de 30 a 40% que sugere concept drift residual.

---

## Slide 12 — Covariate shift explica entre 59% e 72% [~0:40]

Resumindo a triangulação num slide. O LSIF diz 59 a 72%. O RuLSIF, ajustado diretamente, diz só 9 a 23% — mas isso é um artefato: ele produz pesos quase uniformes e sub-responde; quando aplico a transformação de razão relativa *sobre* os pesos do LSIF, a recovery se mantém. E o classifier, o árbitro independente, dá cobertura de 22 a 51%, com sinal positivo em 2021, 2023 e 2025.

Os três métodos apontam na mesma direção: **o drift é majoritariamente covariate**.

---

## Slide 13 — accel_p99 é o vetor dominante do drift [~0:50]

E qual característica carrega esse covariate shift? Esta tabela decompõe a recovery por feature. A leitura é clara e estável entre temporadas: **a aceleração de cauda, o accel_p99, lidera** — em torno de 52 a 57% de recovery marginal, e o menor ESS, que indica o maior deslocamento. Depois vêm o accel_p90 e o accel_mean.

A velocidade contribui menos, e o **giro é residual, abaixo de 5%** — praticamente não mudou. Então o drift tem um endereço: a cauda de aceleração dos robôs. Guardem isso, porque ele volta na detecção.

---

## Slide 14 — Adaptação: EWC é inerte, replay funciona [~1:10]

Terceira pergunta: dá para adaptar sem esquecer? Aqui aparece o catastrophic forgetting. O fine-tuning ingênuo recupera erro — ADE cai 2,7% — mas destrói o passado: o cat_ratio, a fração do regime antigo que piora, vai a 0,82 já na primeira época.

Um ajuste conservador — learning rate baixa, encoder congelado — derruba esse esquecimento de 0,82 para 0,44. Aí testei o EWC, a regularização clássica contra esquecimento, varrendo o lambda por **8 ordens de grandeza**. E o resultado é que o EWC é **inerte**: o cat_ratio nem se mexe, fica em 0,44–0,45. A proteção vinha da learning rate baixa e do congelamento, não da regularização de Fisher.

O que de fato funciona é o mais simples: um buffer de replay com metade de dados antigos. O cat_ratio cai para **0,418**, e ele melhora o regime antigo *e* o novo ao mesmo tempo. Essa é a base recomendada.

---

## Slide 15 — Adaptação: replay uniforme domina, seletivo piora [~1:00]

Esta tabela consolida as estratégias, todas sob o mesmo protocolo conservador. Menor cat_ratio é melhor. A linha em destaque é o **replay uniforme: 0,418**, e é o melhor nas três colunas — menor esquecimento e menor erro nos dois regimes.

O achado contra-intuitivo está nas três últimas linhas, que são dirigidas pelo diagnóstico de aceleração que acabei de mostrar. A gente esperava que curar os dados pela cauda de aceleração ajudasse. Mas o **seletivo por aceleração piora** o esquecimento — chega a 0,454, pior que o controle. O motivo: a cauda é justamente o subconjunto mais distante de 2019; treinar só nela desloca ainda mais os pesos. O corpo intacto da distribuição funciona como âncora contra a interferência.

A lição: o diagnóstico é ótimo para *monitorar* e *explicar*, mas não vira curadoria de dados vantajosa. A rota efetiva é o replay uniforme.

---

## Slide 16 — Detecção online: calibrar antes de confiar [~1:10]

Quarta pergunta: um detector online operacional. Trabalho sobre o stream de ADE, as 288 mil trajetórias.

Duas descobertas de calibração. Primeira: suavizações convencionais — média móvel, EWMA, Holt, Savitzky-Golay — *pioram* o ADWIN, porque introduzem autocorrelação que viola a hipótese i.i.d. do detector; a FAR chega a explodir de 1,2 para 5–9 por mil. A única transformação que *reduz* a FAR é uma normalização z-robusta local, mediana e MAD numa janela de 200 trajetórias — que corta a FAR pela metade sem criar autocorrelação.

Sobre esse sinal, uma busca em grade seleciona o ADWIN com **FAR de 0,042 por mil e cobertura de 5 em 5 anos**. Já o Page-Hinkley zera a FAR mas só detecta 2021 — é validador de salto abrupto, não monitor. E o KSWIN é inadmissível, com FAR acima de 0,4.

---

## Slide 17 — Resultado negativo: AND entre escalas distintas [~0:40]

Um resultado negativo que vale a pena. A ideia natural de combinar dois detectores com AND para reduzir falsos alarmes falha quando eles operam em *escalas temporais distintas*. O ADWIN é por trajetória; o Page-Hinkley agregado é por janela. O par de alarmes mais próximo entre eles dista **214 trajetórias**, maior que a janela de coincidência. Para cobrir mais de um ano, seria preciso uma janela de dez mil trajetórias — cerca de dois meses e meio de dados. Estruturalmente inviável. A lição: para combinar detectores por AND, eles precisam estar na mesma escala.

---

## Slide 18 — Detecção em camadas [~1:10]

E aqui está a arquitetura final, validada com o rigor que faltava na literatura: **holdout 2-fold por jogos**. Todos os números são out-of-sample — metade dos jogos escolhe a configuração, a outra metade, intocada, é avaliada uma vez. Isso elimina o viés de otimizar e reportar no mesmo dado.

São três camadas mais um corroborador. A **primária** é o ADWIN sobre o erro — FAR de 0,04 a 0,13 por mil, cobertura quase total, latência abaixo de um jogo e meio: é ela que aciona retreino. A **vigilância** é o accel_p95, o vetor do drift que diagnosticamos — ele não é admissível como gatilho de retreino sozinho, mas *antecipa* o alarme de erro em **8 de 10 anos-fold**, com latência mínima. A **confirmação** é o AND entre erro e aceleração, na mesma escala agora — zera os falsos alarmes, ao custo de cobertura. E o ADE do Kalman serve de corroborador redundante.

Então o diagnóstico que não virou curadoria de dados vira **arquitetura de monitoramento**.

---

## Slide 19 — Limitações [~0:45]

Sendo honesto sobre o alcance. Só Divisão A, e seis jogos por ano — os intervalos de confiança por bloco são largos. 2021 em simulação mistura a lacuna sim-real com o drift sazonal, então as conclusões de tendência gradual se apoiam em 2022 a 2025. O ADE é single-mode; um modelo multimodal exigiria min-ADE. A normalização z-robusta assume drift gradual — sob mudança abrupta, ela absorveria parte do sinal. E a fração de concept drift residual não foi isolada por feature.

---

## Slide 20 — Conclusões [~1:00]

Cinco mensagens para levar para casa. Um: o drift é real, gradual, e entra pela cauda — **+67% em 2021**, e o Kalman degrada ainda mais. Dois: é majoritariamente covariate shift, direcional, dominado pela aceleração de cauda. Três: na adaptação, o EWC é inerte em 8 ordens de lambda, e um replay simples, com cat_ratio 0,418, é a base. Quatro: a detecção operacional é viável — ADWIN mais z-robusto, FAR 0,042 por mil, cobertura total, latência abaixo de meio jogo no canal OR. E cinco: o AND só funciona na mesma escala temporal.

No fundo, a mensagem única: **o diagnóstico do drift vira arquitetura de monitoramento**. Obrigado — fico à disposição para perguntas.

---

## Slide 21 — Obrigado [—]

*(Slide de encerramento — deixe no ar durante as perguntas. Contato: ivangyamasaki@gmail.com.)*

---

## Slides 22–24 — Backup (só se perguntarem)

- **Slide 22 — divisória "Backup".**
- **Slide 23 — Correlações de Spearman (ADE × features).** Se perguntarem sobre a força da associação por feature: excluindo 2020/2021, o ADE do seq2seq correlaciona mais forte com accel_p90 e accel_p99 — coerente com a decomposição da recovery.
- **Slide 24 — Latência do primeiro alarme, por ano.** Se perguntarem sobre latência: o primário sozinho varia de 0,3 a 2,8 jogos; o canal OR com o Kalman reduz para ≤0,4 jogo em 4 dos 5 anos. São ~8 mil trajetórias por jogo, e é um limite superior da latência real, já que o drift é gradual.

---

### Antecipando perguntas

- **"Por que não uma arquitetura melhor?"** — O objetivo é medir estabilidade temporal de um modelo de referência, não vencer benchmark. Trocar a arquitetura não responde as quatro perguntas.
- **"Por que o RuLSIF discorda do LSIF?"** — É artefato da regularização do ajuste do RuLSIF (pesos quase uniformes, ESS ≥ 0,94), não do estimando. As verificações de truncamento e de razão relativa sobre os pesos do LSIF confirmam que a recovery é robusta.
- **"288 mil trajetórias, mas só 36 jogos — a estatística aguenta?"** — Por isso o bloco do bootstrap é o jogo, não a trajetória; e por isso os ICs de Wilson usam o tamanho amostral efetivo sob autocorrelação. Somos conservadores de propósito.
- **"O detector serve em produção?"** — Como camada de acionamento de retreino, sim: FAR calibrada out-of-sample, cobertura e latência operacionais. A honestidade é que, com só 6 jogos de 2019, afirmar admissibilidade a 95% de confiança pediria mais dados.
