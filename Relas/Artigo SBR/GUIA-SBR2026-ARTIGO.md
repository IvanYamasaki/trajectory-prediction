# Guia Completo — Como escrever e submeter um artigo para o SBR 2026 (Brazilian Symposium on Robotics)

> **Público-alvo deste documento:** um agente de IA (ou um humano) que vai escrever/montar o arquivo `.tex` do artigo.
> **Status:** compilado a partir do site oficial `https://www.natalnet.br/sbr2026/` (Call for Papers, seção `#cfp`) em 26/07/2026, do template oficial IEEE Conference (Overleaf `grfzhhncsfqn`) e das práticas padrão de revisão double-blind da SBC/IEEE.
> **Regra de ouro:** onde este guia disser **[VERIFICAR]**, a informação não está explícita no site oficial e deve ser confirmada antes do envio final.

---

## 0. TL;DR — as 12 regras que não podem ser violadas

1. **Idioma: inglês.** Obrigatório, do título às referências.
2. **Máximo 6 páginas**, incluindo título, resumo, tabelas, figuras **e referências**. Não há página extra paga anunciada.
3. **Formato IEEE Conference padrão** (`IEEEtran`, opção `conference`, duas colunas, 10pt).
4. **Revisão DOUBLE-BLIND.** A primeira versão **não pode** conter nomes de autores, e-mails, afiliações, agradecimentos, número de projeto/bolsa, nem qualquer pista de onde a pesquisa foi feita.
5. **Submissão pelo JEMS3 da SBC:** https://jems3.sbc.org.br/sbr2026
6. **Prazo de submissão: 30 de julho de 2026.** Notificação: 15/08/2026. Câmera-ready: 15/09/2026.
7. **PDF** gerado com todas as fontes embutidas, **sem numeração de página**, sem marcas d'água, sem `\thanks` identificador.
8. **Autocitação em terceira pessoa** ("In [7], the authors propose…"), nunca "In our previous work [7]".
9. **Links de código/dados anonimizados** (ex.: `anonymous.4open.science`) ou omitidos com nota "link omitted for double-blind review".
10. **Metadados do PDF limpos** (Author/Title/Creator não podem conter nome ou instituição).
11. **Apresentação presencial e em inglês**, 20 min (15 + 5 de perguntas). Sem apresentação, o artigo tende a não ir para os anais **[VERIFICAR política exata]**.
12. **Pelo menos uma inscrição Professional por artigo aceito** para publicação nos anais.

---

## 1. Dados oficiais do evento

| Item | Valor |
|---|---|
| Evento | **SBR — Brazilian Symposium on Robotics** (dentro do ROBÓTICA 2026) |
| Local | João Pessoa, Paraíba, Brasil — Centro de Convenções de João Pessoa |
| Datas | **24–27 de novembro de 2026** |
| Site | https://www.natalnet.br/sbr2026/ |
| Eventos irmãos/satélites | CBR, OBR, MNR, **WRE** (Workshop on Robotics in Education), **CTDR** (melhor dissertação/tese) |
| Modalidade | **Somente presencial** |
| Anais | IEEE / SBC **[VERIFICAR indexação IEEE Xplore para 2026 — edições anteriores do LARS/SBR foram publicadas no IEEE Xplore]** |
| Prêmio/convite | Melhores artigos selecionados são convidados a submeter versão estendida ao **Journal of Intelligent & Robotic Systems (JINT)** |

### 1.1 Datas importantes (oficiais)

| Marco | Data |
|---|---|
| Abertura do sistema de registro/submissão | **1 de maio de 2026** |
| **Deadline de submissão** | **30 de julho de 2026** |
| Notificação de aceitação | **15 de agosto de 2026** |
| **Camera-ready** | **15 de setembro de 2026** |
| Inscrição Early (até) | 15 de setembro de 2026 |
| Inscrição Regular | 16/09 – 31/10/2026 |
| Inscrição Late | 01/11/2026 – evento |

> ⚠️ Fuso horário e hora exata do deadline não estão publicados. Assuma **23h59 AoE (Anywhere on Earth)** como pior caso e envie com pelo menos 24 h de folga. **[VERIFICAR no JEMS]**

### 1.2 Regras de inscrição de autores (SBR/WRE)

- Todo artigo aceito exige **no mínimo 1 inscrição Professional**.
- Uma inscrição Professional cobre **até 2 artigos**.
- A partir do 3º artigo: Professional + **Publication Fee** por artigo extra.
- Inscrição de Graduate/Undergraduate + Publication Fee cobre **1 artigo**.

---

## 2. Escopo — tópicos de interesse

Enquadre o artigo explicitamente em pelo menos um destes tópicos (cite-o na introdução e escolha as *keywords* correspondentes):

- Vision in robotics and automation
- Symbol mediated robot behavior control
- Sensory mediated robot behavior control
- Active sensory processing and control
- Multi-Robot and Multi-Agents, Cooperation and Collaboration
- CAD-based robotics (CAD-based vision, reverse engineering)
- Robot simulation and visualization tools
- Microelectromechanical robots
- Robot modeling
- Evolutionary robotics
- Bio-Inspired robotics
- Robot soccer
- Industrial applications of autonomous systems
- Sensor modeling and data interpretation (sensor data integration, 3D scene analysis, environment modeling, pattern recognition)
- Robust techniques in AI and sensing (uncertainty modeling, graceful degradation)
- Robot programming (on-line/off-line, DES, fuzzy logic)
- Robot control architectures
- Robot planning, reasoning, communication, adaptation and learning
- Robotic manipulators
- Self-Localization, Mapping and Navigation (SLAM)
- Robots for surgery and rehabilitation
- Micro/nano-robotics, new devices and materials
- Human-Robot Interaction and Interfaces
- Robotics and Education
- Sensor Networks, embedded hardware/software architectures
- Autonomous vehicles, mobile robot platforms, service/entertainment robots
- Underwater robots, humanoids, multi-robot systems, aerial vehicles

> A lista é "including, but not limited to". Se o trabalho for de **predição de trajetória, percepção para veículos autônomos, aprendizado para navegação**, o enquadramento natural é *Autonomous vehicles / Self-Localization, Mapping and Navigation / Robot planning, reasoning, adaptation and learning*.

**Critério de aceitação declarado:** "Papers should present **new solid results** in theoretical, empirical, and applied research". Ou seja: resultado novo + evidência sólida (experimentos, baselines, métricas, ablação). Artigo puramente descritivo/position paper tende a ser rejeitado.

---

## 3. Formato exigido

### 3.1 Templates oficiais linkados pelo SBR 2026

| Template | URL |
|---|---|
| **Overleaf (IEEE oficial) — recomendado** | https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn |
| LaTeX (zip IEEE) | https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/conference-latex-template_10-17-19.zip |
| Microsoft Word (A4) | https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-template-a4.docx |

### 3.2 Especificação técnica do `IEEEtran` (conference)

```latex
\documentclass[conference]{IEEEtran}
```

Isso já define automaticamente — **não altere manualmente**:

| Parâmetro | Valor imposto pelo template |
|---|---|
| Colunas | 2 |
| Fonte do corpo | Times/Nimbus Roman 10 pt |
| Espaçamento | simples |
| Margens | definidas pela classe (não mexer) |
| Tamanho do papel | **US Letter** por padrão; o Word oficial linkado é **A4** |
| Numeração de página | **desligada** (não ligue) |
| Título | 24 pt, centralizado |
| Seções | Numeração romana (I, II, …), títulos em *small caps* |
| Legendas de figura | "Fig. 1." abaixo da figura |
| Legendas de tabela | "TABLE I" acima da tabela, em maiúsculas |
| Referências | estilo IEEE, numéricas `[1]` |

> **Letter vs A4:** o IEEE Xplore aceita ambos, mas exige consistência. O template Overleaf indicado usa Letter. **Mantenha Letter** (padrão do `IEEEtran`) a menos que o JEMS/organização diga o contrário — **[VERIFICAR]**. Nunca use `\documentclass[a4paper,conference]{IEEEtran}` junto com figuras dimensionadas para Letter sem revisar o layout.

### 3.3 Pacotes seguros x proibidos

**Seguros / recomendados:**
```latex
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}   % opcional em pdfLaTeX moderno
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}      % ou algorithm2e / algpseudocode
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}         % tabelas limpas
\usepackage{subcaption}       % subfiguras (NÃO use subfigure, obsoleto)
\usepackage{url}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}  % carregue por ÚLTIMO
\usepackage{cleveref}         % depois de hyperref
```

**Evite / cuidado:**
- `geometry` alterando margens → **desconfigura o padrão IEEE, motivo de desk reject**.
- `setspace` aumentando entrelinhas.
- `times`/`mathptmx` redundantes (a classe já resolve).
- `subfigure` (obsoleto, conflita).
- `natbib` com estilos autor-ano (IEEE é numérico).
- Comandos que reduzem fonte para caber em 6 páginas (`\small` no corpo inteiro, `\vspace{-Xmm}` agressivo). Revisores percebem e isso conta contra.

**Correção clássica de erro do template:** se aparecer `LaTeX Error: Command \IEEEoverridecommandlockouts...`, use exatamente na ordem:
```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\overrideIEEEmargins   % somente se o template original trouxer
```
Use `\IEEEoverridecommandlockouts` **apenas** se precisar de `\thanks`/copyright na camera-ready.

### 3.4 Limite de 6 páginas — o que conta

Conta tudo: título, resumo, keywords, texto, figuras, tabelas, algoritmos, agradecimentos (só na camera-ready) e **referências**. Não existe menção a páginas extras pagas para o SBR 2026 → **6 páginas é limite duro**.

Táticas legítimas para caber:
- `\vspace{-2mm}` pontual **entre** figura e legenda (aceitável, sem exagero).
- Figuras multi-painel com `subcaption` em vez de 3 figuras separadas.
- Tabelas com `booktabs` + `\footnotesize` (aceitável em tabelas).
- Mover derivações longas para o texto corrido.
- Cortar redundância entre Introdução e Trabalhos Relacionados.
- `\balance` (pacote `balance`) para equilibrar a última página **[opcional]**.

---

## 4. DOUBLE-BLIND — a parte crítica

Texto literal do CFP:

> "This year the reviewing process will be **DOUBLE BLINDED**, so authors must hide their **names, contact information, acknowledgments, and any other reference in the text to where their research takes place**."
> "All submissions will be peer-reviewed. The process is double-blinded; please remove author-identifying information from the **manuscript and supplementary materials**."

### 4.1 O que REMOVER da versão de submissão

| Elemento | Ação |
|---|---|
| Nomes dos autores | Substituir por `Anonymous Author(s)` ou omitir bloco inteiro |
| E-mails | Remover |
| Afiliação / universidade / laboratório / grupo de pesquisa | Remover (inclusive "Natalnet", "LAR", "PPgEEC", etc.) |
| Cidade/país da instituição | Remover |
| Logos institucionais em figuras | Remover/recortar |
| **Agradecimentos** (`\section*{Acknowledgment}`) | **Remover por completo** |
| Números de processo (CNPq, CAPES, FAPESP, FINEP, bolsa) | Remover |
| `\thanks{}` no comando de título | Remover |
| Nome do robô/plataforma exclusivo do laboratório | Generalizar ("a differential-drive platform") se identificar o grupo |
| Nome de dataset proprietário do grupo | Anonimizar ("an in-house dataset") |
| URLs de GitHub/lattes/site pessoal/YouTube do lab | Anonimizar ou remover |
| Nome do arquivo PDF | Use `sbr2026_paper.pdf`, nunca `artigo_ivan_ufxx.pdf` |
| Metadados do PDF (Author, Title, Subject, Creator) | Limpar (§4.5) |
| Comentários no `.tex` com nomes | Remover antes de gerar o PDF final e antes de subir fontes |
| Referência a "our lab's previous system X" | Reescrever em terceira pessoa |
| Fotos do laboratório/pessoas reconhecíveis | Cortar/borrar |
| Menções a competições específicas com o time nomeado | Generalizar |

### 4.2 Como citar o próprio trabalho (autocitação)

**❌ Errado:**
- "In our previous work [7], we showed…"
- "Extending our framework [7]…"
- "[7] (our own work)"

**✅ Certo:**
- "Silva *et al.* [7] proposed a trajectory predictor based on…, which we adopt as baseline."
- "The approach in [7] is closest to ours; it differs in that…"

**Nunca** apague as autocitações — isso prejudica o embasamento e pode parecer omissão de trabalho relacionado. Cite normalmente, em terceira pessoa.

**Caso especial (trabalho anterior ainda não publicado / em revisão):** cite como
```
[X] Anonymous, "Title omitted for double-blind review," under review, 2026.
```

### 4.3 Como tratar código, dados e vídeos

- Repositório: use **https://anonymous.4open.science/** (proxy anônimo do GitHub) ou um link de nuvem sem identificação.
- Vídeo: YouTube "não listado" em conta sem nome, ou anexo no JEMS.
- Se não for possível anonimizar: escreva
  > "Code and dataset will be made publicly available upon acceptance (link omitted for double-blind review)."
- **Material suplementar também é double-blind** — anonimize os anexos igualmente.

### 4.4 Bloco de autores anonimizado no `.tex`

Forma mais segura (sem nomes, sem quebrar o layout do `IEEEtran`):

```latex
\author{\IEEEauthorblockN{Anonymous Author(s)}
\IEEEauthorblockA{\textit{Affiliation omitted for double-blind review}\\
\textit{Institution omitted}\\
Email omitted}}
```

Alternativa mais enxuta:
```latex
\author{\IEEEauthorblockN{Submission for double-blind review}}
```

> Ambas são aceitáveis. Não deixe o campo `\author{}` vazio: alguns compiladores do JEMS/IEEEtran geram avisos ou quebram o cabeçalho.

### 4.5 Limpando metadados do PDF

**No `.tex` (recomendado, funciona no Overleaf):**
```latex
\usepackage[hidelinks]{hyperref}
\hypersetup{
  pdftitle={SBR 2026 Submission},
  pdfauthor={},
  pdfsubject={},
  pdfkeywords={},
  pdfcreator={},
  pdfproducer={}
}
```
> Observação: `pdfproducer` costuma ser sobrescrito pelo pdfTeX ("pdfTeX-1.40.x") — isso é inofensivo e não identifica autor.

**Verificação depois de compilar (linha de comando):**
```bash
pdfinfo sbr2026_paper.pdf          # poppler
exiftool sbr2026_paper.pdf         # mostra tudo
```
Procure por: `Author`, `Title`, `Subject`, `Keywords`, `Creator`, e caminhos de arquivo (`/Users/ivan/...`) que possam vazar identidade.

**Vazamento silencioso comum:** figuras PDF/EPS exportadas do MATLAB/Inkscape/Illustrator carregam o nome do autor e o caminho do arquivo nos metadados internos. Reexporte figuras como PDF limpo ou PNG 300+ dpi.

### 4.6 Checklist final double-blind (rodar antes de submeter)

```bash
# no diretório do .tex
grep -inE "univers|instituto|federal|laborat|thanks|acknowledg|cnpq|capes|fapes|fapesp|grant|github\.com|lattes|@.*\.(br|edu|com)|our previous|our lab|we previously" *.tex *.bib
```
Todo hit deve ser justificado ou removido. Depois:
- [ ] Abrir o PDF e ler o cabeçalho da 1ª página: só título, "Anonymous", abstract.
- [ ] Ctrl+F no PDF pelo seu sobrenome, pela sigla da instituição e pela cidade.
- [ ] `exiftool` limpo.
- [ ] Nome do arquivo neutro.
- [ ] Seção de Acknowledgment ausente.
- [ ] Figuras sem logo/legenda com nome.
- [ ] Links do artigo abertos numa aba anônima: nenhum leva a página identificável.

---

## 5. Estrutura recomendada do artigo (6 páginas)

Orçamento de páginas sugerido para um artigo experimental:

| Seção | Páginas | Conteúdo |
|---|---|---|
| Título + Abstract + Keywords | 0.25 | Abstract 150–250 palavras, sem citações, sem siglas não definidas |
| I. Introduction | 0.75–1.0 | Problema, motivação em robótica, lacuna, **contribuições em bullets**, organização do texto |
| II. Related Work | 0.75–1.0 | Agrupado por abordagem, terminando com "unlike these works, we…" |
| III. Methodology / Proposed Approach | 1.5–2.0 | Formalização, notação, diagrama de arquitetura, algoritmo, equações |
| IV. Experimental Setup | 0.5–0.75 | Dataset/simulador/robô, métricas, baselines, hiperparâmetros, hardware |
| V. Results and Discussion | 1.0–1.5 | Tabela principal + gráficos + **ablação** + análise qualitativa + limitações |
| VI. Conclusion and Future Work | 0.25 | Sem resultados novos; 3–5 linhas de trabalho futuro |
| References | 0.5–0.75 | 15–30 referências, majoritariamente recentes (últimos 5 anos) e de robótica |

### 5.1 Regras de escrita que os revisores do SBR cobram

- **Contribuições explícitas** em lista numerada no fim da Introdução.
- **Baselines comparativos**: artigo sem comparação com pelo menos um método da literatura é frequentemente rejeitado.
- **Métricas quantitativas** com unidade e, quando possível, **média ± desvio** sobre múltiplas execuções/seeds.
- **Reprodutibilidade**: informar versões (ROS 2 Humble, Gazebo Harmonic, PyTorch 2.x), hardware (GPU/CPU), tempo de execução, taxa em Hz para sistemas embarcados.
- **Limitações**: um parágrafo honesto ganha pontos.
- **Inglês revisado**. Erros gramaticais sistemáticos são citados em review. Passe por revisor/ferramenta.
- Evite "novel", "state-of-the-art" sem evidência; prefira números.
- Todas as figuras/tabelas devem ser **referenciadas no texto** (`Fig.~\ref{fig:arch}`, `Table~\ref{tab:results}`).

---

## 6. Convenções LaTeX do estilo IEEE (detalhe fino)

### 6.1 Figuras
```latex
\begin{figure}[!t]                 % coluna única
  \centering
  \includegraphics[width=\columnwidth]{fig/arch.pdf}
  \caption{System architecture. The perception module ...}
  \label{fig:arch}
\end{figure}

\begin{figure*}[!t]                % duas colunas (largura total)
  \centering
  \includegraphics[width=\textwidth]{fig/results.pdf}
  \caption{...}
  \label{fig:results}
\end{figure*}
```
- Use `\columnwidth` / `\textwidth`, nunca larguras absolutas em cm.
- Formato vetorial (**PDF**) para diagramas e gráficos; **PNG ≥ 300 dpi** para fotos/capturas.
- Legenda **abaixo** da figura, terminando com ponto.
- Texto dentro da figura ≥ 8 pt no tamanho final impresso.
- Figuras devem ser legíveis **em preto e branco** (use marcadores + estilos de linha, não só cor).
- `figure*` só flutua para o topo da página seguinte — planeje.

### 6.2 Tabelas
```latex
\begin{table}[!t]
  \caption{Comparison with baselines on the X dataset.}
  \label{tab:results}
  \centering
  \begin{tabular}{lccc}
    \toprule
    Method & ADE (m) $\downarrow$ & FDE (m) $\downarrow$ & Time (ms) \\
    \midrule
    Baseline A \cite{ref1} & 0.52 & 1.10 & 12 \\
    \textbf{Ours}          & \textbf{0.41} & \textbf{0.88} & 15 \\
    \bottomrule
  \end{tabular}
\end{table}
```
- Legenda **acima** da tabela.
- Sem linhas verticais; `booktabs`.
- Melhor resultado em **negrito**, com legenda explicando.

### 6.3 Equações
```latex
\begin{equation}
  \hat{y}_{t+1} = f_\theta(x_{1:t}) , \label{eq:pred}
\end{equation}
```
- Pontuação faz parte da equação (`,` ou `.` no fim).
- Referencie como `\eqref{eq:pred}` ou `(\ref{eq:pred})`.
- Defina toda variável na primeira aparição.

### 6.4 Algoritmos
```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
\begin{algorithm}[t]
\caption{Proposed planner}\label{alg:planner}
\begin{algorithmic}[1]
\Require map $M$, goal $g$
\State $\ldots$
\end{algorithmic}
\end{algorithm}
```

### 6.5 Referências (IEEE)
- Estilo numérico, ordenadas por ordem de aparição: `\bibliographystyle{IEEEtran}`.
- No texto: `[1]`, `[2], [5]`, `[3]--[6]`. Nunca "Ref. [1] shows" no início de frase → use "Reference [1] shows".
- Use `\bibliography{references}` + `references.bib` (BibTeX) ou `thebibliography` manual.
- Abreviações IEEE: `Proc. IEEE Int. Conf. Robot. Autom. (ICRA)`, `IEEE Robot. Autom. Lett.`, etc.
- Inclua DOI quando disponível **[opcional]**.
- Não cite apenas arXiv quando existe versão publicada; cite a versão de conferência/journal.

### 6.6 Miscelânea
- `~` antes de referências: `Fig.~\ref{}`, `Section~\ref{}`, `[10]~\cite{}` → evita quebra de linha ruim.
- Aspas LaTeX: ``` ``texto'' ``` (nunca `"texto"`).
- Travessão: `--` para intervalos (2020--2024), `---` para pausa.
- `\textit{et al.}` em itálico.
- `\SI{}{}` (siunitx) é opcional; se usar, seja consistente.

---

## 7. Submissão no JEMS3

**URL:** https://jems3.sbc.org.br/sbr2026

Passo a passo:
1. Criar/entrar com conta SBC no JEMS3.
2. Selecionar a trilha correta (SBR — full paper). **Cuidado:** o mesmo sistema costuma hospedar WRE e CTDR; não erre a trilha.
3. Preencher título, abstract e **keywords** (os autores *são* cadastrados no sistema — isso é normal e não quebra o double-blind, pois os revisores não veem o metadado; o **PDF** é que precisa estar anônimo).
4. Fazer upload do **PDF anônimo**, nome de arquivo neutro.
5. Anexar material suplementar (também anônimo), se houver.
6. Confirmar a submissão e **guardar o número do paper**.
7. Reabrir o PDF submetido pelo próprio sistema e conferir se subiu a versão certa.

> **[VERIFICAR]** se o JEMS do SBR 2026 exige cadastro prévio de abstract (paper registration) antes do upload do PDF — o site menciona "Paper Registration/Submission Site Open: May 1st, 2026", o que sugere registro e submissão no mesmo sistema.

---

## 8. Camera-ready (após aceitação — deadline 15/09/2026)

Ao preparar a versão final:

1. **Reinserir** nomes, afiliações, e-mails no `\author{}` no formato `IEEEauthorblockN`/`IEEEauthorblockA`.
2. **Reinserir** a seção `\section*{Acknowledgment}` com agências de fomento e números de processo.
3. Restaurar links reais de código/dados/vídeo.
4. Aplicar as **correções pedidas pelos revisores** e, se houver, escrever a carta de resposta.
5. Manter o **limite de 6 páginas** (a desanonimização consome espaço — planeje!).
6. Verificar compliance IEEE Xplore **[VERIFICAR se será exigido]**:
   - **IEEE PDF eXpress** para validar/converter o PDF;
   - **eCopyright (IEEE Copyright Form)**;
   - fontes 100 % embutidas (Type 1/TrueType), sem fontes bitmap Type 3;
   - sem numeração de página, sem cabeçalho/rodapé próprio;
   - possível **notice de copyright** no rodapé da 1ª página, fornecido pela organização:
     ```latex
     \IEEEoverridecommandlockouts
     \IEEEpubid{\makebox[\columnwidth]{979-8-XXXX-XXXX-X/26/\$31.00~\copyright2026 IEEE \hfill}
       \hspace{\columnsep}\makebox[\columnwidth]{}}
     ```
     (só use com o código real informado pela organização)
7. Registrar **pelo menos 1 inscrição Professional** por artigo (ver §1.2).
8. Preparar a apresentação: **inglês, presencial, 15 min + 5 min Q&A, .ppt ou .pdf, levar em pen drive** e entregar no escritório da conferência no dia.

---

## 9. Erros que causam desk reject ou nota baixa

| Erro | Consequência |
|---|---|
| Mais de 6 páginas | Desk reject |
| Nome/afiliação/agradecimento visível | Desk reject por quebra de double-blind |
| Artigo em português | Desk reject |
| Template não-IEEE ou margens alteradas | Desk reject |
| Metadados do PDF com nome do autor | Risco alto de desk reject |
| Autocitação em primeira pessoa ("our previous work") | Quebra de anonimato |
| Plágio / auto-plágio / submissão simultânea a outro evento | Rejeição e sanções |
| Sem comparação com baseline | Rejeição técnica |
| Figuras ilegíveis / texto minúsculo | Nota baixa em "presentation" |
| Referências desatualizadas ou fora do escopo de robótica | Nota baixa em "related work" |
| Uso de LLM não declarado gerando texto/resultados falsos | Violação de ética **[VERIFICAR política de IA generativa do SBC/IEEE 2026]** |

---

## 10. Prompt pronto para outra IA escrever o artigo

Cole isto (junto com este `.md`) no agente que vai redigir:

```
Você vai escrever um artigo científico completo em LaTeX para o SBR 2026
(Brazilian Symposium on Robotics, João Pessoa, 24-27/11/2026).

RESTRIÇÕES OBRIGATÓRIAS:
- Idioma: inglês.
- Classe: \documentclass[conference]{IEEEtran}; duas colunas; NÃO alterar margens/fontes.
- Máximo ABSOLUTO de 6 páginas incluindo figuras, tabelas e referências.
- REVISÃO DOUBLE-BLIND: nenhum nome de autor, e-mail, afiliação, cidade,
  laboratório, agência de fomento, número de processo, agradecimento ou link
  identificável pode aparecer. Autocitações sempre em TERCEIRA PESSOA.
- Bloco de autor: "Anonymous Author(s) / Affiliation omitted for double-blind review".
- Metadados do PDF limpos via \hypersetup{pdfauthor={}, pdftitle={SBR 2026 Submission}}.
- Estrutura: Abstract, Keywords, I. Introduction (com lista numerada de
  contribuições), II. Related Work, III. Methodology, IV. Experimental Setup,
  V. Results and Discussion (com ablação e limitações), VI. Conclusion, References.
- Referências no estilo IEEE (\bibliographystyle{IEEEtran}), 15-30 itens,
  maioria dos últimos 5 anos, com abreviações IEEE dos veículos.
- Figuras: \columnwidth ou \textwidth (figure*), PDF vetorial, legenda abaixo.
- Tabelas: booktabs, legenda acima, "TABLE I".
- O artigo deve apresentar RESULTADOS NOVOS E SÓLIDOS com comparação
  quantitativa contra pelo menos um baseline da literatura.

ENTREGÁVEIS: main.tex compilável, references.bib, e uma lista das figuras
que precisam ser produzidas (com descrição do conteúdo de cada uma).

TEMA / DADOS DO TRABALHO: <preencher>
```

---

## 11. Arquivos desta pasta

| Arquivo | Uso |
|---|---|
| `GUIA-SBR2026-ARTIGO.md` | Este documento — regras completas |
| `main.tex` | Esqueleto `.tex` pronto, já anonimizado, com comentários explicativos |
| `references.bib` | Exemplo de `.bib` no padrão IEEE |
| `CHECKLIST-SUBMISSAO.md` | Checklist final imprimível |
| `check_double_blind.ps1` | Script que varre `.tex`/`.bib`/PDF procurando vazamentos |

O `main.tex` **compila sozinho**, sem nenhum arquivo de imagem: as figuras estão
como caixas-placeholder (`\figph{...}`). Testado com MiKTeX:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
→ 3 páginas, sem erros, sem warnings de citação. Ao inserir as figuras reais,
troque `\figph{fig/arch.pdf}` pela linha `\includegraphics` já comentada logo acima.

---

## 12. Fontes

- Site oficial SBR 2026 — Call for Papers: https://www.natalnet.br/sbr2026/
- Sistema de submissão JEMS3: https://jems3.sbc.org.br/sbr2026
- Template Overleaf IEEE Conference: https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn
- Template LaTeX IEEE (zip): https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/conference-latex-template_10-17-19.zip
- Template Word IEEE (A4): https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-template-a4.docx
- Robótica / LARS-SBR (call histórico): https://robotica.robocup.org.br/
