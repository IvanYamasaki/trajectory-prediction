# Data Drift em um Modelo Seq2Seq — Predição de Trajetórias na RoboCup SSL

Projeto de Iniciação Científica que investiga **data drift** em um modelo seq2seq (encoder–decoder) de predição de trajetórias de robôs, usando logs oficiais da RoboCup Small Size League de **2019 a 2025**. O trabalho cobre: degradação temporal do modelo, detecção formal de drift em stream (ADWIN, Page-Hinkley, KSWIN, Pelt), quantificação de covariate shift via importance weighting (LSIF) e estratégias de adaptação por fine-tuning seletivo.

O modelo base (arquitetura, treino e dataset original) vem do repositório [LucasSte/trajectory-prediction](https://github.com/LucasSte/trajectory-prediction); este repositório o estende com o pipeline completo de análise de drift.

📄 **Artigo e slides**: veja [`docs/`](docs/).

## Estrutura do repositório

| Pasta | Conteúdo |
|---|---|
| `common/` | Código compartilhado: bootstrap de paths, constantes (`PROC_YEAR`, janelas, configs de detectores), métricas, estilo de plots, utilitários de detecção |
| `dataset/` | Download (`download_dataset.py`), leitura dos logs SSL (protobuf) e suavização Kalman (`process_dataset.py`) |
| `model_analise/` | Arquitetura seq2seq (`ai_model/`), núcleo compartilhado (`core/`), treino, comparação e estratégias de adaptação (EWC, replay, targeted, retrain em breakpoints) |
| `drift_analise/` | Pipeline de drift em 4 capítulos (`chapter01..04_*.py` + notebooks `01..04_*.ipynb`), fases de engenharia de detecção (`phase1..4`, `mes10_*`, `mes11_*`) e figuras do artigo (`paper_figs.py`) |
| `Relas/` | Relatórios LaTeX (`checkpoint/`, `main_pt/`, `main_en/`, `Artigo SBR/`) e resultados gerados (`results/`) |
| `docs/` | **Artigo final (`data-drift-seq2seq-robocup-ssl.pdf`) e slides PT/EN** |
| `tests/` | Testes (pytest) |

## Ambiente

O projeto usa **Python 3.12** (versões mais novas ainda não têm wheels para algumas dependências pinadas, ex.: `ruptures`).

```
python -m venv venv
venv\Scripts\activate        # Windows (ou: source venv/bin/activate)
pip install -r requirements.txt
pip install -r requirements-dev.txt   # opcional: pytest + nbstripout
```

## Dados (RoboCup SSL)

Os 36 jogos usados (6 por ano: 2019, 2021–2025) vêm do acervo oficial de game logs da SSL, hospedado no Seafile da TIGERs Mannheim (link publicado em [ssl.robocup.org/collected-data](https://ssl.robocup.org/collected-data/)).

```
python dataset/download_dataset.py --check   # verifica as 36 URLs (rápido)
python dataset/download_dataset.py           # baixa tudo (~10 GB compactado)
python dataset/download_dataset.py --year 2024   # ou só um ano
python dataset/process_dataset.py            # data_set_N.log -> proc_set_N.pkl
```

O mapeamento jogo → ano está em `common/constants.py` (`PROC_YEAR`); as métricas por jogo, em `drift_analise/dataset/dataset.csv`. Os parâmetros calibrados do suavizador de Kalman (`dataset/*_series_params.pkl`) já estão versionados.

## Modelos

```
python model_analise/train_models.py             # treina e salva pesos + stats de normalização
python model_analise/compare.py                  # compara Seq2Seq / MLP / baseline Kalman
python model_analise/compute_trajectory_errors.py  # gera covariate_shift_out/ (ADE/FDE por trajetória)
```

O seq2seq consome janelas de `[x, y, v_x, v_y, psi]` e prediz `[v_x, v_y]`, integrados para obter a trajetória futura, em duas configurações: 30→15 e 60→30 passos. Detalhes da arquitetura no [repositório original](https://github.com/LucasSte/trajectory-prediction).

**Artefatos não versionados** (gerados pelos passos acima): pesos `weights/*.h5`, stats `model/*.pkl`, `dataset/proc_set_*` e `covariate_shift_out/`. Os resultados finais (figuras PT/EN e CSVs) já estão pré-computados em `Relas/results/`.

## Pipeline de análise de drift

Quatro capítulos, cada um com script (`drift_analise/chapterNN_*.py`) e notebook (`drift_analise/NN_*.ipynb`) pareados; saídas em `Relas/results/drift/<capítulo>/`:

1. **Degradação e diagnóstico** — ADE/FDE por ano, features cinemáticas por jogo, covariate shift descritivo (KS, Wasserstein).
2. **Detecção no stream** — ADWIN, Page-Hinkley e KSWIN online; calibração baseline-relative; decomposição do drift.
3. **Covariate shift e explicação** — importance weighting LSIF/RuLSIF, decomposição covariate vs concept, validação por divisão.
4. **Robustez e validação** — teste nulo por permutação, ARL, BOCPD, Pelt e sensibilidade a tamanho de amostra.

A engenharia de detecção que sustenta o capítulo 2 está em `phase1..4_*.py` (suavizadores → grid de 174 configs → ensembles → consolidação) e nas validações `mes10_*` (latência, independência) e `mes11_*` (holdout out-of-sample, master table final). As estratégias de adaptação (EWC, replay, targeted fine-tuning) estão em `model_analise/*_finetune.py`, com resultados em `Relas/results/mes7*/`.

## Testes

```
python -m pytest tests/ -q
```

## Créditos

- Modelo seq2seq original: [Lucas Steuernagel](https://github.com/LucasSte/trajectory-prediction)
- Game logs: [RoboCup Small Size League](https://ssl.robocup.org/) / acervo mantido pela TIGERs Mannheim
