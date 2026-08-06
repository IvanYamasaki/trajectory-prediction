# drift_analise/legacy — Código e resultados arquivados

Este diretório contém métodos descontinuados do projeto de IC sobre data drift
em Seq2Seq de predição de trajetórias no RoboCup SSL. Os arquivos aqui são
mantidos **exclusivamente para auditoria e reprodutibilidade histórica**.
Não os use para produzir resultados novos.

---

## iw_decomposition_mes2_matching.py

**O que faz:** decomposição covariate vs concept drift por *matching de jogos*
compatíveis entre o ano de referência (2019) e cada ano posterior.

**Por que foi arquivado:**

O método exige pares de jogos com características cinemáticas similares
(nº de trajetórias ≈ igual). Com apenas 6 jogos por ano, o dataset produz:

| Ano  | n_compat | Resultado                        |
|------|----------|----------------------------------|
| 2022 | 0        | não estimável                    |
| 2023 | 2        | 87,1 % concept drift (ruidoso)  |
| 2024 | 0        | não estimável                    |
| 2025 | 0        | não estimável                    |

Com n_compat = 2 para 2023 e 0 para todos os demais anos, a estimativa é
instável e não generalizável. O resultado de 87,1 % concept drift para 2023
contrasta com o LSIF do Mês 3 (recovery covariate ≈ 71,8 % para 2023),
indicando que o matching superestima o componente concept por falta de controle
adequado de distribuição.

**Resultado oficial:** o método LSIF do Mês 3 substituiu este.
- Script: `drift_analise/chapter03_visuals.py` (`main_compute_importance_weights`)
- Resultado: `Relas/results/mes3/iw_decomposition.csv`
- Recovery mediana IW: 65,5 % (IC 95 %: 59–72 %)

**Como reproduzir o resultado histórico (somente auditoria):**

```bash
# Exibe o CSV histórico sem re-executar nada
python drift_analise/legacy/iw_decomposition_mes2_matching.py

# Re-executa o pareamento do zero (pode dar resultado diferente
# do CSV histórico se o código original divergia desta reconstrução)
python drift_analise/legacy/iw_decomposition_mes2_matching.py --recompute
```

**Nota sobre proveniência:** o script original que gerou
`Relas/results/mes2/drift_decomposition.csv` não sobreviveu no repositório.
Este arquivo é uma **reconstrução** da lógica a partir do CSV resultante e
da metodologia documentada. O CSV em `outputs/drift_decomposition_mes2.csv`
é uma cópia direta do original em `Relas/results/mes2/`.

---

## outputs/

Cópias dos artefatos gerados pelos scripts deste diretório.

| Arquivo                          | Origem                                        |
|----------------------------------|-----------------------------------------------|
| `drift_decomposition_mes2.csv`   | Cópia de `Relas/results/mes2/drift_decomposition.csv` |

O arquivo original em `Relas/results/mes2/drift_decomposition.csv` é mantido
no lugar porque é lido por `drift_analise/chapter03_visuals.py` e citado
no artigo `Relas/mes1a4/mes1a4.tex` (Seção 4, Mês 2).
