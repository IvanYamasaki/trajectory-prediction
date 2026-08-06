# Mes 10 — Latencia de detecao no stream real

Convencao: fronteira do ano como proxy do onset (drift gradual sem
changepoint formal) — latencia reportada e limite superior.

Escala: ~8000 trajetorias/jogo; 48k trajetorias/ano.

## ADWIN_Seq2Seq ({'delta': 1e-07, 'min_window': 600, 'cooldown': 200, 'max_window': 2000})

| Ano | Latencia (trajs) | ~jogos | % do bloco do ano | alarmes no ano |
|-----|-----------------:|-------:|------------------:|---------------:|
| 2021 | 19,585 | 2.45 | 40.8% | 11 |
| 2022 | 2,481 | 0.31 | 5.2% | 1 |
| 2023 | 22,506 | 2.81 | 46.9% | 5 |
| 2024 | 8,401 | 1.05 | 17.5% | 4 |
| 2025 | 16,432 | 2.05 | 34.2% | 2 |

## ADWIN_Kalman ({'delta': 1e-07, 'min_window': 200, 'cooldown': 1000, 'max_window': 5000})

| Ano | Latencia (trajs) | ~jogos | % do bloco do ano | alarmes no ano |
|-----|-----------------:|-------:|------------------:|---------------:|
| 2021 | 2,574 | 0.32 | 5.4% | 14 |
| 2022 | 7,502 | 0.94 | 15.6% | 11 |
| 2023 | 1,791 | 0.22 | 3.7% | 10 |
| 2024 | 2,986 | 0.37 | 6.2% | 9 |
| 2025 | 34,515 | 4.31 | 71.9% | 1 |
