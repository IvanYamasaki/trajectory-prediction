# Mes 11 — A1: validacao com holdout (split por jogos)

Calibracao: jogos de indice par por ano. Teste: jogos de indice impar
(2019/21/23/24: 3+3 jogos; 2022/25: 3+2). A config vencedora do grid
de calibracao e avaliada UMA vez no teste.

## Resultados no stream de TESTE (out-of-sample)

| variant   | model          | signal   | role         | config                                |   FAR_calib |   cov_calib |   wilson_lo_per1k |   wilson_hi_per1k |   n_alarms |   n_2019_alarms |   n_post_alarms |   FAR_2019_per1k |   year_coverage |   SNR_smooth |
|:----------|:---------------|:---------|:-------------|:--------------------------------------|------------:|------------:|------------------:|------------------:|-----------:|----------------:|----------------:|-----------------:|----------------:|-------------:|
| exact     | Kalman         | raw      | winner_calib | delta=1e-07, mw=200, cd=1000, xw=5000 |      0.25   |           5 |            0.256  |            0.8206 |         47 |              11 |              36 |           0.4583 |               5 |        0.712 |
| exact     | Kalman         | robz     | winner_calib | delta=1e-07, mw=1000, cd=200, xw=5000 |      0.1667 |           5 |            0.1413 |            0.602  |         30 |               7 |              23 |           0.2917 |               5 |        0.692 |
| exact     | Kalman         | robz     | paper_cfg    | delta=1e-07, mw=1000, cd=200, xw=2000 |    nan      |         nan |            0.089  |            0.4877 |         27 |               5 |              22 |           0.2083 |               5 |        0.885 |
| exact     | Seq2Seq        | raw      | winner_calib | delta=1e-07, mw=400, cd=1000, xw=2000 |      0.25   |           5 |            0.1973 |            0.7126 |         36 |               9 |              27 |           0.375  |               5 |        0.646 |
| exact     | Seq2Seq        | robz     | winner_calib | delta=1e-04, mw=600, cd=200, xw=2000  |      0.1667 |           5 |            0.1973 |            0.7126 |         56 |               9 |              47 |           0.375  |               5 |        1.108 |
| exact     | Seq2Seq        | robz     | paper_cfg    | delta=1e-07, mw=200, cd=1000, xw=2000 |    nan      |         nan |            0.0425 |            0.3675 |         15 |               3 |              12 |           0.125  |               4 |        0.75  |
| lite      | Kalman         | robz     | winner_calib | delta=1e-07, mw=200, cd=500, xw=5000  |      0.0833 |           5 |            0.0648 |            0.4285 |         29 |               4 |              25 |           0.1667 |               5 |        1.2   |
| lite      | Kalman         | robz     | paper_cfg    | delta=1e-07, mw=200, cd=1000, xw=5000 |    nan      |         nan |            0.0648 |            0.4285 |         24 |               4 |              20 |           0.1667 |               5 |        0.969 |
| lite      | Seq2Seq        | robz     | winner_calib | delta=1e-04, mw=1000, cd=500, xw=5000 |      0.125  |           4 |            0.0074 |            0.236  |         12 |               1 |              11 |           0.0417 |               5 |        1.385 |
| lite      | Seq2Seq        | robz     | paper_cfg    | delta=1e-07, mw=600, cd=200, xw=2000  |    nan      |         nan |            0      |            0.16   |         12 |               0 |              12 |           0      |               2 |        3     |
| exact     | OR(S2S,Kalman) | robz     | or_channel   | uniao dos vencedores                  |    nan      |         nan |            0.4104 |            1.0827 |         86 |              16 |              70 |           0.6667 |               5 |        0.964 |
| lite      | OR(S2S,Kalman) | robz     | or_channel   | uniao dos vencedores                  |    nan      |         nan |            0.089  |            0.4877 |         41 |               5 |              36 |           0.2083 |               5 |        1.423 |

## Alarmes por ano (teste)

| variant   | model          | role         |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|:----------|:---------------|:-------------|---------:|---------:|---------:|---------:|---------:|---------:|
| exact     | Kalman         | winner_calib |       11 |        7 |        6 |        6 |       10 |        7 |
| exact     | Kalman         | winner_calib |        7 |        7 |        4 |        5 |        6 |        1 |
| exact     | Kalman         | paper_cfg    |        5 |        5 |        5 |        4 |        6 |        2 |
| exact     | Seq2Seq        | winner_calib |        9 |        8 |        6 |        2 |        7 |        4 |
| exact     | Seq2Seq        | winner_calib |        9 |       10 |        9 |       10 |       12 |        6 |
| exact     | Seq2Seq        | paper_cfg    |        3 |        5 |        3 |        2 |        2 |        0 |
| lite      | Kalman         | winner_calib |        4 |       10 |        3 |        4 |        7 |        1 |
| lite      | Kalman         | paper_cfg    |        4 |        6 |        3 |        4 |        6 |        1 |
| lite      | Seq2Seq        | winner_calib |        1 |        6 |        1 |        2 |        1 |        1 |
| lite      | Seq2Seq        | paper_cfg    |        0 |       10 |        0 |        0 |        2 |        0 |
| exact     | OR(S2S,Kalman) | or_channel   |       16 |       17 |       13 |       15 |       18 |        7 |
| lite      | OR(S2S,Kalman) | or_channel   |        5 |       16 |        4 |        6 |        8 |        2 |

## Latencia por ano (teste)

| variant   | model          | channel   |   year |   delay_trajs |   delay_games |   delay_pct_year |   n_alarms_year |
|:----------|:---------------|:----------|-------:|--------------:|--------------:|-----------------:|----------------:|
| exact     | Kalman         | winner    |   2021 |          8666 |          1.08 |             36.1 |               7 |
| exact     | Kalman         | winner    |   2022 |          4483 |          0.56 |             28   |               4 |
| exact     | Kalman         | winner    |   2023 |           876 |          0.11 |              3.6 |               5 |
| exact     | Kalman         | winner    |   2024 |          8857 |          1.11 |             36.9 |               6 |
| exact     | Kalman         | winner    |   2025 |          1826 |          0.23 |             11.4 |               1 |
| exact     | Seq2Seq        | winner    |   2021 |          1238 |          0.15 |              5.2 |              10 |
| exact     | Seq2Seq        | winner    |   2022 |           685 |          0.09 |              4.3 |               9 |
| exact     | Seq2Seq        | winner    |   2023 |          2946 |          0.37 |             12.3 |              10 |
| exact     | Seq2Seq        | winner    |   2024 |          2233 |          0.28 |              9.3 |              12 |
| exact     | Seq2Seq        | winner    |   2025 |           686 |          0.09 |              4.3 |               6 |
| lite      | Kalman         | winner    |   2021 |          8843 |          1.11 |             36.8 |              10 |
| lite      | Kalman         | winner    |   2022 |          4500 |          0.56 |             28.1 |               3 |
| lite      | Kalman         | winner    |   2023 |          1740 |          0.22 |              7.2 |               4 |
| lite      | Kalman         | winner    |   2024 |          6044 |          0.76 |             25.2 |               7 |
| lite      | Kalman         | winner    |   2025 |          2452 |          0.31 |             15.3 |               1 |
| lite      | Seq2Seq        | winner    |   2021 |          7432 |          0.93 |             31   |               6 |
| lite      | Seq2Seq        | winner    |   2022 |          7994 |          1    |             50   |               1 |
| lite      | Seq2Seq        | winner    |   2023 |          4316 |          0.54 |             18   |               2 |
| lite      | Seq2Seq        | winner    |   2024 |         11564 |          1.45 |             48.2 |               1 |
| lite      | Seq2Seq        | winner    |   2025 |          7806 |          0.98 |             48.8 |               1 |
| exact     | OR(S2S,Kalman) | or        |   2021 |          1238 |          0.15 |              5.2 |              17 |
| exact     | OR(S2S,Kalman) | or        |   2022 |           685 |          0.09 |              4.3 |              13 |
| exact     | OR(S2S,Kalman) | or        |   2023 |           876 |          0.11 |              3.6 |              15 |
| exact     | OR(S2S,Kalman) | or        |   2024 |          2233 |          0.28 |              9.3 |              18 |
| exact     | OR(S2S,Kalman) | or        |   2025 |           686 |          0.09 |              4.3 |               7 |
| lite      | OR(S2S,Kalman) | or        |   2021 |          7432 |          0.93 |             31   |              16 |
| lite      | OR(S2S,Kalman) | or        |   2022 |          4500 |          0.56 |             28.1 |               4 |
| lite      | OR(S2S,Kalman) | or        |   2023 |          1740 |          0.22 |              7.2 |               6 |
| lite      | OR(S2S,Kalman) | or        |   2024 |          6044 |          0.76 |             25.2 |               8 |
| lite      | OR(S2S,Kalman) | or        |   2025 |          2452 |          0.31 |             15.3 |               2 |

Nota: o pre-processamento robz_w200 foi FIXADO a partir da Fase 1
(selecao in-sample no stream completo); o braco `signal=raw` (exact)
quantifica quanto o robz vale fora da amostra.