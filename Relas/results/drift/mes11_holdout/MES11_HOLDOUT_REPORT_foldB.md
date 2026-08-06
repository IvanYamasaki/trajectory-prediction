# Mes 11 — A1: validacao com holdout (split por jogos, fold B)

Fold A: calibracao = jogos pares, teste = impares; fold B: invertido.
A config vencedora do grid de calibracao e avaliada UMA vez no teste.

## Resultados no stream de TESTE (out-of-sample)

| fold   | variant   | model          | signal   | role         | config                                |   FAR_calib |   cov_calib |   wilson_lo_per1k |   wilson_hi_per1k |   n_alarms |   n_2019_alarms |   n_post_alarms |   FAR_2019_per1k |   year_coverage |   SNR_smooth |
|:-------|:----------|:---------------|:---------|:-------------|:--------------------------------------|------------:|------------:|------------------:|------------------:|-----------:|----------------:|----------------:|-----------------:|----------------:|-------------:|
| B      | exact     | Kalman         | raw      | winner_calib | delta=1e-07, mw=600, cd=1000, xw=5000 |      0.4167 |           5 |            0.1146 |            0.5454 |         56 |               6 |              50 |           0.25   |               5 |        1.286 |
| B      | exact     | Kalman         | robz     | winner_calib | delta=1e-07, mw=1000, cd=200, xw=2000 |      0.2083 |           5 |            0.0648 |            0.4285 |         27 |               4 |              23 |           0.1667 |               5 |        0.847 |
| B      | exact     | Kalman         | robz     | paper_cfg    | delta=1e-07, mw=1000, cd=200, xw=2000 |    nan      |         nan |            0.0648 |            0.4285 |         27 |               4 |              23 |           0.1667 |               5 |        0.847 |
| B      | exact     | Seq2Seq        | raw      | winner_calib | delta=1e-07, mw=1000, cd=200, xw=2000 |      0.2083 |           5 |            0.1146 |            0.5454 |         36 |               6 |              30 |           0.25   |               5 |        0.782 |
| B      | exact     | Seq2Seq        | robz     | winner_calib | delta=1e-05, mw=200, cd=1000, xw=2000 |      0.0833 |           5 |            0.089  |            0.4877 |         27 |               5 |              22 |           0.2083 |               4 |        0.676 |
| B      | exact     | Seq2Seq        | robz     | paper_cfg    | delta=1e-07, mw=200, cd=1000, xw=2000 |    nan      |         nan |            0.0229 |            0.3038 |         16 |               2 |              14 |           0.0833 |               3 |        0.882 |
| B      | lite      | Kalman         | robz     | winner_calib | delta=1e-07, mw=200, cd=200, xw=5000  |      0.1667 |           5 |            0.0229 |            0.3038 |         25 |               2 |              23 |           0.0833 |               5 |        1.412 |
| B      | lite      | Kalman         | robz     | paper_cfg    | delta=1e-07, mw=200, cd=1000, xw=5000 |    nan      |         nan |            0.0229 |            0.3038 |         23 |               2 |              21 |           0.0833 |               5 |        1.294 |
| B      | lite      | Seq2Seq        | robz     | winner_calib | delta=1e-04, mw=1000, cd=200, xw=5000 |      0.0417 |           5 |            0.0425 |            0.3675 |         16 |               3 |              13 |           0.125  |               4 |        0.618 |
| B      | lite      | Seq2Seq        | robz     | paper_cfg    | delta=1e-07, mw=600, cd=200, xw=2000  |    nan      |         nan |            0.0229 |            0.3038 |         12 |               2 |              10 |           0.0833 |               3 |        0.647 |
| B      | exact     | OR(S2S,Kalman) | robz     | or_channel   | uniao dos vencedores                  |    nan      |         nan |            0.1973 |            0.7126 |         54 |               9 |              45 |           0.375  |               5 |        0.812 |
| B      | lite      | OR(S2S,Kalman) | robz     | or_channel   | uniao dos vencedores                  |    nan      |         nan |            0.089  |            0.4877 |         41 |               5 |              36 |           0.2083 |               5 |        1.088 |

## Alarmes por ano (teste)

| variant   | model          | role         |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|:----------|:---------------|:-------------|---------:|---------:|---------:|---------:|---------:|---------:|
| exact     | Kalman         | winner_calib |        6 |        9 |       10 |       12 |       10 |        9 |
| exact     | Kalman         | winner_calib |        4 |        4 |        1 |       11 |        6 |        1 |
| exact     | Kalman         | paper_cfg    |        4 |        4 |        1 |       11 |        6 |        1 |
| exact     | Seq2Seq        | winner_calib |        6 |        5 |        6 |        7 |        9 |        3 |
| exact     | Seq2Seq        | winner_calib |        5 |        6 |        1 |        9 |        6 |        0 |
| exact     | Seq2Seq        | paper_cfg    |        2 |        4 |        0 |        6 |        4 |        0 |
| lite      | Kalman         | winner_calib |        2 |        7 |        7 |        6 |        2 |        1 |
| lite      | Kalman         | paper_cfg    |        2 |        7 |        7 |        4 |        2 |        1 |
| lite      | Seq2Seq        | winner_calib |        3 |        5 |        1 |        6 |        1 |        0 |
| lite      | Seq2Seq        | paper_cfg    |        2 |        4 |        1 |        5 |        0 |        0 |
| exact     | OR(S2S,Kalman) | or_channel   |        9 |       10 |        2 |       20 |       12 |        1 |
| lite      | OR(S2S,Kalman) | or_channel   |        5 |       12 |        8 |       12 |        3 |        1 |

## Latencia por ano (teste)

| fold   | variant   | model          | channel   |   year |   delay_trajs |   delay_games |   delay_pct_year |   n_alarms_year |
|:-------|:----------|:---------------|:----------|-------:|--------------:|--------------:|-----------------:|----------------:|
| B      | exact     | Kalman         | winner    |   2021 |           103 |          0.01 |              0.4 |               4 |
| B      | exact     | Kalman         | winner    |   2022 |         15176 |          1.42 |             47.4 |               1 |
| B      | exact     | Kalman         | winner    |   2023 |           944 |          0.12 |              3.9 |              11 |
| B      | exact     | Kalman         | winner    |   2024 |          5095 |          0.64 |             21.2 |               6 |
| B      | exact     | Kalman         | winner    |   2025 |         15012 |          1.41 |             46.9 |               1 |
| B      | exact     | Seq2Seq        | winner    |   2021 |            71 |          0.01 |              0.3 |               6 |
| B      | exact     | Seq2Seq        | winner    |   2022 |           896 |          0.08 |              2.8 |               1 |
| B      | exact     | Seq2Seq        | winner    |   2023 |          6834 |          0.85 |             28.5 |               9 |
| B      | exact     | Seq2Seq        | winner    |   2024 |          4967 |          0.62 |             20.7 |               6 |
| B      | exact     | Seq2Seq        | winner    |   2025 |            -1 |        nan    |            nan   |               0 |
| B      | lite      | Kalman         | winner    |   2021 |          2594 |          0.32 |             10.8 |               7 |
| B      | lite      | Kalman         | winner    |   2022 |          7502 |          0.7  |             23.4 |               7 |
| B      | lite      | Kalman         | winner    |   2023 |          1791 |          0.22 |              7.5 |               6 |
| B      | lite      | Kalman         | winner    |   2024 |          2752 |          0.34 |             11.5 |               2 |
| B      | lite      | Kalman         | winner    |   2025 |          1751 |          0.16 |              5.5 |               1 |
| B      | lite      | Seq2Seq        | winner    |   2021 |          2707 |          0.34 |             11.3 |               5 |
| B      | lite      | Seq2Seq        | winner    |   2022 |          2588 |          0.24 |              8.1 |               1 |
| B      | lite      | Seq2Seq        | winner    |   2023 |          9262 |          1.16 |             38.6 |               6 |
| B      | lite      | Seq2Seq        | winner    |   2024 |          7468 |          0.93 |             31.1 |               1 |
| B      | lite      | Seq2Seq        | winner    |   2025 |            -1 |        nan    |            nan   |               0 |
| B      | exact     | OR(S2S,Kalman) | or        |   2021 |            71 |          0.01 |              0.3 |              10 |
| B      | exact     | OR(S2S,Kalman) | or        |   2022 |           896 |          0.08 |              2.8 |               2 |
| B      | exact     | OR(S2S,Kalman) | or        |   2023 |           944 |          0.12 |              3.9 |              20 |
| B      | exact     | OR(S2S,Kalman) | or        |   2024 |          4967 |          0.62 |             20.7 |              12 |
| B      | exact     | OR(S2S,Kalman) | or        |   2025 |         15012 |          1.41 |             46.9 |               1 |
| B      | lite      | OR(S2S,Kalman) | or        |   2021 |          2594 |          0.32 |             10.8 |              12 |
| B      | lite      | OR(S2S,Kalman) | or        |   2022 |          2588 |          0.24 |              8.1 |               8 |
| B      | lite      | OR(S2S,Kalman) | or        |   2023 |          1791 |          0.22 |              7.5 |              12 |
| B      | lite      | OR(S2S,Kalman) | or        |   2024 |          2752 |          0.34 |             11.5 |               3 |
| B      | lite      | OR(S2S,Kalman) | or        |   2025 |          1751 |          0.16 |              5.5 |               1 |

Nota: o pre-processamento robz_w200 foi FIXADO a partir da Fase 1
(selecao in-sample no stream completo); o braco `signal=raw` (exact)
quantifica quanto o robz vale fora da amostra.