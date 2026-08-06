# Mes 11 — Bateria de detectores dirigidos pelo diagnostico (accel)

Motivacao: tab:iw_feat mostra accel_p99/p90 como vetor dominante do
covariate shift. Aqui: detectores ADWIN+robz sobre features da janela
de entrada (model-free), holdout 2-fold, vs o detector de erro (ADE).

## Resultados no teste (out-of-sample)

| fold   | detector             | family   | config                                |   FAR_calib |   cov_calib |   wilson_lo_per1k |   wilson_hi_per1k |   n_alarms |   n_2019_alarms |   n_post_alarms |   FAR_2019_per1k |   year_coverage |   SNR_smooth |
|:-------|:---------------------|:---------|:--------------------------------------|------------:|------------:|------------------:|------------------:|-----------:|----------------:|----------------:|-----------------:|----------------:|-------------:|
| A      | ADE (erro do modelo) | erro     | delta=1e-04, mw=1000, cd=500, xw=5000 |      0.125  |           4 |            0.0074 |            0.236  |         12 |               1 |              11 |           0.0417 |               5 |        1.385 |
| A      | accel_p95            | feature  | delta=1e-07, mw=200, cd=200, xw=2000  |      0.25   |           5 |            0.1413 |            0.602  |         55 |               7 |              48 |           0.2917 |               5 |        1.413 |
| A      | accel_mean           | feature  | delta=1e-07, mw=200, cd=1000, xw=2000 |      0.2083 |           5 |            0.1413 |            0.602  |         38 |               7 |              31 |           0.2917 |               5 |        0.923 |
| A      | speed_p95            | feature  | delta=1e-07, mw=200, cd=1000, xw=2000 |      0.375  |           5 |            0.1689 |            0.6577 |         47 |               8 |              39 |           0.3333 |               5 |        1.026 |
| A      | speed_mean           | feature  | delta=1e-06, mw=200, cd=1000, xw=2000 |      0.4167 |           5 |            0.1973 |            0.7126 |         54 |               9 |              45 |           0.375  |               5 |        1.062 |
| A      | AND(ADE, accel_p95)  | ensemble | gate fase3, W=200, deb=5000           |    nan      |         nan |            0      |            0.16   |          3 |               0 |               3 |           0      |               3 |        0.923 |
| A      | OR(ADE, accel_p95)   | ensemble | uniao                                 |    nan      |         nan |            0.1689 |            0.6577 |         66 |               8 |              58 |           0.3333 |               5 |        1.513 |
| B      | ADE (erro do modelo) | erro     | delta=1e-04, mw=1000, cd=200, xw=5000 |      0.0417 |           5 |            0.0425 |            0.3675 |         16 |               3 |              13 |           0.125  |               4 |        0.618 |
| B      | accel_p95            | feature  | delta=1e-07, mw=600, cd=200, xw=2000  |      0.2083 |           5 |            0.1146 |            0.5454 |         57 |               6 |              51 |           0.25   |               5 |        1.311 |
| B      | accel_mean           | feature  | delta=1e-06, mw=600, cd=200, xw=2000  |      0.2917 |           5 |            0.2861 |            0.8738 |         67 |              12 |              55 |           0.5    |               5 |        0.76  |
| B      | speed_p95            | feature  | delta=1e-07, mw=600, cd=1000, xw=2000 |      0.2917 |           5 |            0.1973 |            0.7126 |         57 |               9 |              48 |           0.375  |               5 |        0.865 |
| B      | speed_mean           | feature  | delta=1e-07, mw=200, cd=200, xw=2000  |      0.2917 |           5 |            0.5069 |            1.2362 |        100 |              19 |              81 |           0.7917 |               5 |        0.724 |
| B      | AND(ADE, accel_p95)  | ensemble | gate fase3, W=200, deb=5000           |    nan      |         nan |            0      |            0.16   |          2 |               0 |               2 |           0      |               2 |        0.529 |
| B      | OR(ADE, accel_p95)   | ensemble | uniao                                 |    nan      |         nan |            0.1973 |            0.7126 |         73 |               9 |              64 |           0.375  |               5 |        1.147 |

## Alarmes por ano (teste)

| fold   | detector             |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|:-------|:---------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| A      | ADE (erro do modelo) |        1 |        6 |        1 |        2 |        1 |        1 |
| A      | accel_p95            |        7 |       20 |        5 |       10 |        7 |        6 |
| A      | accel_mean           |        7 |       10 |        2 |        7 |        7 |        5 |
| A      | speed_p95            |        8 |       10 |        5 |        9 |        8 |        7 |
| A      | speed_mean           |        9 |       11 |        6 |       10 |        9 |        9 |
| A      | AND(ADE, accel_p95)  |        0 |        1 |        0 |        1 |        1 |        0 |
| A      | OR(ADE, accel_p95)   |        8 |       25 |        6 |       12 |        8 |        7 |
| B      | ADE (erro do modelo) |        3 |        5 |        1 |        6 |        1 |        0 |
| B      | accel_p95            |        6 |       13 |        6 |       24 |        4 |        4 |
| B      | accel_mean           |       12 |       12 |        4 |       27 |       10 |        2 |
| B      | speed_p95            |        9 |        8 |        8 |       13 |       14 |        5 |
| B      | speed_mean           |       19 |       14 |       17 |       29 |       16 |        5 |
| B      | AND(ADE, accel_p95)  |        0 |        1 |        0 |        1 |        0 |        0 |
| B      | OR(ADE, accel_p95)   |        9 |       18 |        7 |       30 |        5 |        4 |

## Latencia (teste)

| fold   | detector             |   year |   delay_trajs |   delay_games |   delay_pct_year |   n_alarms_year |
|:-------|:---------------------|-------:|--------------:|--------------:|-----------------:|----------------:|
| A      | ADE (erro do modelo) |   2021 |          7432 |          0.93 |             31   |               6 |
| A      | ADE (erro do modelo) |   2022 |          7994 |          1    |             50   |               1 |
| A      | ADE (erro do modelo) |   2023 |          4316 |          0.54 |             18   |               2 |
| A      | ADE (erro do modelo) |   2024 |         11564 |          1.45 |             48.2 |               1 |
| A      | ADE (erro do modelo) |   2025 |          7806 |          0.98 |             48.8 |               1 |
| A      | accel_p95            |   2021 |          8638 |          1.08 |             36   |              20 |
| A      | accel_p95            |   2022 |          1349 |          0.17 |              8.4 |               5 |
| A      | accel_p95            |   2023 |          1010 |          0.13 |              4.2 |              10 |
| A      | accel_p95            |   2024 |          8795 |          1.1  |             36.6 |               7 |
| A      | accel_p95            |   2025 |          1259 |          0.16 |              7.9 |               6 |
| A      | accel_mean           |   2021 |          8824 |          1.1  |             36.8 |              10 |
| A      | accel_mean           |   2022 |          5249 |          0.66 |             32.8 |               2 |
| A      | accel_mean           |   2023 |           996 |          0.12 |              4.2 |               7 |
| A      | accel_mean           |   2024 |          4289 |          0.54 |             17.9 |               7 |
| A      | accel_mean           |   2025 |           440 |          0.06 |              2.8 |               5 |
| A      | speed_p95            |   2021 |           430 |          0.05 |              1.8 |              10 |
| A      | speed_p95            |   2022 |          5256 |          0.66 |             32.9 |               5 |
| A      | speed_p95            |   2023 |          1435 |          0.18 |              6   |               9 |
| A      | speed_p95            |   2024 |          9472 |          1.18 |             39.5 |               8 |
| A      | speed_p95            |   2025 |          2287 |          0.29 |             14.3 |               7 |
| A      | speed_mean           |   2021 |           688 |          0.09 |              2.9 |              11 |
| A      | speed_mean           |   2022 |          2814 |          0.35 |             17.6 |               6 |
| A      | speed_mean           |   2023 |          1383 |          0.17 |              5.8 |              10 |
| A      | speed_mean           |   2024 |          8677 |          1.08 |             36.2 |               9 |
| A      | speed_mean           |   2025 |          1037 |          0.13 |              6.5 |               9 |
| A      | AND(ADE, accel_p95)  |   2021 |          8839 |          1.1  |             36.8 |               1 |
| A      | AND(ADE, accel_p95)  |   2022 |            -1 |        nan    |            nan   |               0 |
| A      | AND(ADE, accel_p95)  |   2023 |          6588 |          0.82 |             27.4 |               1 |
| A      | AND(ADE, accel_p95)  |   2024 |         11622 |          1.45 |             48.4 |               1 |
| A      | AND(ADE, accel_p95)  |   2025 |            -1 |        nan    |            nan   |               0 |
| A      | OR(ADE, accel_p95)   |   2021 |          7432 |          0.93 |             31   |              25 |
| A      | OR(ADE, accel_p95)   |   2022 |          1349 |          0.17 |              8.4 |               6 |
| A      | OR(ADE, accel_p95)   |   2023 |          1010 |          0.13 |              4.2 |              12 |
| A      | OR(ADE, accel_p95)   |   2024 |          8795 |          1.1  |             36.6 |               8 |
| A      | OR(ADE, accel_p95)   |   2025 |          1259 |          0.16 |              7.9 |               7 |
| B      | ADE (erro do modelo) |   2021 |          2707 |          0.34 |             11.3 |               5 |
| B      | ADE (erro do modelo) |   2022 |          2588 |          0.24 |              8.1 |               1 |
| B      | ADE (erro do modelo) |   2023 |          9262 |          1.16 |             38.6 |               6 |
| B      | ADE (erro do modelo) |   2024 |          7468 |          0.93 |             31.1 |               1 |
| B      | ADE (erro do modelo) |   2025 |            -1 |        nan    |            nan   |               0 |
| B      | accel_p95            |   2021 |          1041 |          0.13 |              4.3 |              13 |
| B      | accel_p95            |   2022 |          1009 |          0.09 |              3.2 |               6 |
| B      | accel_p95            |   2023 |          2603 |          0.33 |             10.8 |              24 |
| B      | accel_p95            |   2024 |          7674 |          0.96 |             32   |               4 |
| B      | accel_p95            |   2025 |           841 |          0.08 |              2.6 |               4 |
| B      | accel_mean           |   2021 |           501 |          0.06 |              2.1 |              12 |
| B      | accel_mean           |   2022 |          1301 |          0.12 |              4.1 |               4 |
| B      | accel_mean           |   2023 |          9923 |          1.24 |             41.3 |              27 |
| B      | accel_mean           |   2024 |          1509 |          0.19 |              6.3 |              10 |
| B      | accel_mean           |   2025 |           421 |          0.04 |              1.3 |               2 |
| B      | speed_p95            |   2021 |           399 |          0.05 |              1.7 |               8 |
| B      | speed_p95            |   2022 |           388 |          0.04 |              1.2 |               8 |
| B      | speed_p95            |   2023 |          5428 |          0.68 |             22.6 |              13 |
| B      | speed_p95            |   2024 |          1443 |          0.18 |              6   |              14 |
| B      | speed_p95            |   2025 |          1652 |          0.15 |              5.2 |               5 |
| B      | speed_mean           |   2021 |           681 |          0.09 |              2.8 |              14 |
| B      | speed_mean           |   2022 |           390 |          0.04 |              1.2 |              17 |
| B      | speed_mean           |   2023 |          1426 |          0.18 |              5.9 |              29 |
| B      | speed_mean           |   2024 |          2197 |          0.27 |              9.2 |              16 |
| B      | speed_mean           |   2025 |          5113 |          0.48 |             16   |               5 |
| B      | AND(ADE, accel_p95)  |   2021 |         10385 |          1.3  |             43.3 |               1 |
| B      | AND(ADE, accel_p95)  |   2022 |            -1 |        nan    |            nan   |               0 |
| B      | AND(ADE, accel_p95)  |   2023 |         22245 |          2.78 |             92.7 |               1 |
| B      | AND(ADE, accel_p95)  |   2024 |            -1 |        nan    |            nan   |               0 |
| B      | AND(ADE, accel_p95)  |   2025 |            -1 |        nan    |            nan   |               0 |
| B      | OR(ADE, accel_p95)   |   2021 |          1041 |          0.13 |              4.3 |              18 |
| B      | OR(ADE, accel_p95)   |   2022 |          1009 |          0.09 |              3.2 |               7 |
| B      | OR(ADE, accel_p95)   |   2023 |          2603 |          0.33 |             10.8 |              30 |
| B      | OR(ADE, accel_p95)   |   2024 |          7468 |          0.93 |             31.1 |               5 |
| B      | OR(ADE, accel_p95)   |   2025 |           841 |          0.08 |              2.6 |               4 |