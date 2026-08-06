# Mes 11 — A3: correcoes de consistencia

## (1) FAR da Frente B nas duas unidades

A FAR publicada da Frente B estava por 1000 *janelas* (`n_2019_total // N`); Frentes A/C usam 1000 *trajetorias*.
Config citada no paper (Seq2Seq, N=50, K=50, delta=2.0):

| model   |   N_window |   K_thr |   K_delta |   n_alarms |   n_2019_alarms |   n_post_alarms |   year_coverage |   FAR_2019_per1k_win |   FAR_2019_per1k_traj |
|:--------|-----------:|--------:|----------:|-----------:|----------------:|----------------:|----------------:|---------------------:|----------------------:|
| Seq2Seq |         50 |      50 |         2 |         13 |               0 |              13 |               1 |                    0 |                     0 |

Como o melhor PH_agg tem 0 alarmes em 2019, o headline (FAR=0) nao
muda — mas qualquer comparacao cross-frente com FAR>0 deve usar a
coluna `FAR_2019_per1k_traj`.

## (2) Gate AND unificado (mesma escala)

Delta_min(ADE, ACC) = 3 trajetorias (identico sob ambos os gates,
pois depende so das listas de alarmes).

| gate          |    W |   n_confirmed |   delta_min |   FAR_2019_per1k |   year_coverage |   SNR_smooth |   n_post |   n_2019 |   n_2021 |   n_2022 |   n_2023 |   n_2024 |   n_2025 |
|:--------------|-----:|--------------:|------------:|-----------------:|----------------:|-------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| mes10_nearest |   50 |             4 |           3 |           0.0208 |               3 |          0.4 |        3 |        1 |        1 |        0 |        1 |        1 |        0 |
| phase3_deque  |   50 |             4 |           3 |           0.0208 |               3 |          0.4 |        3 |        1 |        1 |        0 |        1 |        1 |        0 |
| mes10_nearest |  100 |             4 |           3 |           0.0208 |               3 |          0.4 |        3 |        1 |        1 |        0 |        1 |        1 |        0 |
| phase3_deque  |  100 |             4 |           3 |           0.0208 |               3 |          0.4 |        3 |        1 |        1 |        0 |        1 |        1 |        0 |
| mes10_nearest |  200 |             7 |           3 |           0.0208 |               3 |          0.7 |        6 |        1 |        3 |        0 |        1 |        2 |        0 |
| phase3_deque  |  200 |             7 |           3 |           0.0208 |               3 |          0.7 |        6 |        1 |        3 |        0 |        1 |        2 |        0 |
| mes10_nearest |  500 |             9 |           3 |           0.0208 |               3 |          0.9 |        8 |        1 |        3 |        0 |        2 |        3 |        0 |
| phase3_deque  |  500 |             9 |           3 |           0.0208 |               3 |          0.9 |        8 |        1 |        3 |        0 |        2 |        3 |        0 |
| mes10_nearest | 1000 |             9 |           3 |           0.0208 |               3 |          0.9 |        8 |        1 |        3 |        0 |        2 |        3 |        0 |
| phase3_deque  | 1000 |             9 |           3 |           0.0208 |               3 |          0.9 |        8 |        1 |        3 |        0 |        2 |        3 |        0 |
| mes10_nearest | 2000 |            10 |           3 |           0.0208 |               4 |          1   |        9 |        1 |        3 |        1 |        2 |        3 |        0 |
| phase3_deque  | 2000 |            10 |           3 |           0.0208 |               4 |          1   |        9 |        1 |        3 |        1 |        2 |        3 |        0 |

Leitura: se `n_confirmed`/FAR/cobertura coincidem entre `mes10_nearest`
e `phase3_deque`, a conclusao do Mes 10 nao depende da implementacao
do gate; divergencias devem ser reportadas na secao com o gate da
Fase 3 como canonico.