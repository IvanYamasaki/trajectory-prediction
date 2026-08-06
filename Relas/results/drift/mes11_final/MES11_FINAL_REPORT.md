# Mes 11 — Master table final (A2+A4)

Decisao do primario pelos numeros OUT-OF-SAMPLE do holdout (regra: cobertura DESC, FAR ASC, SNR_smooth DESC).

| detector                                     | config                                | escopo                      |   FAR_2019_per1k | IC_Wilson_per1k   |   year_coverage |   SNR_smooth |   n_post | latencia_jogos   | admissivel   | papel_final                                                                            |
|:---------------------------------------------|:--------------------------------------|:----------------------------|-----------------:|:------------------|----------------:|-------------:|---------:|:-----------------|:-------------|:---------------------------------------------------------------------------------------|
| ADWIN_lite (S2S)                             | delta=1e-04, mw=1000, cd=500, xw=5000 | teste (out-of-sample)       |           0.0417 | [0.0074; 0.236]   |               5 |        1.385 |       11 | 0.54-1.45        | True         | Detetor primario recomendado                                                           |
| ADWIN_exact (S2S)                            | delta=1e-04, mw=600, cd=200, xw=2000  | teste (out-of-sample)       |           0.375  | [0.1973; 0.7126]  |               5 |        1.108 |       47 | 0.09-0.37        | False        | Robustez de implementacao                                                              |
| OR(S2S, Kalman) lite                         | uniao dos vencedores                  | teste (out-of-sample)       |           0.2083 | [0.089; 0.4877]   |               5 |        1.423 |       36 | 0.22-0.93        | False        | OR inadmissivel no teste (FAR > 0,20/1k)                                               |
| AND(ADE, accel_p95) W=200                    | dois ADWIN por trajetoria, robz_w200  | stream completo (in-sample) |           0.0208 |                   |               3 |        0.7   |        6 |                  | True         | Alta especificidade (custo: cobertura)                                                 |
| Negativos: KSWIN; PH_agg; AND(ADWIN, PH_agg) | ver apendice                          | stream completo (in-sample) |         nan      |                   |               0 |      nan     |        0 |                  | False        | KSWIN inadmissivel (0,438/1k); PH_agg so 2021; AND entre escalas vazio (Delta_min=214) |

## Candidatos a primario considerados (teste, S2S, robz)

| variant   | config                                |   FAR_2019_per1k |   wilson_lo_per1k |   wilson_hi_per1k |   year_coverage |   SNR_smooth |
|:----------|:--------------------------------------|-----------------:|------------------:|------------------:|----------------:|-------------:|
| lite      | delta=1e-04, mw=1000, cd=500, xw=5000 |           0.0417 |            0.0074 |            0.236  |               5 |        1.385 |
| exact     | delta=1e-04, mw=600, cd=200, xw=2000  |           0.375  |            0.1973 |            0.7126 |               5 |        1.108 |