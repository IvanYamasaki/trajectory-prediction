# Mes 10 — Ensemble AND de mesma escala temporal

Sinais: ADE Seq2Seq e accel_p95 por trajetoria (janela de entrada),
ambos robz_w200, por trajetoria (n=288,000).

- ADWIN_ADE (config final): FAR=0.0417/1k, cov=5/5, n_post=23
- ADWIN_ACC (melhor do grid, `d1e-07_mw200_cd200_xw2000`): FAR=0.2708/1k, cov=5/5, n_post=101
- **Delta_min = 3 trajetorias** (Fase 3, escalas distintas: Delta_min=214)

## Sweep W (AND) + OR

| Ensemble | W | n_conf | FAR/1k | year_cov | SNR_smooth |
|----------|--:|-------:|-------:|---------:|-----------:|
| AND | 50 | 4 | 0.0208 | 3/5 | 0.400 |
| AND | 100 | 4 | 0.0208 | 3/5 | 0.400 |
| AND | 200 | 7 | 0.0208 | 3/5 | 0.700 |
| AND | 500 | 9 | 0.0208 | 3/5 | 0.900 |
| AND | 1000 | 9 | 0.0208 | 3/5 | 0.900 |
| AND | 2000 | 10 | 0.0208 | 4/5 | 1.000 |
| OR | 0 | 139 | 0.3125 | 5/5 | 1.562 |
| ADE_standalone | 0 | 25 | 0.0417 | 5/5 | 1.600 |
| ACC_standalone | 0 | 114 | 0.2708 | 5/5 | 1.457 |