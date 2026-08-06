# Varredura de calibracao Mes 2 (ADWIN/PH baseline-relative)

Stream: ordenado por (year, match_id, traj_id), horizonte 30->15.

Baseline: 2019 do PROPRIO modelo (MAD escalonado).

## Tabela de alarmes

| model | detector | param | param_val | K_mad | thr_mm | n_stream | min_instances | n_alarms | rate_per_1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seq2Seq | ADWIN | delta | 0.0010 | nan | nan | 288000 | 200 | 355 | 1.2326 |
| Seq2Seq | ADWIN | delta | 0.0001 | nan | nan | 288000 | 200 | 276 | 0.9583 |
| Seq2Seq | ADWIN | delta | 0.0000 | nan | nan | 288000 | 200 | 220 | 0.7639 |
| Seq2Seq | ADWIN | delta | 0.0000 | nan | nan | 288000 | 200 | 190 | 0.6597 |
| Seq2Seq | PH | K_thr | 10.0000 | 10.0000 | 16.8204 | 288000 | 600 | 1053 | 3.6562 |
| Seq2Seq | PH | K_thr | 20.0000 | 20.0000 | 33.6408 | 288000 | 600 | 793 | 2.7535 |
| Seq2Seq | PH | K_thr | 30.0000 | 30.0000 | 50.4612 | 288000 | 600 | 609 | 2.1146 |
| Seq2Seq | PH | K_thr | 50.0000 | 50.0000 | 84.1020 | 288000 | 600 | 424 | 1.4722 |
| Seq2Seq | PH | K_thr | 80.0000 | 80.0000 | 134.5632 | 288000 | 600 | 304 | 1.0556 |
| Seq2Seq | PH | K_thr | 120.0000 | 120.0000 | 201.8448 | 288000 | 600 | 228 | 0.7917 |
| Kalman | ADWIN | delta | 0.0010 | nan | nan | 288000 | 200 | 421 | 1.4618 |
| Kalman | ADWIN | delta | 0.0001 | nan | nan | 288000 | 200 | 327 | 1.1354 |
| Kalman | ADWIN | delta | 0.0000 | nan | nan | 288000 | 200 | 261 | 0.9062 |
| Kalman | ADWIN | delta | 0.0000 | nan | nan | 288000 | 200 | 239 | 0.8299 |
| Kalman | PH | K_thr | 10.0000 | 10.0000 | 65.1525 | 288000 | 600 | 1007 | 3.4965 |
| Kalman | PH | K_thr | 20.0000 | 20.0000 | 130.3051 | 288000 | 600 | 743 | 2.5799 |
| Kalman | PH | K_thr | 30.0000 | 30.0000 | 195.4576 | 288000 | 600 | 607 | 2.1076 |
| Kalman | PH | K_thr | 50.0000 | 50.0000 | 325.7627 | 288000 | 600 | 447 | 1.5521 |
| Kalman | PH | K_thr | 80.0000 | 80.0000 | 521.2203 | 288000 | 600 | 352 | 1.2222 |
| Kalman | PH | K_thr | 120.0000 | 120.0000 | 781.8305 | 288000 | 600 | 275 | 0.9549 |

## Recomendacao default (apos varredura)

- ADWIN delta = 1e-5 (z_crit ~ 4.94)
- PH K_thr   = 50 MAD
- PH K_delta = 0.5 MAD
- min_instances = 600 (Seq2Seq/Kalman 290k); 100-300 em sample
- cooldown      = 200

Justificativa: na varredura, K=50 produz uma quantidade
balanceada de alarmes entre Seq2Seq e Kalman (sem o vies
de threshold absoluto que dava 6619 PH em Kalman vs 721 em Seq2Seq).
