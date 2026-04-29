# Pipeline de Drift - trajectory-prediction

Este documento descreve o pipeline usado pelo notebook `drift_analise/drift_analise.ipynb`.
O notebook consome artefatos prontos; a lógica pesada fica nos scripts
`drift_analise/plot_covariate_shift.py` e
`model_analise/compute_trajectory_errors.py`.

## Fluxograma textual

1. `dataset/proc_set_*.pkl`
2. `drift_analise/plot_covariate_shift.py`
3. `covariate_shift_out/per_game_context.csv`
4. `covariate_shift_out/per_window_drift.csv`
5. `covariate_shift_out/per_trajectory_features.csv`
6. `covariate_shift_out/drift_vs_baseline.csv`
7. `covariate_shift_out/drift_vs_baseline_robust.csv`
8. `covariate_shift_out/ks_wd_per_dataset.csv`
9. `model_analise/compute_trajectory_errors.py`
10. `covariate_shift_out/trajectory_errors_sample.parquet`
11. `drift_analise/drift_analise.ipynb`
12. `Relas/results/mes1/*`
13. `Relas/results/mes2/*`

## Como Rodar Do Zero

```powershell
.\\venv\\Scripts\\python.exe -m pip install -r requirements.txt
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\python.exe drift_analise/plot_covariate_shift.py
$env:PYTHONIOENCODING='utf-8'; .\\venv\\Scripts\\python.exe -u model_analise/compute_trajectory_errors.py --n_per_year 3 --seed 42
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute drift_analise/drift_analise.ipynb --output drift_analise.executed.ipynb --output-dir drift_analise --ExecutePreprocessor.timeout=600
```

## Inputs Imutáveis

- `drift_analise/dataset/dataset.csv`: tabela original de métricas por dataset, modelo, horizonte e ano.
- `drift_analise/dataset/dataset_enriched.csv`: gerado pelo notebook (`dataset.csv` + `per_game_context.csv`).
- `dataset/proc_set_*.pkl`: logs processados com trajetórias de robôs e bola.
- `weights/robot_30_15_t.weights.h5`: pesos Seq2Seq para horizonte 30->15.
- `weights/robot_60_30_t.weights.h5`: pesos Seq2Seq para horizonte 60->30.
- `model/normalization_stats_30_15.pkl`: normalização do treino para 30->15.
- `model/normalization_stats_60_30.pkl`: normalização do treino para 60->30.

## Configuração Auxiliar

- `division_map.csv`: mapa manual `proc_set_file,division`.
- `proc_set_file`: nome do arquivo processado, como `proc_set_3.pkl`.
- `division`: divisão SSL inferida ou mapeada manualmente, `A` ou `B`.
- Logs de 2021 permanecem sem divisão A/B porque a fonte oficial marca participação `Both`.

## Data Dictionary - covariate_shift_out

### `per_game_context.csv`

- `match_id`: identificador curto do jogo.
- `log_file`: nome do log original.
- `proc_set_file`: arquivo processado de origem.
- `year`: ano da competição.
- `division`: divisão A/B quando identificável.
- `avg_robots_per_stoppage`: proxy de trajetórias por stoppage.
- `n_trajectories`: número de trajetórias de robôs extraídas.
- `n_unique_stop_ids`: quantidade de pares `(source, stop_id)`.
- `duration_estimate_s`: soma aproximada de frames dividida por `FRAME_RATE_HZ`.
- `frame_rate_hz`: frame rate usado na estimativa.
- `speed_mean`: média da velocidade.
- `speed_p50`: mediana da velocidade.
- `speed_p90`: percentil 90 da velocidade.
- `speed_p99`: percentil 99 da velocidade.
- `speed_max`: máximo de velocidade observado.
- `accel_mean`: média da aceleração.
- `accel_p50`: mediana da aceleração.
- `accel_p90`: percentil 90 da aceleração.
- `accel_p99`: percentil 99 da aceleração.
- `accel_max`: máximo de aceleração observado.
- `turn_mean`: média da variação angular absoluta.
- `turn_p50`: mediana da variação angular.
- `turn_p90`: percentil 90 da variação angular.

### `per_window_drift.csv`

- `match_id`: identificador curto do jogo.
- `log_file`: nome do log original.
- `proc_set_file`: arquivo processado de origem.
- `year`: ano da competição.
- `window_id`: índice da janela dentro do jogo.
- `n_trajectories_in_window`: quantidade de trajetórias na janela.
- `ks_speed_window`: KS da velocidade contra baseline de treino.
- `ks_accel_window`: KS da aceleração contra baseline de treino.
- `ks_turn_window`: KS da variação angular contra baseline de treino.
- `wd_speed_window`: Wasserstein da velocidade contra baseline.
- `wd_accel_window`: Wasserstein da aceleração contra baseline.
- `wd_turn_window`: Wasserstein da variação angular contra baseline.

### `trajectory_errors_sample.parquet`

- `match_id`: identificador curto do jogo.
- `year`: ano da competição.
- `log_file`: nome do log original.
- `proc_set_file`: arquivo processado de origem.
- `model`: `Seq2Seq` ou `Kalman`.
- `horizon`: horizonte avaliado, `30->15` ou `60->30`.
- `traj_id`: índice da trajetória/janela dentro do arquivo.
- `ade_traj`: ADE por trajetória, em mm.
- `fde_traj`: FDE por trajetória, em mm.
- `n_steps_pred`: número de passos previstos.

### `drift_vs_baseline.csv`

- `year`: ano avaliado.
- `feature`: `vx`, `vy`, `speed`, `accel` ou `turn`.
- `wasserstein`: distância Wasserstein contra baseline.
- `ks`: estatística KS contra baseline.

### `drift_vs_baseline_robust.csv`

- `year`: ano avaliado.
- `feature`: feature cinemática.
- `wasserstein`: distância Wasserstein após clipping robusto.
- `ks`: KS após clipping robusto.
- `clip_q`: quantil de clipping usado.
- `clip_hi`: valor máximo de clipping para a feature.

### `ks_wd_per_dataset.csv`

- `file`: arquivo processado avaliado.
- `year`: ano do arquivo.
- `feature`: feature avaliada.
- `ks`: KS contra baseline.
- `wasserstein`: Wasserstein contra baseline.

## Data Dictionary - Mes 1

### `summary_granularity.csv`

- `year`: ano.
- `n_jogos`: número de jogos.
- `n_trajetorias_total`: total de trajetórias no ano.
- `n_janelas_total`: total de janelas no ano.

### `tail_stats.csv`

- `year`: ano.
- `model`: modelo avaliado.
- `horizon`: horizonte avaliado.
- `p50`: mediana do ADE por trajetória.
- `p90`: percentil 90 do ADE por trajetória.
- `p99`: percentil 99 do ADE por trajetória.
- `n_trajs`: número de trajetórias.

### `window_intra_year_var.csv`

- `year`: ano.
- `ks_speed_window__p50`: mediana intra-ano do KS de velocidade.
- `ks_speed_window__p90`: p90 intra-ano do KS de velocidade.
- `wd_speed_window__p50`: mediana intra-ano do Wasserstein de velocidade.
- `wd_speed_window__p90`: p90 intra-ano do Wasserstein de velocidade.
- `ks_accel_window__p50`: mediana intra-ano do KS de aceleração.
- `ks_accel_window__p90`: p90 intra-ano do KS de aceleração.
- `wd_accel_window__p50`: mediana intra-ano do Wasserstein de aceleração.
- `wd_accel_window__p90`: p90 intra-ano do Wasserstein de aceleração.

## Data Dictionary - Mes 2

### `alarms_consolidated.csv`

- `method`: detector ou método de mudança.
- `drift_type`: tipo de drift (`Erro online`, `Feature online`, `Erro offline`).
- `stream_index`: índice no stream usado pelo detector.
- `year`: ano mapeado do índice.
- `match_id`: jogo mapeado do índice.
- `log_file`: log original mapeado do índice.

### `alarms_summary_by_year.csv`

- `method`: detector ou método.
- `drift_type`: tipo de drift.
- `year`: ano.
- `n_alarmes`: quantidade de alarmes no ano.

### `drift_decomposition.csv`

- `year`: ano avaliado.
- `n_games_full`: jogos disponíveis no ano.
- `n_games_compat`: jogos compatíveis para decomposição.
- `ade_baseline_2019`: ADE baseline de 2019.
- `ade_full`: ADE médio completo do ano.
- `ade_compat`: ADE médio no subconjunto compatível.
- `excess_total_mm`: excesso total de ADE contra baseline.
- `excess_concept_mm`: parcela atribuída a concept drift.
- `excess_covariate_mm`: parcela atribuída a covariate shift.
- `pct_explained_by_concept`: percentual do excesso explicado por concept drift.

### `null_distribution.csv`

- `method`: detector testado contra null.
- `n_alarmes_observados`: alarmes observados.
- `null_mean`: média de alarmes no null.
- `null_p50`: mediana de alarmes no null.
- `null_p95`: percentil 95 de alarmes no null.
- `p_value`: p-valor empírico.

### `latency_curve.csv`

- `detector`: detector avaliado.
- `shift_mm`: shift sintético aplicado.
- `n_detected`: número de simulações detectadas.
- `n_total`: número total de simulações.
- `lat_p50`: latência mediana.
- `lat_p10`: percentil 10 da latência.
- `lat_p90`: percentil 90 da latência.

### `ruptures_elbow.csv`

- `pen`: penalidade Pelt testada.
- `n_bps`: número de breakpoints detectados.

## Figuras - Mes 1

- `2a_window_timeline_ks_accel_window_pt.pdf`, `2a_window_timeline_ks_accel_window_en.pdf`: timeline do KS de aceleração por janela.
- `2a_window_timeline_ks_speed_window_pt.pdf`, `2a_window_timeline_ks_speed_window_en.pdf`: timeline do KS de velocidade por janela.
- `2a_window_timeline_wd_accel_window_pt.pdf`, `2a_window_timeline_wd_accel_window_en.pdf`: timeline do Wasserstein de aceleração por janela.
- `2a_window_timeline_wd_speed_window_pt.pdf`, `2a_window_timeline_wd_speed_window_en.pdf`: timeline do Wasserstein de velocidade por janela.
- `2b_window_box_ks_accel_window_pt.pdf`, `2b_window_box_ks_accel_window_en.pdf`: boxplot intra-ano do KS de aceleração.
- `2b_window_box_ks_speed_window_pt.pdf`, `2b_window_box_ks_speed_window_en.pdf`: boxplot intra-ano do KS de velocidade.
- `2b_window_box_wd_accel_window_pt.pdf`, `2b_window_box_wd_accel_window_en.pdf`: boxplot intra-ano do Wasserstein de aceleração.
- `2b_window_box_wd_speed_window_pt.pdf`, `2b_window_box_wd_speed_window_en.pdf`: boxplot intra-ano do Wasserstein de velocidade.
- `3a_ade_traj_hist_Kalman_30_15_pt.pdf`, `3a_ade_traj_hist_Kalman_30_15_en.pdf`: histograma ADE Kalman 30->15.
- `3a_ade_traj_hist_Kalman_60_30_pt.pdf`, `3a_ade_traj_hist_Kalman_60_30_en.pdf`: histograma ADE Kalman 60->30.
- `3a_ade_traj_hist_Seq2Seq_30_15_pt.pdf`, `3a_ade_traj_hist_Seq2Seq_30_15_en.pdf`: histograma ADE Seq2Seq 30->15.
- `3a_ade_traj_hist_Seq2Seq_60_30_pt.pdf`, `3a_ade_traj_hist_Seq2Seq_60_30_en.pdf`: histograma ADE Seq2Seq 60->30.
- `3c_p99_p50_ratio_pt.pdf`, `3c_p99_p50_ratio_en.pdf`: razão p99/p50 do ADE ao longo dos anos.
- `4a_ade_year_per_game_pt.pdf`, `4a_ade_year_per_game_en.pdf`: ADE por jogo com IC bootstrap.
- `4b_ade_vs_accel_p90_pt.pdf`, `4b_ade_vs_accel_p90_en.pdf`: ADE vs p90 de aceleração.
- `4b_ade_vs_ks_speed_pt.pdf`, `4b_ade_vs_ks_speed_en.pdf`: ADE vs KS de velocidade.
- `4b_ade_vs_speed_p90_pt.pdf`, `4b_ade_vs_speed_p90_en.pdf`: ADE vs p90 de velocidade.
- `4d_corr_spearman_pt.pdf`, `4d_corr_spearman_en.pdf`: heatmap de correlação Spearman.

## Figuras - Mes 2

- `6_concept_drift_adwin_ph_pt.pdf`, `6_concept_drift_adwin_ph_en.pdf`: ADWIN e Page-Hinkley sobre ADE Seq2Seq.
- `6_concept_drift_kalman_pt.pdf`, `6_concept_drift_kalman_en.pdf`: concept drift no erro Kalman.
- `6_concept_drift_seq2seq_pt.pdf`, `6_concept_drift_seq2seq_en.pdf`: concept drift no erro Seq2Seq.
- `7_kswin_timeline_ks_accel_window_pt.pdf`, `7_kswin_timeline_ks_accel_window_en.pdf`: KSWIN sobre KS de aceleração.
- `7_kswin_timeline_ks_speed_window_pt.pdf`, `7_kswin_timeline_ks_speed_window_en.pdf`: KSWIN sobre KS de velocidade.
- `8_pelt_elbow_pt.pdf`, `8_pelt_elbow_en.pdf`: curva de penalidade Pelt.
- `8_ruptures_pelt_rbf_per_game_pt.pdf`, `8_ruptures_pelt_rbf_per_game_en.pdf`: Pelt RBF no ADE médio por jogo.
- `10a_null_distribution_pt.pdf`, `10a_null_distribution_en.pdf`: distribuição nula de alarmes.
- `10b_latency_curve_pt.pdf`, `10b_latency_curve_en.pdf`: curva de latência sintética.
- `11_drift_decomposition_pt.pdf`, `11_drift_decomposition_en.pdf`: decomposição covariate vs concept.

## Data Dictionary - Mes 3

### `iw_decomposition.csv`

- `year`: ano alvo avaliado.
- `ade_y`: ADE médio direto do ano y (mm).
- `ade_y_ci_lo`, `ade_y_ci_hi`: CI95 bootstrap do ADE direto.
- `ade_2019`: ADE baseline 2019 (mm).
- `ade_iw`: ADE importance-weighted de y (ponderado por p_2019(x)/p_y(x)).
- `ade_iw_ci_lo`, `ade_iw_ci_hi`: CI95 bootstrap do ADE_IW.
- `ess_ratio`: Effective Sample Size ratio = (Σw)²/(n·Σw²). < 0.3 = pesos instáveis.
- `ess_stable`: booleano, True se ess_ratio ≥ 0.3.
- `recovery_pct`: % do excesso de ADE recuperado pelo IW = (ADE_y − ADE_IW)/(ADE_y − ADE_2019)×100.
- `n_source`: número de trajetórias no baseline 2019.
- `n_target`: número de trajetórias no ano y.

### `retrain_results.csv`

- `breakpoint_label`: rótulo do breakpoint Pelt (ex. `antes_2022`).
- `breakpoint_idx`: índice numérico do breakpoint.
- `n_train`: janelas de treinamento usadas no fine-tuning.
- `n_test`: janelas de avaliação.
- `ade_before`: ADE médio no conjunto de teste antes do fine-tuning (mm).
- `ade_after`: ADE médio no conjunto de teste após o fine-tuning (mm).
- `delta_mm`: ade_after − ade_before.
- `delta_pct`: variação percentual do ADE.
- `recovery_pct`: % do excesso de ADE recuperado pelo fine-tuning.
- `n_traj_improved`: trajetórias com ADE menor após fine-tuning.
- `n_traj_degraded`: trajetórias com ADE maior após fine-tuning.
- `catastrophic_ratio`: n_traj_degraded / n_traj_improved (< 0.3 = aceite).

### `sample_size_sensitivity.csv`

- `n_per_year`: número de jogos amostrados por ano (3 ou 6).
- `year`: ano avaliado.
- `model`: Seq2Seq ou Kalman.
- `horizon`: 30→15 ou 60→30.
- `ade_mean`: ADE médio (mm).
- `ci_lo`, `ci_hi`: IC95 bootstrap.
- `n_trajs`: número de trajetórias.

### `by_division_decomposition.csv`

- `division`: divisão SSL (A, B, Unknown).
- `year`: ano avaliado.
- `ade_2019`: ADE baseline 2019 dentro da divisão (mm).
- `ade_y`: ADE do ano y dentro da divisão (mm).
- `excess_mm`: ade_y − ade_2019.
- `n_games`: número de jogos disponíveis.
- `insufficient`: True se n_games < 2 (excluído de conclusões).

### `by_division_2x2_table.csv`

- `division`: divisão SSL.
- `ade_antes_2022`: ADE médio em anos < 2022.
- `ade_apos_2022`: ADE médio em anos ≥ 2022.
- `delta`: ade_apos_2022 − ade_antes_2022 (> 0 = drift sobrevive ao corte).

## Figuras - Mes 3

- `iw_ade_comparison.pdf`: ADE direto vs ADE importance-weighted por ano, com IC95 bootstrap.
- `iw_recovery.pdf`: barra de recovery_pct por ano com sinalização de ESS instável.
- `iw_weights_dist.pdf`: distribuição dos pesos p_2019(x)/p_y(x) por ano.
- `by_division_ade_per_game.pdf`: ADE por jogo facetado por divisão SSL, com IC bootstrap.

## Scripts do Mês 3

```powershell
# Frente 1 — Após rodar compute_trajectory_errors com n=6:
.\\venv\\Scripts\\python.exe model_analise\\sample_size_sensitivity.py

# Frente 2 — Importance weighting (todos os anos):
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\python.exe drift_analise\\compute_importance_weights.py
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\python.exe drift_analise\\plot_iw_results.py

# Frente 3 — Retreino seletivo:
.\\venv\\Scripts\\python.exe model_analise\\retrain_at_breakpoints.py --penalty 1 --epochs 10

# Frente 4 — Validação por divisão:
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\python.exe drift_analise\\division_validation.py

# Adicionar seções 13-16 ao notebook:
.\\venv\\Scripts\\python.exe drift_analise\\add_mes3_sections.py

# Executar notebook completo:
$env:MPLBACKEND='Agg'; .\\venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute drift_analise/drift_analise.ipynb --output drift_analise.executed.ipynb --output-dir drift_analise --ExecutePreprocessor.timeout=900
```
