# Trajectory Prediction
This repository introduces an encoder-decoder sequence to sequence neural network to forecast trajectories.
The neural network can be configured to a variable length input and predict a reasonable number of future time steps.

We utilized data from RoboCup SSL games to train the model. The inputs are a sequence of position, velocity and orientation 
for each time step, like the following:
```
[x y v_x v_y psi]
```

The network predicts a sequence of `[v_x v_y]`, which we integrate to find the
robot's future trajectory. A representation of the neural network is depicted below:

![alt text](https://github.com/LucasSte/trajectory-prediction/raw/master/docs/Robot_overview_nn.png)

We also analysed adding information about the ball, in an attempt to improve prediction. We conceived a ball encoder that
processes a sequence of position and velocity for the ball and aggregates that into the prediction. A diagram containing the architecture to aggregate 
information about the ball is available below:

![alt text](https://github.com/LucasSte/trajectory-prediction/raw/master/docs/ball_encoder.png)

#### More information

To find out more about the model's architecture, training and testing procedures, please check out
my [graduation thesis](https://github.com/LucasSte/Research/blob/4c6dd15c91670505114df42b3bab0490a8bf1844/tese.pdf).

### Running the models

#### Prepare the dataset
1. Enter the `dataset` folder, by doing `cd dataset`.
2. Run `download_dataset.sh`. This file downloads the dataset from RoboCup official logs repositories.
3. Run `python3 process_dataset.py` to process the dataset and prepare it for training and testing.

#### Train the models

From the root folder, run `python3 model_analise/train_models.py`. It will train three models.
* Two models that consume only data about the robots.
* Two models that consume data about tha ball and the robots.
* A multilayer perceptron network.

Each model has been trained in two configurations:
1. A look back window of 30 time steps and a prediction of 15 time steps.
2. A look back window of 60 time steps and a prediction of 30 time steps.

If you would like to visualize plots of batch error and validation error during training,
uncomment the `plot` function in `model_analise/train_models.py` and in
`model_analise/comparison_tests.py`.

#### Testing the models

Running `python3 model_analise/compare_models.py` will run all the trained configurations in a testing set.
It will calculate the mean average error, average displacement error and final displacement error and print them.
It will also measure such metrics for a Kalman predictor, which serves as a reference for comparison.

### Data Drift Analysis

This repository extends the original model with a full **data drift analysis pipeline** covering temporal degradation detection and covariate-shift adaptation across RoboCup SSL seasons (2019–2025).

The pipeline has four stages:
1. **Granular infrastructure** (`model_analise/compute_trajectory_errors.py`, `drift_analise/chapter01_descriptive_pipeline.py` — covariate-shift tables and per-game kinematic features) — trajectory-level ADE/FDE.
2. **Formal drift detection** (`drift_analise/drift_analise.ipynb`, `drift_analise/chapter02_deteccao_pipeline.py`) — ADWIN, Page-Hinkley, KSWIN online detectors and offline Pelt change-point detection with null-detector permutation test.
3. **Covariate-shift quantification** (`drift_analise/chapter03_visuals.py` — `main_compute_importance_weights`) — LSIF importance weighting decomposes excess ADE into covariate vs concept drift; per-feature analysis ranks kinematic dimensions by explanatory power.
4. **Selective retraining** (`model_analise/retrain_at_breakpoints.py`) — fine-tuning at Pelt breakpoints with early-stop on catastrophic-interference ratio.

All generated figures (PT/EN) and CSV artefacts are under `Relas/results/drift/<chapter>/`. For the full data dictionary and reproduction steps see [`Relas/PIPELINE.md`](Relas/PIPELINE.md).

### Project layout

The workspace is organized by responsibility:

* `weights/`: model weight files (`*.h5`).
* `model_analise/`: model architecture, training, comparison, debugging, and result extraction scripts.
* `drift_analise/`: drift-analysis notebooks and drift-specific scripts.
* `Relas/`: report material, pipeline documentation, and generated report outputs.
