import os
import pickle
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dataset.load_dataset import LoadDataSet
from model_analise.ai_model.predictor import RobotOnlyPredictor, BallRobotPredictor
from model_analise.ai_model.losses import TestLoss
from comparison_tests import MLPComparison, KalmanFilterComparison

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dataset = 'clean_proc_set_8'
files = ['dataset/'+dataset]

# Valores de Referencia (2019)
stats_ref = {
    'mean': 3.37,
    'std': 9.54,
    'max': 87.50
}

# --- 1. CÁLCULO DE COVARIATE SHIFT ---
all_vels = []
for f_path in files:
    target = f_path + ".pkl" if not f_path.endswith(".pkl") else f_path
    if not os.path.exists(target):
        print(f"Arquivo não encontrado: {target}")
        continue
    
    with open(target, 'rb') as f:
        data_raw = pickle.load(f)
    
    # Nos arquivos CLEAN, a estrutura é: data[time][robo_id] = [ [p1, p2...], [p10, p11...] ]
    for category in ['yellow', 'blue', 'ball']:
        if category in data_raw:
            obj_dict = data_raw[category]
            for obj_id, list_of_subtrajs in obj_dict.items():
                # list_of_subtrajs é uma lista de pedaços de trajetórias
                for subtraj in list_of_subtrajs:
                    if len(subtraj) < 2:
                        continue
                    
                    # Calcula a distância entre frames consecutivos dentro do pedaço limpo
                    for i in range(len(subtraj) - 1):
                        p1 = subtraj[i]
                        p2 = subtraj[i+1]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        all_vels.append(dist)

if len(all_vels) == 0:
    print("ERRO: Nenhuma velocidade calculada. Verifique se os arquivos 'clean_' possuem dados.")
    stats_atual = {'mean': 0, 'std': 0, 'max': 0}
else:
    all_vels = np.array(all_vels)
    stats_atual = {
        'mean': np.mean(all_vels),
        'std': np.std(all_vels),
        'max': np.max(all_vels)
    }

print("\n" + "="*50)
print("   DETECCAO DE COVARIATE SHIFT (VELOCIDADE)")
print("="*50)
print(f"{'Metrica':<15} | {'Referencia':<12} | {'Atual':<12} | {'Shift %'}")
print("-" * 50)

for m in ['mean', 'std', 'max']:
    v_ref = stats_ref[m]
    v_atual = stats_atual[m]
    shift = ((v_atual / v_ref) - 1) * 100
    print(f"{m.capitalize():<15} | {v_ref:<12.2f} | {v_atual:<12.2f} | {shift:>+7.1f}%")
print("="*50 + "\n")

# --- 2. EXECUCAO DOS MODELOS ---
loader_global = LoadDataSet(30, 15)
loader_global.load_params('dataset/norm_params')

def compare_models(look_back, look_forth, output_dims, robot_model_name, ball_model_name):
    loader = LoadDataSet(look_back, look_forth)
    loader.ball_avg = loader_global.ball_avg
    loader.ball_std = loader_global.ball_std
    loader.robots_avg = loader_global.robots_avg
    loader.robots_std = loader_global.robots_std

    robot_x, ball_x, ball_mask, y = loader.load_data(files, for_test=True)
    if robot_x.size == 0:
        print("Abortando: Não há sequências longas o suficiente no dataset para este teste.")
        return
    
    loader.convert_to_real(y)

    # Robot Only
    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.build((None, look_back, 5)) 
    seq_predictor.load_model(robot_model_name)
    res = seq_predictor.predict(robot_x, batch_size=1024)
    y_pred_conv = loader.convert_batch(robot_x, res)

    print(f'--- Results for robot model {look_back} -> {look_forth}')
    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    test_loss.print_error()

    # Ball Robot
    seq_predictor = BallRobotPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor.build([(None, look_back, 5), (None, look_back, 4), (None, look_back)])
    seq_predictor.load_model(ball_model_name)
    res = seq_predictor.predict([robot_x, ball_x, ball_mask], batch_size=1024)
    y_pred_conv = loader.convert_batch(robot_x, res)

    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for ball model {look_back} -> {look_forth}')
    test_loss.print_error()

compare_models(30, 15, 2, 'weights/robot_30_15_t', 'weights/ball_30_15_t')
compare_models(60, 30, 2, 'weights/robot_60_30_t', 'weights/ball_60_30_t')

print('--- Results for MLP model 30 -> 15')
MLPComparison(30, 15, 2).test_model(files, 'mlp_comp')

print('--- Results for MLP model 60 -> 30')
MLPComparison(60, 30, 2).test_model(files, 'mlp_comp_2')

print('--- Results for Kalman model 30 -> 15')
kf_comp = KalmanFilterComparison(30, 15, 'dataset/'+dataset, 'dataset/position_series_params')
kf_comp.perform_test()

print('--- Results for Kalman model 60 -> 30')
kf_comp_2 = KalmanFilterComparison(60, 30, 'dataset/'+dataset, 'dataset/position_series_params')
kf_comp_2.perform_test()
