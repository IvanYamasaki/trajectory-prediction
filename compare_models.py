import os
from dataset.load_dataset import LoadDataSet
from ai_model.predictor import RobotOnlyPredictor, BallRobotPredictor
from ai_model.losses import TestLoss
from comparison_tests import MLPComparison, KalmanFilterComparison


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dataset = 'proc_set_4'
files = ['dataset/'+dataset]

loader_global = LoadDataSet(30, 15) # O shape aqui não importa para carregar
loader_global.load_params('dataset/norm_params')

def compare_models(look_back, look_forth, output_dims, robot_model_name, ball_model_name):
    loader = LoadDataSet(look_back, look_forth)
    loader.ball_avg = loader_global.ball_avg
    loader.ball_std = loader_global.ball_std
    loader.robots_avg = loader_global.robots_avg
    loader.robots_std = loader_global.robots_std

    robot_x, ball_x, ball_mask, y = loader.load_data(files, for_test=True)
    loader.convert_to_real(y)

    # --- MODELO ROBOT ONLY (1 ENTRADA) ---
    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    
    # FORÇA A CONSTRUÇÃO: Robô tem 5 features [x, y, v_x, v_y, psi]
    input_shape_robot = (None, look_back, 5)
    seq_predictor.build(input_shape_robot) 

    seq_predictor.load_model(robot_model_name)
    res = seq_predictor.predict(robot_x, batch_size=1024)

    y_pred_conv = loader.convert_batch(robot_x, res)

    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for robot model {look_back} -> {look_forth}')
    test_loss.print_error()

    # --- MODELO BALL ROBOT (3 ENTRADAS) ---
    seq_predictor = BallRobotPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    
    # FORÇA A CONSTRUÇÃO: Modelo tem 3 entradas, passadas como lista
    input_shape_robot_full = (None, look_back, 5)
    input_shape_ball = (None, look_back, 4) 
    input_shape_mask = (None, look_back)
    
    # Passa as 3 formas de entrada como uma lista para o build()
    seq_predictor.build([input_shape_robot_full, input_shape_ball, input_shape_mask])

    seq_predictor.load_model(ball_model_name)
    res = seq_predictor.predict([robot_x, ball_x, ball_mask], batch_size=1024)
    y_pred_conv = loader.convert_batch(robot_x, res)

    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    print(f'--- Results for ball model {look_back} -> {look_forth}')
    test_loss.print_error()


compare_models(30, 15, 2, 'robot_30_15_t', 'ball_30_15_t')
compare_models(60, 30, 2, 'robot_60_30_t', 'ball_60_30_t')

# --- Resultados para MLP 30 -> 15 ---
print('--- Results for MLP model 30 -> 15')
mlp_comparison_model_15 = MLPComparison(30, 15, 2)
mlp_comparison_model_15.loader.robots_avg = loader_global.robots_avg
mlp_comparison_model_15.loader.robots_std = loader_global.robots_std
mlp_comparison_model_15.loader.ball_avg = loader_global.ball_avg
mlp_comparison_model_15.loader.ball_std = loader_global.ball_std
mlp_comparison_model_15.test_model(files, 'mlp_comp')

# --- Resultados para MLP 60 -> 30 ---
print('--- Results for MLP model 60 -> 30')
mlp_comparison_model_30 = MLPComparison(60, 30, 2)
mlp_comparison_model_30.loader.robots_avg = loader_global.robots_avg
mlp_comparison_model_30.loader.robots_std = loader_global.robots_std
mlp_comparison_model_30.loader.ball_avg = loader_global.ball_avg
mlp_comparison_model_30.loader.ball_std = loader_global.ball_std
mlp_comparison_model_30.test_model(files, 'mlp_comp_2')

kf_comp = KalmanFilterComparison(30, 15, 'dataset/'+dataset, 'dataset/position_series_params')
kf_comp.perform_test()

kf_comp_2 = KalmanFilterComparison(60, 30, 'dataset/'+dataset, 'dataset/position_series_params')
kf_comp_2.perform_test()


