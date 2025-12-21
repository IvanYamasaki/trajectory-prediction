from .read_logs import process_log
from .smoother_params import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params
from .smooth_data import Smoother

data_set_files = ['data_set_1', 'data_set_2', 'data_set_3', 'data_set_4', 'data_set_5']
processed_data_files = ['proc_set_1', 'proc_set_2', 'proc_set_3', 'proc_set_4', 'proc_set_5']

data_set_files = ['data_set_5']
processed_data_files = ['proc_set_5']

print("---- Lendo arquivos de log brutos ----")
for file in data_set_files:
    print(file)
    process_log(file)

print('---- Processando parâmetros do Kalman (Ball) ----')
get_ball_position_series_params()
print('---- Processando parâmetros do Kalman (Robot Position) ----')
get_robot_position_series_params()
print('---- Processando parâmetros do Kalman (Robot Heading) ----')
get_robot_heading_series_params()

print("---- Suavizando os dados de todas as trajetórias ----")
smoother = Smoother()
for (source, dest) in zip(data_set_files, processed_data_files):
    print(f"Suavizando {source} -> {dest}")
    smoother.smooth_data(source, dest)

print("---- Processamento de dados concluído! ----")