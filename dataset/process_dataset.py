import os
import sys

# Crucial: Isso permite que o Python encontre os scripts dentro da pasta dataset
# sem precisar dos pontos de importação relativa.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from read_logs import process_log # Sem ponto
from smoother_params import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params # Sem ponto
from smooth_data import Smoother # Sem ponto

data_set_files = ['data_set_33', 'data_set_34', 'data_set_35', 'data_set_36', 'data_set_37', 'data_set_38']
processed_data_files = ['proc_set_33', 'proc_set_34', 'proc_set_35', 'proc_set_36', 'proc_set_37', 'proc_set_38']

print("---- Lendo arquivos de log brutos (Segmentos de Jogo) ----")
for file in data_set_files:
    print(f"Processando: {file}")
    process_log(file)

# print('---- Calibrando Parâmetros do Kalman ----')
# get_ball_position_series_params()
# get_robot_position_series_params()
# get_robot_heading_series_params()

print("---- Suavizando os dados ----")
smoother = Smoother()
for (source, dest) in zip(data_set_files, processed_data_files):
    print(f"Suavizando {source} -> {dest}")
    smoother.smooth_data(source, dest)

print("---- Processamento Concluído! ----")