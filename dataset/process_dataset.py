import pickle
import numpy as np
import os
from .read_logs import process_log
from .smoother_params import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params
from .smooth_data import Smoother

data_set_files = ['data_set_1', 'data_set_2', 'data_set_3', 'data_set_4', 'data_set_5']
processed_data_files = ['proc_set_1', 'proc_set_2', 'proc_set_3', 'proc_set_4', 'proc_set_5']

data_set_files = ['data_set_10']
processed_data_files = ['proc_set_10']

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

def clean_file(input_path, output_path, threshold=120):
    if not os.path.exists(input_path):
        print(f"Arquivo nao encontrado: {input_path}")
        return

    print(f"Lendo: {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    clean_data = {'yellow': {}, 'blue': {}, 'ball': {}}

    for category in ['yellow', 'blue', 'ball']:
        # Bola pode ser mais rapida que robos
        current_threshold = threshold if category != 'ball' else threshold * 2.0
        
        team_data = data.get(category, {})
        items = team_data.items() if isinstance(team_data, dict) else enumerate(team_data)

        for obj_id, trajectory in items:
            # Filtra apenas pontos validos e tenta converter para float
            valid_traj = []
            for p in trajectory:
                if p is not None:
                    try:
                        # Converte coordenadas para float para evitar erro de string
                        valid_traj.append([float(p[0]), float(p[1])])
                    except (ValueError, TypeError, IndexError):
                        continue

            if len(valid_traj) < 2:
                continue

            new_sequences = []
            current_seq = [valid_traj[0]]

            for i in range(len(valid_traj) - 1):
                p1 = valid_traj[i]
                p2 = valid_traj[i+1]
                
                # Calculo da distancia Euclidiana
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

                if dist < current_threshold:
                    current_seq.append(p2)
                else:
                    # Salto detectado: quebra a sequencia
                    if len(current_seq) >= 15:
                        new_sequences.append(current_seq)
                    current_seq = [p2]
            
            if len(current_seq) >= 15:
                new_sequences.append(current_seq)

            if new_sequences:
                clean_data[category][obj_id] = new_sequences

    with open(output_path, 'wb') as f:
        pickle.dump(clean_data, f)
    print(f"Sucesso: {output_path} gerado.\n")

for d in processed_data_files:
    input_file = f"dataset/{d}.pkl"
    output_file = f"dataset/clean_{d}.pkl"
    clean_file(input_file, output_file)

print("---- Processamento de dados concluído! ----")