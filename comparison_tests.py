import os
import pickle
import numpy as np
from pykalman import KalmanFilter
from dataset.kalman_smoother import KalmanSmoother
from ai_model.losses import TestLoss, SequenceLoss
from scipy.stats import ks_2samp
import tensorflow as tf
from dataset.load_dataset import LoadDataSet
import matplotlib.pyplot as plt


class KalmanFilterComparison:

    def __init__(self, look_back, look_forth, data_file, params_file):
        self.look_back = look_back
        self.look_forth = look_forth

        with open(data_file + '.pkl', 'rb') as f:
            self.robots_t = pickle.load(f)

        with open(params_file + '.pkl', 'rb') as f:
            params = pickle.load(f)

        if hasattr(params, 'C'):
            self.transition_matrix = params.A
            self.observation_matrix = params.C
            self.observation_noise = np.linalg.inv(np.matmul(params.V_neg_sqrt, params.V_neg_sqrt))
            self.process_noise = np.linalg.inv(np.matmul(params.W_neg_sqrt, params.W_neg_sqrt))
        elif isinstance(params, dict):
            self.transition_matrix = params.get('A')
            self.observation_matrix = params.get('H') or params.get('C')
            self.process_noise = params.get('Q')
            self.observation_noise = params.get('R')
        else:
            self.transition_matrix = getattr(params, 'A', None)
            self.observation_matrix = getattr(params, 'C', getattr(params, 'H', None))
            self.process_noise = getattr(params, 'Q', None)
            self.observation_noise = getattr(params, 'R', None)

        self.smoother = KalmanSmoother()
        self.smoother.load_params(params_file)
        self.true = []
        self.predicted = []

    def get_future(self, a_matrix, last_pos):
        res = []
        for i in range(self.look_forth):
            pos = np.inner(a_matrix, last_pos)
            last_pos = pos
            res.append([pos[0], pos[2]])
        return res

    def process_robots(self, robots_dict):
        for robot_id, list_of_subtrajs in robots_dict.items():
            for series in list_of_subtrajs:
                series = np.array(series)
                if len(series) < (self.look_back + self.look_forth):
                    continue

                for i in range(len(series) - self.look_back - self.look_forth + 1):
                    obs_window = series[i : i + self.look_back]
                    target_window = series[i + self.look_back : i + self.look_back + self.look_forth]

                    # SUBSTITUIÇÃO: Usamos a predição manual com as matrizes A e C
                    prediction = self.predict_kalman(obs_window, self.look_forth)

                    self.true.append(target_window)
                    self.predicted.append(prediction)

    def predict_kalman(self, obs_window, horizon):
        """
        Realiza a predição linear usando as matrizes de Espaço de Estados.
        """
        # 1. Estimativa do estado inicial [x, y, vx, vy] a partir do final da observação
        p_t = obs_window[-1]
        p_t_1 = obs_window[-2]
        v_t = p_t - p_t_1
        
        # Estado: [pos_x, pos_y, vel_x, vel_y]
        state = np.array([p_t[0], p_t[1], v_t[0], v_t[1]])
        
        preds = []
        curr_state = state
        
        for _ in range(horizon):
            # Equação de Transição: x(k+1) = A * x(k)
            curr_state = np.matmul(self.transition_matrix, curr_state)
            
            # Equação de Medição: y(k) = C * x(k)
            # Extraímos a posição (x, y) projetada
            obs = np.matmul(self.observation_matrix, curr_state)
            preds.append(obs[:2]) # Garante que retornamos apenas as coordenadas (x, y)
            
        return np.array(preds)
    def perform_test(self):
        # Itera sobre os times no dicionário limpo
        if 'blue' in self.robots_t:
            self.process_robots(self.robots_t['blue'])
        if 'yellow' in self.robots_t:
            self.process_robots(self.robots_t['yellow'])
            
        self.true = np.array(self.true)
        self.predicted = np.array(self.predicted)

        if self.true.size == 0:
            print("Pulo: Sem dados suficientes para o teste de Kalman.")
            return

        test_loss = TestLoss()
        test_loss(self.true, self.predicted)
        test_loss.print_error()

    def calculate_velocity_distribution(self, trajectories):
        """
        Calcula velocidades a partir das trajetórias (x, y)
        trajectories: shape (amostras, frames, features)
        """
        # Calcula a diferença entre frames (dx, dy)
        # Assumindo que x e y são as primeiras features
        diff = np.diff(trajectories[:, :, :2], axis=1)
        
        # Velocidade escalar por frame: sqrt(dx^2 + dy^2)
        velocities = np.sqrt(np.sum(diff**2, axis=-1)).flatten()
        
        return {
            'mean': np.mean(velocities),
            'std': np.std(velocities),
            'max': np.max(velocities),
            'raw': velocities # Para o teste estatístico
        }    

class MLPBatchLogs(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MLPBatchLogs, self).__init__()
        self.batch_logs = []
        self.val_logs = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_logs.append(logs['loss'])

    def on_test_end(self, logs=None):
        self.val_logs.append(logs['loss'])


class MLPComparison:
    model = None

    def __init__(self, look_back, look_forth, output_dims, use_cuda=True):
        self.look_back = look_back
        self.output_dims = output_dims
        self.look_forth = look_forth
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if use_cuda else '-1'
        self.loader = LoadDataSet(look_back, look_forth)

    def create_model(self):
        data_input = tf.keras.Input(shape=(self.look_back, 5))
        x = tf.keras.layers.Dense(128, activation='relu')(data_input)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.output_dims * self.look_forth)(x)
        x = tf.keras.layers.Reshape((self.look_forth, self.output_dims))(x)

        return tf.keras.Model(inputs=data_input, outputs=x)

    def train_model(self, file_path: list, model_name):
        if self.model is None:
            self.model = self.create_model()
        robot_x, _, _, y = self.loader.load_data(file_path)
        batch_logs = MLPBatchLogs()

        self.model.compile(optimizer=tf.optimizers.Adam(), loss=SequenceLoss(), run_eagerly=False)
        self.model.fit(robot_x, y, epochs=10, batch_size=512, callbacks=[batch_logs], validation_split=0.1)

        # Uncomment this to visualize training metrics
        plt.figure()
        plt.plot(batch_logs.batch_logs)
        plt.title('Batch loss during training')
        plt.plot(batch_logs.val_logs)
        plt.title('Batch loss during validation')
        self.model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanSquaredError())
        self.model.save(model_name + '.h5')

    def test_model(self, file_path: list, model_name):
        if self.model is None:
            self.model = tf.keras.models.load_model(model_name + '.h5')
            
        robot_x, _, _, y = self.loader.load_data(file_path, for_test=True)

        if robot_x is None or robot_x.size == 0:
            print(f"Pulo: Amostras insuficientes para o horizonte {self.look_back}->{self.look_forth}")
            return

        response = self.model.predict(robot_x)
        y_pred_conv = self.loader.convert_batch(robot_x, response)
        self.loader.convert_to_real(y)

        test_loss = TestLoss()
        test_loss(y[:, :, 0:2], y_pred_conv)
        test_loss.print_error()