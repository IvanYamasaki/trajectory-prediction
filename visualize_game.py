import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

FILE_PATH = 'dataset/proc_set_23.pkl'

def plot_ssl_div_a():
    global FILE_PATH
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]

    if not os.path.exists(FILE_PATH):
        print("Arquivo não encontrado:", FILE_PATH)
        return

    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(15, 11))

    # --- CAMPO SSL DIVISÃO A (12000 x 9000 mm) ---
    plt.plot(
        [-6000, 6000, 6000, -6000, -6000],
        [-4500, -4500, 4500, 4500, -4500],
        'k--', alpha=0.5, label='Campo SSL Divisão A (12x9m)'
    )
    plt.axvline(0, color='k', alpha=0.2, linestyle=':')
    plt.axhline(0, color='k', alpha=0.2, linestyle=':')

    def plot_team(team_data, color, name):
        x_trajs = team_data.get('position', {}).get('x', [])
        y_trajs = team_data.get('position', {}).get('y', [])
        n = min(len(x_trajs), len(y_trajs))

        total = 0
        for i in range(n):
            x_np = np.asarray(x_trajs[i], dtype=float)
            y_np = np.asarray(y_trajs[i], dtype=float)
            if x_np.size > 1 and y_np.size > 1:
                plt.plot(x_np, y_np, color=color, alpha=0.18, linewidth=0.6)
                total += 1
        return total

    def plot_ball(ball_dict, color='g'):
        # ball é um dict: stop_id -> {'x','y',...}
        total = 0
        if not isinstance(ball_dict, dict):
            return 0
        for _, seg in ball_dict.items():
            x = seg.get('x', None)
            y = seg.get('y', None)
            if x is None or y is None:
                continue
            x_np = np.asarray(x, dtype=float)
            y_np = np.asarray(y, dtype=float)
            if x_np.size > 1 and y_np.size > 1:
                plt.plot(x_np, y_np, color=color, alpha=0.25, linewidth=0.8)
                total += 1
        return total

    blue = data.get('blue', {})
    yellow = data.get('yellow', {})
    ball = data.get('ball', {})

    n_blue = plot_team(blue, color='b', name='blue')
    n_yellow = plot_team(yellow, color='orange', name='yellow')
    n_ball = plot_ball(ball, color='g')

    plt.title(f"{os.path.basename(FILE_PATH)} | blue={n_blue} yellow={n_yellow} ball={n_ball} trajetórias")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim(-7000, 7000)
    plt.ylim(-5500, 5500)

    print(f"-> Plot concluído: blue={n_blue}, yellow={n_yellow}, ball={n_ball}")
    plt.show()

if __name__ == "__main__":
    plot_ssl_div_a()
