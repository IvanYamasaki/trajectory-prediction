import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = 'dataset/proc_set_5.pkl'

def plot_ssl_div_a():
    if not os.path.exists(FILE_PATH):
        print("Arquivo não encontrado.")
        return

    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)

    # Extração das trajetórias conforme a estrutura identificada
    blue = data.get('blue', {})
    x_trajs = blue.get('position', {}).get('x', [])
    y_trajs = blue.get('position', {}).get('y', [])

    plt.figure(figsize=(15, 11))
    
    # --- DESENHO DO CAMPO DIVISÃO A (12000 x 9000 mm) ---
    # Linhas externas: Comprimento de -6000 a 6000, Largura de -4500 a 4500
    plt.plot([-6000, 6000, 6000, -6000, -6000], [-4500, -4500, 4500, 4500, -4500], 
             'k--', alpha=0.5, label='Campo SSL Divisão A (12x9m)')
    
    # Linha central
    plt.axvline(0, color='k', alpha=0.2, linestyle=':')
    plt.axhline(0, color='k', alpha=0.2, linestyle=':')

    # Plotagem das trajetórias
    total_plotted = 0
    for x_list, y_list in zip(x_trajs, y_trajs):
        x_np = np.array(x_list, dtype=float)
        y_np = np.array(y_list, dtype=float)
        
        if x_np.size > 0:
            # Mantendo alpha baixo para visualizar a densidade de 1587 trajetórias
            plt.plot(x_np, y_np, alpha=0.25, linewidth=0.6)
            total_plotted += 1

    plt.title(f"SSL Divisão A: {total_plotted} Trajetórias em Milímetros Reais (mm)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    
    # Mantém a proporção real 1:1
    plt.axis('equal') 
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.legend(loc='upper right')
    
    # Limites de visualização com margem de segurança
    plt.xlim(-7000, 7000)
    plt.ylim(-5500, 5500)
    
    print(f"-> Plotagem concluída no formato Division A.")
    plt.show()

if __name__ == "__main__":
    plot_ssl_div_a()