import pickle
import numpy as np
import os

def get_clean_reference_stats(files):
    all_vels = []
    
    for f_path in files:
        if not os.path.exists(f_path):
            continue
            
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            
        for category in ['yellow', 'blue', 'ball']:
            if category in data:
                for obj_id, subtrajs in data[category].items():
                    for t in subtrajs:
                        if len(t) < 2: continue
                        t_np = np.array(t)
                        dists = np.sqrt(np.sum(np.diff(t_np, axis=0)**2, axis=1))
                        all_vels.extend(dists.tolist())

    all_vels = np.array(all_vels)
    print("=== NOVAS METRICAS DE REFERENCIA (CLEAN) ===")
    print(f"Mean: {np.mean(all_vels):.4f}")
    print(f"Std:  {np.std(all_vels):.4f}")
    print(f"Max:  {np.max(all_vels):.4f}")
    print("============================================")

if __name__ == "__main__":
    reference_files = ["dataset/clean_proc_set_1.pkl", "dataset/clean_proc_set_2.pkl"]
    get_clean_reference_stats(reference_files)