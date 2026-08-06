"""Processa logs brutos (data_set_<N>.log) em conjuntos suavizados (proc_set_<N>.pkl).

Uso (de qualquer cwd):
    python dataset/process_dataset.py           # processa todos os data_set_<N>.log presentes
    python dataset/process_dataset.py 33 34     # apenas os conjuntos 33 e 34

Os logs brutos vem de dataset/download_dataset.py. A suavizacao usa os
parametros de Kalman ja calibrados (dataset/*_series_params.pkl, versionados);
a recalibracao (smoother_params.py) so e necessaria se eles forem removidos.
"""
import argparse
import os
import re
import sys
from pathlib import Path

# Permite importar os modulos locais da pasta dataset sem import relativo,
# e fixa o cwd na raiz do repo (read_logs/smooth_data usam paths 'dataset/...').
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir(os.path.dirname(current_dir))

from read_logs import process_log  # noqa: E402
from smooth_data import Smoother  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="*", type=int, help="numeros dos data_sets (default: todos os .log presentes)")
    args = ap.parse_args()

    if args.sets:
        ns = sorted(args.sets)
    else:
        ns = sorted(
            int(m.group(1))
            for f in Path(current_dir).glob("data_set_*.log")
            if (m := re.fullmatch(r"data_set_(\d+)\.log", f.name))
        )
    if not ns:
        sys.exit("nenhum data_set_<N>.log encontrado em dataset/ — rode antes: python dataset/download_dataset.py")

    missing = [n for n in ns if not (Path(current_dir) / f"data_set_{n}.log").exists()]
    if missing:
        sys.exit(f"faltam os arquivos data_set_{missing}.log — rode antes: python dataset/download_dataset.py {' '.join(map(str, missing))}")

    print("---- Lendo arquivos de log brutos (Segmentos de Jogo) ----")
    for n in ns:
        print(f"Processando: data_set_{n}")
        process_log(f"data_set_{n}")

    print("---- Suavizando os dados ----")
    smoother = Smoother()
    for n in ns:
        print(f"Suavizando data_set_{n} -> proc_set_{n}")
        smoother.smooth_data(f"data_set_{n}", f"proc_set_{n}")

    print("---- Processamento Concluido! ----")


if __name__ == "__main__":
    main()
