"""Baixa os 36 game logs da RoboCup SSL usados no projeto (data_set_3..38).

Fonte: acervo oficial de gamelogs da SSL, hospedado no Seafile da TIGERs
Mannheim (link publicado em https://ssl.robocup.org/collected-data/ como
https://download.tigers-mannheim.de/gamelogs/). O antigo mirror da
RoboJackets referenciado pelo download_dataset.sh original saiu do ar.

Uso (da raiz do repositorio ou de qualquer cwd):
    python dataset/download_dataset.py             # baixa os 36 jogos (~10 GB)
    python dataset/download_dataset.py 33 34       # apenas os data_set_33 e 34
    python dataset/download_dataset.py --year 2024 # apenas os jogos de 2024
    python dataset/download_dataset.py --check     # so verifica as URLs (rapido)

Cada jogo e salvo descomprimido como dataset/data_set_<N>.log (~350-700 MB
cada). Depois rode: python dataset/process_dataset.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SEAFILE = "https://seafile.tigers-mannheim.de"
SHARE_TOKEN = "e85851d9bc9944bf95bb"
DATASET_DIR = Path(__file__).resolve().parent

# proc_set N -> caminho no share /gamelogs/ (verificado em 2026-08-06).
# O mapeamento jogo->ano segue common.constants.PROC_YEAR.
GAMES = {
    3: "2019/div-a/2019-07-04_06-55_KIKS-vs-TIGERs_Mannheim.log.gz",
    4: "2019/div-a/2019-07-07_01-49_RoboTeam_Twente-vs-MRL.log.gz",
    5: "2019/div-a/2019-07-04_12-15_ZJUNlict-vs-RoboTeam_Twente.log.gz",
    6: "2019/div-a/2019-07-04_14-10_KIKS-vs-MRL.log.gz",
    7: "2019/div-a/2019-07-05_04-18_KIKS-vs-OP-AmP.log.gz",
    8: "2019/div-a/2019-07-06_03-11_KIKS-vs-ZJUNlict.log.gz",
    9: "2021/2021-06-21_09-01_TIGERs_Mannheim-vs-KIKS.log.gz",
    10: "2021/2021-06-24_13-10_OMID-vs-RoboJackets.log.gz",
    11: "2021/2021-06-20_14-43_KIKS-vs-RoboDragons.log.gz",
    12: "2021/2021-06-24_14-36_ER-Force-vs-TIGERs_Mannheim.log.gz",
    13: "2021/2021-06-21_07-06_RoboTeam_Twente-vs-ER-Force.log.gz",
    14: "2021/friendly/2021-06-30_20-01_TIGERs_Mannheim-vs-RoboFEI.log.gz",
    15: "2023/div-a/2023-07-06_12-47_RoboTeam_Twente-vs-RoboDragons.log.gz",
    16: "2023/div-a/2023-07-08_11-01_ZJUNlict-vs-RoboDragons.log.gz",
    17: "2023/div-a/2023-07-08_09-11_RoboTeam_Twente-vs-KIKS.log.gz",
    18: "2023/div-a/2023-07-08_17-03_ZJUNlict-vs-TIGERs_Mannheim.log.gz",
    19: "2023/div-a/2023-07-08_18-31_Immortals-vs-KIKS.log.gz",
    20: "2023/div-a/2023-07-09_13-02_TIGERs_Mannheim-vs-ZJUNlict.log.gz",
    21: "2025/robocup/div-a/2025-07-19_15-05_ELIMINATION_PHASE_ER-Force-vs-RoboDragons.log.gz",
    22: "2025/robocup/div-a/2025-07-17_21-33_GROUP_PHASE_RoboDragons-vs-ZJUNlict.log.gz",
    23: "2025/robocup/div-a/2025-07-17_17-01_GROUP_PHASE_TIGERs_Mannheim-vs-RoboDragons.log.gz",
    24: "2025/robocup/div-a/2025-07-18_19-02_GROUP_PHASE_TIGERs_Mannheim-vs-ZJUNlict.log.gz",
    25: "2025/robocup/div-a/2025-07-19_13-32_ELIMINATION_PHASE_RobôCin-vs-ZJUNlict.log.gz",
    26: "2025/robocup/div-a/2025-07-20_15-31_ELIMINATION_PHASE_TIGERs_Mannheim-vs-ZJUNlict.log.gz",
    27: "2022/div-a/2022-07-13_09-12_RoboDragons-vs-ER-Force.log.gz",
    28: "2022/div-a/2022-07-13_10-49_RoboTeam_Twente-vs-TIGERs_Mannheim.log.gz",
    29: "2022/div-a/2022-07-13_12-10_KIKS-vs-ER-Force.log.gz",
    30: "2022/div-a/2022-07-16_11-34_TIGERs_Mannheim-vs-ER-Force.log.gz",
    31: "2022/div-a/2022-07-14_10-55_TIGERs_Mannheim-vs-KIKS.log.gz",
    32: "2022/div-a/2022-07-15_12-08_TIGERs_Mannheim-vs-KIKS.log.gz",
    33: "2024/div-a/2024-07-20_13-16_ELIMINATION_PHASE_RoboDragons-vs-ER-Force.log.gz",
    34: "2024/div-a/2024-07-18_11-33_GROUP_PHASE_RoboDragons-vs-Immortals.log.gz",
    35: "2024/div-a/2024-07-21_11-00_ELIMINATION_PHASE_ZJUNlict-vs-TIGERs_Mannheim.log.gz",
    36: "2024/div-a/2024-07-18_10-01_GROUP_PHASE_TIGERs_Mannheim-vs-RoboTeam_Twente.log.gz",
    37: "2024/div-a/2024-07-18_16-01_GROUP_PHASE_TIGERs_Mannheim-vs-Immortals.log.gz",
    38: "2024/div-a/2024-07-18_07-01_GROUP_PHASE_KIKS-vs-RobôCin.log.gz",
}


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})


def _download_url(share_path: str) -> str:
    p = urllib.parse.quote("/gamelogs/" + share_path)
    return f"{SEAFILE}/d/{SHARE_TOKEN}/files/?p={p}&dl=1"


def _api_list(path: str) -> list[dict]:
    url = (
        f"{SEAFILE}/api/v2.1/share-links/{SHARE_TOKEN}/dirents/"
        f"?path={urllib.parse.quote(path)}"
    )
    with urllib.request.urlopen(_request(url), timeout=30) as r:
        return json.load(r)["dirent_list"]


def _find_in_year(year: str, filename: str, path: str | None = None) -> str | None:
    """Fallback: procura ``filename`` recursivamente sob /gamelogs/<year>/."""
    path = path or f"/gamelogs/{year}/"
    for d in _api_list(path):
        if d.get("is_dir"):
            hit = _find_in_year(year, filename, path + d["folder_name"] + "/")
            if hit:
                return hit
        elif d.get("file_name") == filename:
            return (path + filename)[len("/gamelogs/"):]
    return None


def check(n: int, share_path: str) -> bool:
    try:
        with urllib.request.urlopen(_request(_download_url(share_path), "HEAD"), timeout=30) as r:
            size_mb = int(r.headers.get("Content-Length", 0)) / 1e6
            print(f"[ok] data_set_{n}: {share_path} ({size_mb:.0f} MB)")
            return True
    except Exception as exc:
        print(f"[!!] data_set_{n}: {share_path} -> {exc}")
        return False


def download(n: int, share_path: str) -> None:
    dest = DATASET_DIR / f"data_set_{n}.log"
    if dest.exists():
        print(f"[skip] {dest.name} ja existe")
        return
    url = _download_url(share_path)
    print(f"[baixando] data_set_{n} <- {share_path}")
    try:
        resp = urllib.request.urlopen(_request(url), timeout=60)
    except urllib.error.HTTPError:
        year, filename = share_path.split("/", 1)[0], share_path.rsplit("/", 1)[-1]
        relocated = _find_in_year(year, filename)
        if relocated is None:
            raise
        print(f"[aviso] caminho mudou no servidor; usando {relocated}")
        resp = urllib.request.urlopen(_request(_download_url(relocated)), timeout=60)
    tmp = dest.with_suffix(".log.part")
    with resp, gzip.open(resp, "rb") as gz, open(tmp, "wb") as out:
        shutil.copyfileobj(gz, out, length=1024 * 1024)
    tmp.replace(dest)
    print(f"[ok] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="*", type=int, help="numeros dos data_sets (default: todos)")
    ap.add_argument("--year", type=int, help="baixa apenas os jogos de um ano")
    ap.add_argument("--check", action="store_true", help="apenas verifica as URLs, sem baixar")
    args = ap.parse_args()

    selected = dict(GAMES)
    if args.sets:
        selected = {n: GAMES[n] for n in args.sets}
    if args.year:
        selected = {n: p for n, p in selected.items() if p.startswith(str(args.year))}
    if not selected:
        sys.exit("nenhum jogo selecionado")

    if args.check:
        ok = [check(n, p) for n, p in sorted(selected.items())]
        print(f"\n{sum(ok)}/{len(ok)} URLs ok")
        sys.exit(0 if all(ok) else 1)

    total = len(selected)
    for i, (n, p) in enumerate(sorted(selected.items()), 1):
        print(f"--- {i}/{total}")
        download(n, p)
    print("---- Download concluido. Agora rode: python dataset/process_dataset.py ----")


if __name__ == "__main__":
    main()
