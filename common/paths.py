"""Central paths for drift analysis outputs under Relas/results/drift/.

Convention:
  CH01 — descriptive degradation & diagnostics (PDF/PNG + summary CSVs)
  CH02 — stream-based drift detection (concept drift, calibration, decomposition)
  CH03 — covariate shift / importance weighting / division validation
  CH04 — robustness & methodological validation (null, ARL, Pelt/BOCPD, sweeps)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

__all__ = [
    "project_root",
    "DRIFT_RESULTS",
    "CH01",
    "CH02",
    "CH03",
    "CH04",
    "COVARIATE_SHIFT_OUT",
    "DRIFT_DECOMPOSITION_CSV",
    "prepend_matching_venv_site_packages",
    "ensure_chapter_dirs",
    "ensure_drift_decomposition_csv",
    "COV_DIR",
    "DATASET_CSV",
    "weights_path",
    "norm_stats_path",
    "proc_set_path",
    "MES7_DIR",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DRIFT_RESULTS = project_root() / "Relas" / "results" / "drift"
CH01 = DRIFT_RESULTS / "01_degradacao_e_diagnostico"
CH02 = DRIFT_RESULTS / "02_deteccao_no_stream"
CH03 = DRIFT_RESULTS / "03_covariate_shift_e_explicacao"
CH04 = DRIFT_RESULTS / "04_robustez_e_validacao"

COVARIATE_SHIFT_OUT = project_root() / "covariate_shift_out"
DRIFT_DECOMPOSITION_CSV = CH02 / "drift_decomposition.csv"

COV_DIR = COVARIATE_SHIFT_OUT
DATASET_CSV = project_root() / "drift_analise" / "dataset" / "dataset.csv"
MES7_DIR = project_root() / "Relas" / "results" / "mes7"


def weights_path(look_back, look_forth):
    return project_root() / "weights" / f"robot_{look_back}_{look_forth}_t.weights.h5"


def norm_stats_path(look_back, look_forth):
    return project_root() / "model" / f"normalization_stats_{look_back}_{look_forth}.pkl"


def proc_set_path(n):
    return project_root() / "dataset" / f"proc_set_{n}"


def _pyvenv_python_mm(cfg: Path) -> tuple[int, int] | None:
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.lower().startswith("version"):
            continue
        if "=" not in line:
            continue
        ver = line.split("=", 1)[1].strip()
        parts = ver.split(".")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None
    return None


def _venv_site_packages(venv_dir: Path) -> Path | None:
    if sys.platform == "win32":
        sp = venv_dir / "Lib" / "site-packages"
        return sp if sp.is_dir() else None
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    sp = venv_dir / "lib" / f"python{pyver}" / "site-packages"
    return sp if sp.is_dir() else None


def _venv_has_ruptures(site: Path) -> bool:
    return (site / "ruptures").is_dir()


def _sys_path_head_is_repo_root(base: Path) -> bool:
    if not sys.path:
        return False
    p0 = sys.path[0]
    try:
        if not p0:
            return Path.cwd().resolve() == base
        return Path(p0).resolve() == base
    except OSError:
        return False


def prepend_matching_venv_site_packages(root: Path | None = None) -> None:
    """Prepend a project-local venv's ``site-packages`` when its Python matches.

    Prefer a venv that actually contains ``ruptures`` (needed for Pelt notebooks
    on Windows where ``.venv`` may be 3.14 without wheels while ``.venv312`` has
    them). If none has ``ruptures``, fall back to the first version-matched venv
    (same as only matching ``pyvenv.cfg``).
    """
    base = (root or project_root()).resolve()
    want = (sys.version_info.major, sys.version_info.minor)
    chosen: Path | None = None
    fallback: Path | None = None
    for name in (".venv312", ".venv", "venv"):
        vdir = base / name
        cfg = vdir / "pyvenv.cfg"
        site = _venv_site_packages(vdir)
        if site is None or not cfg.is_file():
            continue
        mm = _pyvenv_python_mm(cfg)
        if mm is None or mm != want:
            continue
        if _venv_has_ruptures(site):
            chosen = site
            break
        if fallback is None:
            fallback = site
    site = chosen or fallback
    if site is None:
        return
    s = str(site.resolve())
    if s in sys.path:
        return
    if _sys_path_head_is_repo_root(base):
        sys.path.insert(1, s)
    else:
        sys.path.insert(0, s)


def ensure_chapter_dirs() -> None:
    prepend_matching_venv_site_packages()
    for p in (CH01, CH02, CH03, CH04):
        p.mkdir(parents=True, exist_ok=True)


def ensure_drift_decomposition_csv() -> Path:
    """Ensure drift_decomposition.csv exists for chapter03_visuals / legacy workflows.

    Copies from the old mes2 location or legacy outputs if CH02 file is missing.
    """
    ensure_chapter_dirs()
    if DRIFT_DECOMPOSITION_CSV.exists():
        return DRIFT_DECOMPOSITION_CSV
    candidates = [
        project_root() / "Relas" / "results" / "mes2" / "drift_decomposition.csv",
        project_root() / "drift_analise" / "legacy" / "outputs" / "drift_decomposition_mes2.csv",
    ]
    for src in candidates:
        if src.exists():
            shutil.copy2(src, DRIFT_DECOMPOSITION_CSV)
            return DRIFT_DECOMPOSITION_CSV
    return DRIFT_DECOMPOSITION_CSV
