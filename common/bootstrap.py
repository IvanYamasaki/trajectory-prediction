"""Project bootstrap: sys.path + cwd setup shared by all scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def init_project() -> Path:
    """Insert the repo root at sys.path[0] and chdir to it; return the root.

    Replaces the preamble copied across ~30 scripts (PROJECT_ROOT =
    Path(__file__).resolve().parents[1]; sys.path.insert(0, ...); os.chdir(...)).
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return root
