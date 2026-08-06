"""CLI helpers when the same entrypoints run under Jupyter.

ipykernel injects ``-f`` / ``--f=...`` (connection file) into ``sys.argv``;
``argparse.ArgumentParser.parse_args()`` then fails with "unrecognized arguments".
"""
from __future__ import annotations

import argparse
import sys


def _in_zmq_notebook_shell() -> bool:
    """True in Jupyter / VS Code / Cursor notebook kernels (not plain ``ipython`` REPL).

    ZMQ-backed shells expose a ``.kernel``; terminal IPython generally does not.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return False
    if ip is None:
        return False
    return getattr(ip, "kernel", None) is not None


def strip_jupyter_kernel_argv(raw: list[str]) -> list[str]:
    """Remove ipykernel / Jupyter connection-file flags (not for app CLIs)."""
    out: list[str] = []
    skip_one = False
    for a in raw:
        if skip_one:
            skip_one = False
            continue
        if a in ("-f", "--file"):
            skip_one = True
            continue
        low = a.lower()
        if low.startswith("-f=") or low.startswith("--f="):
            continue
        out.append(a)
    return out


def parse_args_skip_kernel(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Like ``parse_args()`` but ignores kernel connection flags and unknown extras.

    In a ZMQ notebook kernel, ``argv is None`` uses an empty argv so defaults apply
    and ``sys.argv`` is never consulted (covers front-ends that pass odd flags).
    """
    if argv is not None:
        raw = strip_jupyter_kernel_argv(list(argv))
    elif _in_zmq_notebook_shell():
        raw = []
    else:
        raw = strip_jupyter_kernel_argv(sys.argv[1:])
    args, _unknown = parser.parse_known_args(raw)
    return args
