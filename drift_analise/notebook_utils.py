"""Helpers for Jupyter notebooks: display saved figures in deterministic order."""
from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path
import shutil
import subprocess

from IPython.display import HTML, Image, display


def save_figure(fig, path: Path | str, *, dpi: int = 150, **kwargs) -> list[Path]:
    """Save a matplotlib figure to ``path`` and also to a PNG sibling.

    Notebooks display the PNG copy to avoid PDF viewer prompts in VS Code /
    Cursor. The requested path is still written, so existing PDF-based paper
    workflows keep working.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    fig.savefig(p, dpi=dpi, bbox_inches="tight", **kwargs)
    saved.append(p)
    png = p.with_suffix(".png")
    if p.suffix.lower() != ".png":
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        saved.append(png)
    return saved


def display_png(path: Path | str, *, width: int | None = 900) -> None:
    p = Path(path)
    if not p.exists():
        display(f"[missing] {p}")
        return
    display(Image(filename=str(p), width=width))


def display_pdf(
    path: Path | str,
    *,
    width: int = 900,
    height: int = 520,
    max_bytes: int = 12_000_000,
) -> None:
    """Show a PDF inline in the notebook.

    VS Code / Cursor often render PDF iframes as a blank output. Prefer a PNG
    preview of the first page, generated with pdftoppm when available, and keep
    the embedded-PDF fallback for front-ends that support it.
    """
    p = Path(path).resolve()
    if not p.exists():
        display(f"[missing] {p}")
        return
    preview = _pdf_first_page_png(p)
    if preview is not None:
        display(Image(filename=str(preview), width=width))
        return
    data = p.read_bytes()
    if len(data) > max_bytes:
        display(
            HTML(
                f"<p><b>PDF omitted ({len(data) // 1_000_000} MB)</b> "
                f"(inline limit {max_bytes // 1_000_000} MB). "
                f"Open on disk:</p><pre>{escape(str(p))}</pre>"
            )
        )
        return
    b64 = b64encode(data).decode("ascii")
    display(
        HTML(
            f'<iframe title="{escape(p.name)}" width="{width}" height="{height}" '
            'style="border:1px solid #ccc" '
            f'src="data:application/pdf;base64,{b64}"></iframe>'
        )
    )


def _pdf_first_page_png(path: Path, *, dpi: int = 144) -> Path | None:
    """Render the first PDF page to a cached PNG using the local pdftoppm."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        return None
    cache_dir = path.parent / ".notebook_previews"
    out_png = cache_dir / f"{path.stem}.png"
    if out_png.exists() and out_png.stat().st_mtime >= path.stat().st_mtime:
        return out_png
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = cache_dir / path.stem
    try:
        subprocess.run(
            [
                pdftoppm,
                "-f",
                "1",
                "-l",
                "1",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(path),
                str(out_prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out_png if out_png.exists() else None


def display_sorted_glob(
    directory: Path | str,
    pattern: str,
    *,
    prefer_pdf: bool = True,
) -> None:
    """Display all matches under ``directory`` sorted by name."""
    d = Path(directory)
    if not d.exists():
        display(f"[missing dir] {d}")
        return
    paths = sorted(d.glob(pattern))
    if not paths:
        display(f"[no matches] {d}/{pattern}")
        return
    for p in paths:
        if p.suffix.lower() == ".pdf":
            png = p.with_suffix(".png")
            display_png(png) if png.exists() else display_pdf(p)
        elif p.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
            display_png(p)
        elif prefer_pdf and (p.with_suffix(".pdf")).exists():
            display_pdf(p.with_suffix(".pdf"))
        else:
            display(p)
