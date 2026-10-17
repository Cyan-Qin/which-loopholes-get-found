"""Repository paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_path(path):
    """Path relative to the repository when it lies inside it, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
