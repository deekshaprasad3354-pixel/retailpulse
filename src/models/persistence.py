"""Model persistence helpers.

Uses joblib when it is installed (better for large NumPy arrays) and falls back
to the standard-library pickle. The file extension is reported back so callers
can log the real artefact path.
"""
from __future__ import annotations

from pathlib import Path

from src.optional_deps import JOBLIB


def save_model(payload, path: Path) -> Path:
    """Persist an arbitrary Python object; returns the path actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if JOBLIB.ok:
        import joblib

        joblib.dump(payload, path)
        return path

    import pickle

    # a .joblib file written by pickle is confusing, so be explicit
    if path.suffix == ".joblib":
        path = path.with_suffix(".pkl")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: Path):
    """Load a payload written by :func:`save_model`."""
    path = Path(path)
    if not path.exists() and path.suffix == ".pkl":
        alt = path.with_suffix(".joblib")
        if alt.exists():
            path = alt
    if JOBLIB.ok and path.suffix == ".joblib":
        import joblib

        return joblib.load(path)

    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)
