"""Optional-dependency detection.

RetailPulse runs on **pandas + NumPy alone**. Everything else is an accelerator:

    scikit-learn  -> faster K-Means and tree ensembles
    matplotlib    -> PNG charts
    seaborn       -> nicer chart styling
    prophet       -> an extra forecasting model
    joblib        -> model persistence
    mlflow        -> experiment tracking

Each optional package is probed once here. When it is missing, the pipeline
falls back to an implementation written in NumPy, so a clean build never breaks
because a wheel could not be downloaded.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

# Set RETAILPULSE_FORCE_FALLBACK=1 to exercise the built-in NumPy/SVG
# implementations even when scikit-learn and matplotlib are installed.
# This exists so the fallback path is testable, not just theoretical.
FORCE_FALLBACK = os.environ.get("RETAILPULSE_FORCE_FALLBACK", "") not in ("", "0", "false", "False")


@dataclass(frozen=True)
class Availability:
    name: str
    module: str

    @property
    def ok(self) -> bool:
        if FORCE_FALLBACK and self.module in ("sklearn", "matplotlib", "seaborn", "prophet"):
            return False
        return importlib.util.find_spec(self.module) is not None

    @property
    def version(self) -> str:
        if not self.ok:
            return "-"
        try:
            mod = importlib.import_module(self.module)
            return str(getattr(mod, "__version__", "?"))
        except Exception:
            return "?"


SKLEARN = Availability("scikit-learn", "sklearn")
MATPLOTLIB = Availability("matplotlib", "matplotlib")
SEABORN = Availability("seaborn", "seaborn")
PROPHET = Availability("prophet", "prophet")
JOBLIB = Availability("joblib", "joblib")
MLFLOW = Availability("mlflow", "mlflow")
PLOTLY = Availability("plotly", "plotly")
STREAMLIT = Availability("streamlit", "streamlit")
STATSMODELS = Availability("statsmodels", "statsmodels")

ALL = [SKLEARN, MATPLOTLIB, SEABORN, PROPHET, JOBLIB, MLFLOW, PLOTLY, STREAMLIT, STATSMODELS]


def report() -> str:
    """Human-readable summary, printed at the start of a pipeline run."""
    lines = ["Optional dependencies:"]
    for dep in ALL:
        mark = "available" if dep.ok else "not installed (using built-in fallback)"
        lines.append(f"  {dep.name:<16s} {dep.version:<12s} {mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
