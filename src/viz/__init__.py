"""RetailPulse — charting utilities (matplotlib when available, SVG otherwise)."""

from .svg_charts import (
    bar_chart,
    grouped_bar_chart,
    heatmap,
    line_chart,
    save,
    scatter_chart,
)

__all__ = [
    "bar_chart",
    "grouped_bar_chart",
    "heatmap",
    "line_chart",
    "save",
    "scatter_chart",
]
