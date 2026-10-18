"""Minimal SVG charting — the fallback when matplotlib is unavailable.

Produces clean, self-contained SVG files with no third-party dependency. The
goal is not to replace matplotlib, only to guarantee that a pipeline run always
leaves behind readable charts.

Every function returns an SVG string and, via :func:`save`, can write it to
``reports/figures/``.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

PALETTE = ["#2563EB", "#0891B2", "#7C3AED", "#16A34A", "#F59E0B", "#DC2626",
           "#0F172A", "#DB2777"]

FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _esc(text) -> str:
    return html.escape(str(text))


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values a human would choose."""
    if not (hi > lo):
        hi = lo + 1
    raw = (hi - lo) / max(count, 1)
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if step >= raw:
            break
    start = (lo // step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{v / 1_000:.0f}k"
    if a >= 1:
        return f"{v:.0f}"
    return f"{v:.2f}"


def _frame(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" font-family="{FONT}">
  <rect width="{width}" height="{height}" fill="#FFFFFF"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="16" font-weight="700"
        fill="#0F172A">{_esc(title)}</text>
  <text x="{width - 12}" y="{height - 10}" text-anchor="end" font-size="9"
        fill="#94A3B8">RetailPulse &middot; {datetime.now():%Y-%m-%d}</text>
{body}
</svg>"""


# --------------------------------------------------------------------------
# Line chart
# --------------------------------------------------------------------------
def line_chart(series: dict[str, tuple[list, list]], title: str = "",
               width: int = 1000, height: int = 420, y_label: str = "",
               x_label: str = "",
               band: tuple[list, list, list] | None = None) -> str:
    """Multi-series line chart.

    series : {label: (x_values, y_values)} where x values may be dates or numbers
    band   : optional (x, lower, upper) shaded confidence region
    """
    if not series:
        return _frame(width, height, title, "<text x='40' y='60' font-size='12'>no data</text>")

    pad_l, pad_r, pad_t, pad_b = 70, 24, 46, 62
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    all_x = [v for xs, _ in series.values() for v in xs]
    all_y = [v for _, ys in series.values() for v in ys if v == v]
    if band is not None:
        all_y += [v for v in band[2] if v == v]

    def _num(v):
        """Coerce a value to a float, converting datetimes to epoch seconds."""
        return v.timestamp() if hasattr(v, "timestamp") else float(v)

    x_is_time = hasattr(all_x[0], "timestamp")
    xs_num = [_num(v) for v in all_x]
    x_lo, x_hi = min(xs_num), max(xs_num)
    y_lo, y_hi = min(all_y or [0]), max(all_y or [1])
    pad_y = (y_hi - y_lo) * 0.08 or 1
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

    def sx(v):
        return pad_l + (_num(v) - x_lo) / max(x_hi - x_lo, 1e-9) * plot_w

    def sy(v):
        return pad_t + plot_h - (v - y_lo) / max(y_hi - y_lo, 1e-9) * plot_h

    body = [f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            f'fill="#FBFDFF" stroke="#E2E8F0"/>']

    y_ticks = _nice_ticks(y_lo, y_hi, 5)
    for t in y_ticks:
        if not (y_lo <= t <= y_hi):
            continue
        y = sy(t)
        body.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                    f'stroke="#EDF2F7" stroke-width="1"/>')
        body.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                    f'fill="#475569">{_fmt(t)}</text>')

    n_x = 6
    for i in range(n_x + 1):
        v = x_lo + (x_hi - x_lo) * i / n_x
        x = sx(v)
        label = (datetime.fromtimestamp(v).strftime("%b %Y") if x_is_time else _fmt(v))
        body.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
                    f'font-size="10" fill="#475569">{_esc(label)}</text>')

    if band is not None:
        bx, bl, bu = band
        pts_top = " ".join(f"{sx(x):.1f},{sy(u):.1f}" for x, u in zip(bx, bu) if u == u)
        pts_bot = " ".join(f"{sx(x):.1f},{sy(l):.1f}" for x, l in reversed(list(zip(bx, bl))) if l == l)
        if pts_top and pts_bot:
            body.append(f'<polygon points="{pts_top} {pts_bot}" fill="#2563EB" '
                        f'fill-opacity="0.15" stroke="none"/>')

    for i, (label, (xs, ys)) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys) if y == y)
        if not pts:
            continue
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                    f'stroke-width="2" stroke-linejoin="round"/>')
        lx = pad_l + 8 + i * 170
        body.append(f'<rect x="{lx}" y="{pad_t - 22}" width="11" height="11" rx="2" fill="{color}"/>'
                    f'<text x="{lx + 16}" y="{pad_t - 12}" font-size="11" fill="#334155">'
                    f'{_esc(label)}</text>')

    if y_label:
        body.append(f'<text x="16" y="{pad_t + plot_h / 2}" font-size="11" fill="#475569" '
                    f'transform="rotate(-90 16 {pad_t + plot_h / 2})" text-anchor="middle">'
                    f'{_esc(y_label)}</text>')
    if x_label:
        body.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 26}" text-anchor="middle" '
                    f'font-size="11" fill="#475569">{_esc(x_label)}</text>')

    return _frame(width, height, title, "\n".join(body))


# --------------------------------------------------------------------------
# Bar chart
# --------------------------------------------------------------------------
def bar_chart(labels: list, values: list, title: str = "", width: int = 1000,
              height: int = 420, horizontal: bool = False, colors: list | None = None,
              value_suffix: str = "", x_label: str = "") -> str:
    if not labels:
        return _frame(width, height, title, "")

    pad_l, pad_r, pad_t, pad_b = (170 if horizontal else 70), 40, 46, (60 if horizontal else 76)
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    values = [float(v) if v == v else 0.0 for v in values]

    body = [f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            f'fill="#FBFDFF" stroke="#E2E8F0"/>']
    v_hi = max(values + [0]) or 1
    ticks = _nice_ticks(0, v_hi, 5)

    if horizontal:
        slot = plot_h / len(labels)
        bar_h = min(slot * 0.72, 34)
        for t in ticks:
            if t > v_hi * 1.05:
                continue
            x = pad_l + t / max(v_hi, 1e-9) * plot_w
            body.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" '
                        f'stroke="#EDF2F7"/>')
            body.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
                        f'font-size="10" fill="#475569">{_fmt(t)}{_esc(value_suffix)}</text>')
        for i, (lab, val) in enumerate(zip(labels, values)):
            y = pad_t + i * slot + (slot - bar_h) / 2
            w = max(val / max(v_hi, 1e-9) * plot_w, 0)
            color = (colors[i] if colors else PALETTE[i % len(PALETTE)])
            body.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" '
                        f'rx="3" fill="{color}"/>')
            body.append(f'<text x="{pad_l - 10}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" '
                        f'font-size="11" fill="#334155">{_esc(lab)}</text>')
            body.append(f'<text x="{pad_l + w + 6:.1f}" y="{y + bar_h * 0.7:.1f}" font-size="10" '
                        f'fill="#475569">{_fmt(val)}{_esc(value_suffix)}</text>')
    else:
        slot = plot_w / len(labels)
        bar_w = min(slot * 0.7, 64)
        for t in ticks:
            if t > v_hi * 1.05:
                continue
            y = pad_t + plot_h - t / max(v_hi, 1e-9) * plot_h
            body.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                        f'stroke="#EDF2F7"/>')
            body.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                        f'fill="#475569">{_fmt(t)}{_esc(value_suffix)}</text>')
        for i, (lab, val) in enumerate(zip(labels, values)):
            x = pad_l + i * slot + (slot - bar_w) / 2
            h = max(val / max(v_hi, 1e-9) * plot_h, 0)
            y = pad_t + plot_h - h
            color = (colors[i] if colors else PALETTE[i % len(PALETTE)])
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                        f'rx="3" fill="{color}"/>')
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                        f'font-size="10" fill="#475569">{_fmt(val)}{_esc(value_suffix)}</text>')
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{pad_t + plot_h + 16}" '
                        f'text-anchor="end" font-size="10" fill="#334155" '
                        f'transform="rotate(-35 {x + bar_w / 2:.1f} {pad_t + plot_h + 16})">'
                        f'{_esc(lab)[:22]}</text>')

    if x_label:
        body.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 26}" text-anchor="middle" '
                    f'font-size="11" fill="#475569">{_esc(x_label)}</text>')
    return _frame(width, height, title, "\n".join(body))


# --------------------------------------------------------------------------
# Grouped bar chart
# --------------------------------------------------------------------------
def grouped_bar_chart(categories: list, groups: dict[str, list], title: str = "",
                      width: int = 1000, height: int = 420, value_suffix: str = "") -> str:
    if not categories or not groups:
        return _frame(width, height, title, "")

    pad_l, pad_r, pad_t, pad_b = 70, 40, 46, 70
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n_groups = len(groups)
    slot = plot_w / len(categories)
    bar_w = min(slot * 0.8 / n_groups, 40)

    v_hi = max([v for vs in groups.values() for v in vs if v == v] + [0]) or 1
    body = [f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            f'fill="#FBFDFF" stroke="#E2E8F0"/>']

    for t in _nice_ticks(0, v_hi, 5):
        if t > v_hi * 1.05:
            continue
        y = pad_t + plot_h - t / v_hi * plot_h
        body.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                    f'stroke="#EDF2F7"/>')
        body.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                    f'fill="#475569">{_fmt(t)}{_esc(value_suffix)}</text>')

    for gi, (gname, vals) in enumerate(groups.items()):
        color = PALETTE[gi % len(PALETTE)]
        for ci, (cat, val) in enumerate(zip(categories, vals)):
            val = val if val == val else 0.0
            x = pad_l + ci * slot + slot * 0.1 + gi * bar_w
            h = max(val / v_hi * plot_h, 0)
            y = pad_t + plot_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                        f'rx="3" fill="{color}"/>')
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                        f'font-size="9" fill="#475569">{_fmt(val)}</text>')
        lx = pad_l + 8 + gi * 190
        body.append(f'<rect x="{lx}" y="{pad_t - 22}" width="11" height="11" rx="2" fill="{color}"/>'
                    f'<text x="{lx + 16}" y="{pad_t - 12}" font-size="11" fill="#334155">'
                    f'{_esc(gname)}</text>')

    for ci, cat in enumerate(categories):
        x = pad_l + ci * slot + slot / 2
        body.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
                    f'font-size="11" fill="#334155">{_esc(cat)}</text>')
    return _frame(width, height, title, "\n".join(body))


# --------------------------------------------------------------------------
# Scatter plot
# --------------------------------------------------------------------------
def scatter_chart(points: list[tuple[float, float, str]], title: str = "",
                  width: int = 1000, height: int = 460, x_label: str = "",
                  y_label: str = "", max_points: int = 1500) -> str:
    """points : list of (x, y, group_label). Colours are assigned per group."""
    if not points:
        return _frame(width, height, title, "")

    pad_l, pad_r, pad_t, pad_b = 74, 30, 46, 64
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    stride = max(len(points) // max_points, 1)
    pts = points[::stride]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = (x_hi - x_lo) * 0.05 or 1
    y_pad = (y_hi - y_lo) * 0.08 or 1
    x_lo, x_hi, y_lo, y_hi = x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad

    def sx(v):
        return pad_l + (v - x_lo) / (x_hi - x_lo) * plot_w

    def sy(v):
        return pad_t + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    body = [f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            f'fill="#FBFDFF" stroke="#E2E8F0"/>']
    for t in _nice_ticks(y_lo, y_hi, 5):
        if y_lo <= t <= y_hi:
            body.append(f'<line x1="{pad_l}" y1="{sy(t):.1f}" x2="{pad_l + plot_w}" '
                        f'y2="{sy(t):.1f}" stroke="#EDF2F7"/>')
            body.append(f'<text x="{pad_l - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" '
                        f'font-size="10" fill="#475569">{_fmt(t)}</text>')
    for t in _nice_ticks(x_lo, x_hi, 6):
        if x_lo <= t <= x_hi:
            body.append(f'<text x="{sx(t):.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
                        f'font-size="10" fill="#475569">{_fmt(t)}</text>')

    groups = sorted({p[2] for p in pts})
    cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    for g in groups:
        for x, y, grp in pts:
            if grp != g:
                continue
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" '
                        f'fill="{cmap[g]}" fill-opacity="0.55"/>')
    for i, g in enumerate(groups):
        lx = pad_l + 8 + i * 150
        body.append(f'<rect x="{lx}" y="{pad_t - 22}" width="11" height="11" rx="2" '
                    f'fill="{cmap[g]}"/><text x="{lx + 16}" y="{pad_t - 12}" font-size="11" '
                    f'fill="#334155">{_esc(g)}</text>')

    if x_label:
        body.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 24}" text-anchor="middle" '
                    f'font-size="11" fill="#475569">{_esc(x_label)}</text>')
    if y_label:
        body.append(f'<text x="18" y="{pad_t + plot_h / 2}" font-size="11" fill="#475569" '
                    f'transform="rotate(-90 18 {pad_t + plot_h / 2})" text-anchor="middle">'
                    f'{_esc(y_label)}</text>')
    return _frame(width, height, title, "\n".join(body))


# --------------------------------------------------------------------------
# Heatmap
# --------------------------------------------------------------------------
def heatmap(row_labels: list, col_labels: list, values: list[list],
            title: str = "", width: int = 640, height: int = 420) -> str:
    pad_l, pad_t = 90, 56
    cell_w = (width - pad_l - 40) / max(len(col_labels), 1)
    cell_h = (height - pad_t - 60) / max(len(row_labels), 1)
    flat = [v for row in values for v in row if v == v] or [1]
    v_max = max(flat) or 1

    body = []
    for r, rlab in enumerate(row_labels):
        for c, clab in enumerate(col_labels):
            v = values[r][c] if values[r][c] == values[r][c] else 0
            frac = v / v_max
            # light yellow -> deep red
            rr = int(255 - frac * 90)
            gg = int(237 - frac * 200)
            bb = int(160 - frac * 130)
            color = f"rgb({max(rr, 120)},{max(gg, 20)},{max(bb, 20)})"
            x, y = pad_l + c * cell_w, pad_t + r * cell_h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                        f'height="{cell_h:.1f}" fill="{color}" stroke="#FFFFFF"/>')
            body.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                        f'text-anchor="middle" font-size="14" font-weight="600" '
                        f'fill="#0F172A">{_fmt(v)}</text>')
        body.append(f'<text x="{pad_l - 12}" y="{pad_t + r * cell_h + cell_h / 2 + 5:.1f}" '
                    f'text-anchor="end" font-size="12" fill="#334155">{_esc(rlab)}</text>')
    for c, clab in enumerate(col_labels):
        x = pad_l + c * cell_w + cell_w / 2
        body.append(f'<text x="{x:.1f}" y="{pad_t - 12}" text-anchor="middle" font-size="12" '
                    f'fill="#334155">{_esc(clab)}</text>')
    return _frame(width, height, title, "\n".join(body))


# --------------------------------------------------------------------------
def save(svg: str, path: Path) -> Path:
    """Write SVG text to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
