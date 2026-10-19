"""RetailPulse dashboard — shared design system.

One place for colours, KPI cards and Plotly theming, so every page looks like
part of the same product instead of five separate scripts.
"""
from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
NAVY = "#0F2B4C"
NAVY_LIGHT = "#1A3D6B"
BG = "#EEF2F7"
CARD = "#FFFFFF"
INK = "#0F172A"
MUTED = "#475569"
BORDER = "#CBD5E1"

BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#16A34A"
AMBER = "#F59E0B"
RED = "#DC2626"
PURPLE = "#7C3AED"

CHART_SEQUENCE = [BLUE, CYAN, PURPLE, GREEN, AMBER, RED]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK),
    margin=dict(l=10, r=10, t=48, b=10),
    title_font=dict(size=15, color=INK),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
    xaxis=dict(gridcolor="#E2E8F0", linecolor=BORDER, zeroline=False,
               tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="#E2E8F0", linecolor=BORDER, zeroline=False,
               tickfont=dict(color=MUTED)),
    colorway=CHART_SEQUENCE,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
)


def apply_theme() -> None:
    """Inject the global stylesheet. Call once per script run, before layout."""
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, .stApp {{
    background: {BG};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: {INK};
  }}
  .block-container {{ padding: 1.6rem 2rem 3rem 2rem; max-width: 1500px; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
    border-right: none;
  }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {{ color: #CBD5E1; }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label span {{
    color: #F1F5F9 !important; font-size: 0.9rem !important; font-weight: 500 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] span {{
    color: #FFFFFF !important; font-weight: 700 !important;
  }}

  /* ── Headings ── */
  h1, h2, h3, h4 {{ color: {INK} !important; font-weight: 700 !important;
                    letter-spacing: -0.01em; }}

  /* ── Native metrics ── */
  [data-testid="stMetricValue"] {{ font-size: 1.7rem !important; font-weight: 700 !important;
                                   color: {INK} !important; }}
  [data-testid="stMetricLabel"] {{ font-size: 0.76rem !important; font-weight: 600 !important;
                                   color: {MUTED} !important; text-transform: uppercase;
                                   letter-spacing: 0.05em; }}

  hr {{ border-color: {BORDER} !important; }}
  .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
  footer, #MainMenu, header {{ visibility: hidden; }}

  .stSelectbox label, .stMultiSelect label, .stTextInput label, .stNumberInput label,
  .stTextArea label, .stDateInput label, .stCheckbox label, .stRadio label,
  [data-testid="stSlider"] label {{
    color: #1E293B !important; font-weight: 600 !important;
  }}
  .stDownloadButton button {{
    color: #1E293B !important; font-weight: 600 !important; border: 1px solid {BORDER} !important;
    background: #FFFFFF !important; border-radius: 8px !important;
  }}
  .stDownloadButton button:hover {{ background: #F1F5F9 !important;
                                    border-color: {CYAN} !important; color: {CYAN} !important; }}
  .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: #1E293B; }}
  .stTabs [data-baseweb="tab"] {{ color: #334155 !important; font-weight: 600 !important; }}

  /* ── Custom components ── */
  .rp-hero {{
    background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 55%, {CYAN} 160%);
    border-radius: 16px; padding: 26px 30px; color: #FFFFFF; margin-bottom: 20px;
    box-shadow: 0 10px 24px rgba(15,43,76,0.18);
  }}
  .rp-hero h1 {{ color: #FFFFFF !important; margin: 0 0 6px 0; font-size: 1.75rem;
                 font-weight: 800; }}
  .rp-hero p {{ color: #CBD5E1 !important; margin: 0; font-size: 0.92rem; line-height: 1.5; }}

  .rp-kpi {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 18px; height: 100%;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
  }}
  .rp-kpi .label {{ color: {MUTED}; font-size: 0.7rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }}
  .rp-kpi .value {{ color: {INK}; font-size: 1.55rem; font-weight: 800; line-height: 1.15; }}
  .rp-kpi .delta {{ font-size: 0.76rem; font-weight: 600; margin-top: 5px; }}
  .rp-kpi .hint  {{ font-size: 0.72rem; color: {MUTED}; margin-top: 4px; }}

  .rp-section {{
    display: flex; align-items: center; gap: 10px;
    margin: 26px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #E2E8F0;
  }}
  .rp-section .bar {{ width: 4px; height: 20px; background: {BLUE}; border-radius: 2px; }}
  .rp-section .text {{ font-size: 1.06rem; font-weight: 700; color: {INK}; }}
  .rp-section .sub {{ font-size: 0.8rem; color: {MUTED}; font-weight: 500; }}

  .rp-note {{
    background: #F8FAFC; border-left: 4px solid {CYAN}; border-radius: 8px;
    padding: 12px 16px; font-size: 0.85rem; color: #334155; line-height: 1.55;
  }}
  .rp-note code {{
    background: #E2E8F0 !important; color: #0F172A !important;
    padding: 2px 7px !important; border-radius: 5px !important;
    font-size: 0.82rem !important; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-weight: 600 !important; border: 1px solid {BORDER} !important;
  }}
  .rp-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
  }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(f"""
<div class="rp-hero">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>""", unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f'<span class="sub">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
<div class="rp-section">
  <div class="bar"></div>
  <div class="text">{title}</div>
  {sub}
</div>""", unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = "", delta_color: str = MUTED, hint: str = "") -> None:
    delta_html = f'<div class="delta" style="color:{delta_color}">{delta}</div>' if delta else ""
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(f"""
<div class="rp-kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {delta_html}{hint_html}
</div>""", unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """Render KPI cards in equal-width columns."""
    cols = st.columns(len(items), gap="medium")
    for col, item in zip(cols, items):
        with col:
            kpi(**item)


def note(text: str) -> None:
    st.markdown(f'<div class="rp-note">{text}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = BLUE) -> str:
    return f'<span class="rp-badge" style="background:{color}1A;color:{color}">{text}</span>'


def style_fig(fig, title: str = "", height: int = 380):
    """Apply the shared Plotly layout to a figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"))
    return fig


def money(value: float, symbol: str = "£") -> str:
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{symbol}{v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{symbol}{v / 1_000:,.1f}K"
    return f"{symbol}{v:,.0f}"


def number(value: float) -> str:
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:,.1f}K"
    return f"{v:,.0f}"
