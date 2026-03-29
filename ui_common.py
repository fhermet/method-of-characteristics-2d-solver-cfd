"""Shared UI constants and Plotly thesis-style layout for the MOC 2D app."""

from __future__ import annotations

COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

COLOR_CPLUS = "#1f77b4"
COLOR_CMINUS = "#d62728"
COLOR_SHOCK = "#000000"
COLOR_WALL = "#555555"

_AXIS_COMMON = dict(
    showline=True,
    linewidth=1,
    linecolor="black",
    mirror=True,
    ticks="inside",
    tickwidth=1,
    ticklen=5,
    showgrid=True,
    gridcolor="#cccccc",
    gridwidth=0.5,
    tickfont=dict(family="STIX Two Text, serif", size=13),
    title=dict(font=dict(family="STIX Two Text, serif", size=15)),
)

THESIS_LAYOUT = dict(
    font=dict(family="STIX Two Text, serif", size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=80, b=60),
)

_LEGEND_COMMON = dict(
    font=dict(family="STIX Two Text, serif", size=12),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#cccccc",
    borderwidth=1,
)

MACH_COLORSCALE = "Turbo"


def thesis_axis(**overrides):
    """Return axis dict merged with thesis defaults."""
    ax = dict(_AXIS_COMMON)
    if "title_text" in overrides:
        text = overrides.pop("title_text")
        ax["title"] = dict(text=text, font=dict(family="STIX Two Text, serif", size=15))
    ax.update(overrides)
    return ax
