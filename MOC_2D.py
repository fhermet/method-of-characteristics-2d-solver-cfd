"""MOC 2D — Application Streamlit principale.

Point d'entrée de l'application : réseau de caractéristiques et champs aérodynamiques.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from moc2d.config import (
    GasProperties,
    GeometryType,
    InletCondition,
    SimulationConfig,
    WallDefinition,
    WallPoint,
)
from moc2d.results import interpolate_to_grid
from moc2d.solver import SolverResult, solve
from moc2d.test_cases import CASE_REGISTRY
from ui_common import (
    COLOR_CMINUS,
    COLOR_CPLUS,
    COLOR_SHOCK,
    COLOR_WALL,
    MACH_COLORSCALE,
    THESIS_LAYOUT,
    thesis_axis,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MOC 2D — Méthode des Caractéristiques",
    page_icon="✈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached solver
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_solve(
    case_name: str,
    M_inf: float,
    gamma: float,
    n_char: int,
    angle1: float,
    angle2: float,
    geom_kind: str,
) -> SolverResult:
    """Build config from parameters and run the solver."""
    if case_name == "Rampe simple":
        config = CASE_REGISTRY["Rampe simple"](
            angle_deg=angle1, M_inf=M_inf, n_char_lines=n_char
        )
        # Override gamma
        config = SimulationConfig(
            gas=GasProperties(gamma=gamma),
            inlet=InletCondition(mach=M_inf),
            walls=config.walls,
            geometry_type=config.geometry_type,
            n_char_lines=n_char,
        )
    elif case_name == "Coin de detente":
        config = CASE_REGISTRY["Coin de detente"](
            angle_deg=angle1, M_inf=M_inf, n_char_lines=n_char
        )
        config = SimulationConfig(
            gas=GasProperties(gamma=gamma),
            inlet=InletCondition(mach=M_inf),
            walls=config.walls,
            geometry_type=config.geometry_type,
            n_char_lines=n_char,
        )
    elif case_name == "Double rampe":
        config = CASE_REGISTRY["Double rampe"](
            angle1_deg=angle1, angle2_deg=angle2, M_inf=M_inf, n_char_lines=n_char
        )
        config = SimulationConfig(
            gas=GasProperties(gamma=gamma),
            inlet=InletCondition(mach=M_inf),
            walls=config.walls,
            geometry_type=config.geometry_type,
            n_char_lines=n_char,
        )
    elif case_name == "Tuyere plane":
        config = CASE_REGISTRY["Tuyere plane"](n_char_lines=n_char)
        config = SimulationConfig(
            gas=GasProperties(gamma=gamma),
            inlet=config.inlet,
            walls=config.walls,
            geometry_type=config.geometry_type,
            n_char_lines=n_char,
        )
    elif case_name == "Tuyere axisymetrique":
        config = CASE_REGISTRY["Tuyere axisymetrique"](n_char_lines=n_char)
        config = SimulationConfig(
            gas=GasProperties(gamma=gamma),
            inlet=config.inlet,
            walls=config.walls,
            geometry_type=config.geometry_type,
            n_char_lines=n_char,
        )
    else:
        # Fallback — should not be reached for predefined cases
        config = CASE_REGISTRY["Rampe simple"](
            angle_deg=angle1, M_inf=M_inf, n_char_lines=n_char
        )
    return solve(config)


@st.cache_data(show_spinner=False)
def cached_solve_custom(
    upper_pts: tuple[tuple[float, float], ...],
    lower_pts: tuple[tuple[float, float], ...],
    M_inf: float,
    gamma: float,
    n_char: int,
    geom_kind: str,
) -> SolverResult:
    """Run the solver for a custom geometry."""
    upper_wall = WallDefinition(
        points=tuple(WallPoint(x, y) for x, y in upper_pts),
        name="upper",
    )
    lower_name = "axis" if geom_kind == "axisymmetric" else "lower"
    lower_wall = WallDefinition(
        points=tuple(WallPoint(x, y) for x, y in lower_pts),
        name=lower_name,
    )
    config = SimulationConfig(
        gas=GasProperties(gamma=gamma),
        inlet=InletCondition(mach=M_inf),
        walls=(upper_wall, lower_wall),
        geometry_type=GeometryType(kind=geom_kind),
        n_char_lines=n_char,
    )
    return solve(config)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _wall_traces(result: SolverResult) -> list[go.BaseTraceType]:
    """Build wall traces from config walls."""
    traces = []
    if result.config is None:
        return traces
    for wall in result.config.walls:
        xs = [p.x for p in wall.points]
        ys = [p.y for p in wall.points]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=COLOR_WALL, width=3),
                name=f"Paroi ({wall.name})",
                legendgroup="walls",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    return traces


def _shock_traces(result: SolverResult) -> list[go.BaseTraceType]:
    """Build shock traces."""
    traces = []
    for i, line_idx in enumerate(result.shock_lines):
        spts = [result.shock_points[j] for j in line_idx if j < len(result.shock_points)]
        if len(spts) < 2:
            continue
        traces.append(
            go.Scatter(
                x=[p.x for p in spts],
                y=[p.y for p in spts],
                mode="lines",
                line=dict(color=COLOR_SHOCK, width=2.5),
                name="Choc" if i == 0 else None,
                legendgroup="shocks",
                showlegend=(i == 0),
                hoverinfo="skip",
            )
        )
    return traces


def build_characteristic_figure(
    result: SolverResult,
    show_mach_bg: bool = False,
) -> go.Figure:
    """Build the characteristic network Plotly figure."""
    fig = go.Figure()

    # --- Optional Mach background heatmap ---
    if show_mach_bg and result.char_points:
        grid = interpolate_to_grid(result, nx=120, ny=60)
        fig.add_trace(
            go.Heatmap(
                x=grid["x"],
                y=grid["y"],
                z=grid["mach"],
                colorscale=MACH_COLORSCALE,
                opacity=0.55,
                colorbar=dict(title="Mach", tickfont=dict(size=11)),
                showscale=True,
                hoverinfo="skip",
                name="Mach (fond)",
            )
        )

    # --- C+ lines ---
    for i, line_idx in enumerate(result.c_plus_lines):
        pts = [result.char_points[j] for j in line_idx if j < len(result.char_points)]
        if len(pts) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in pts],
                y=[p.y for p in pts],
                mode="lines",
                line=dict(color=COLOR_CPLUS, width=0.8),
                name="C+" if i == 0 else None,
                legendgroup="cplus",
                showlegend=(i == 0),
                hoverinfo="skip",
                opacity=0.7,
            )
        )

    # --- C- lines ---
    for i, line_idx in enumerate(result.c_minus_lines):
        pts = [result.char_points[j] for j in line_idx if j < len(result.char_points)]
        if len(pts) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in pts],
                y=[p.y for p in pts],
                mode="lines",
                line=dict(color=COLOR_CMINUS, width=0.8),
                name="C\u2212" if i == 0 else None,
                legendgroup="cminus",
                showlegend=(i == 0),
                hoverinfo="skip",
                opacity=0.7,
            )
        )

    # --- Characteristic points (colored by Mach with hover) ---
    if result.char_points:
        machs = [p.mach for p in result.char_points]
        p0_ref = result.config.inlet.p0 if result.config else 1.0
        T0_ref = result.config.inlet.T0 if result.config else 1.0
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in result.char_points],
                y=[p.y for p in result.char_points],
                mode="markers",
                marker=dict(
                    color=machs,
                    colorscale=MACH_COLORSCALE,
                    size=3,
                    opacity=0.8,
                    colorbar=dict(
                        title="Mach",
                        tickfont=dict(size=11),
                        x=1.02,
                    ),
                    showscale=not show_mach_bg,
                ),
                customdata=[
                    [
                        p.mach,
                        math.degrees(p.theta),
                        p.p / p0_ref,
                        p.T / T0_ref,
                    ]
                    for p in result.char_points
                ],
                hovertemplate=(
                    "M = %{customdata[0]:.3f}<br>"
                    "θ = %{customdata[1]:.2f}°<br>"
                    "p/p₀ = %{customdata[2]:.4f}<br>"
                    "T/T₀ = %{customdata[3]:.4f}<extra></extra>"
                ),
                name="Points caractéristiques",
                showlegend=False,
            )
        )

    # --- Shock lines ---
    for tr in _shock_traces(result):
        fig.add_trace(tr)

    # --- Walls ---
    for tr in _wall_traces(result):
        fig.add_trace(tr)

    # --- Layout ---
    fig.update_layout(
        **THESIS_LAYOUT,
        title="Réseau de caractéristiques",
        xaxis=thesis_axis(title_text="x"),
        yaxis=thesis_axis(
            title_text="y",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            font=dict(family="STIX Two Text, serif", size=12),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        height=520,
    )
    return fig


def build_field_figure(
    result: SolverResult,
    variable: str = "Mach",
) -> go.Figure:
    """Build the aerodynamic field contour/heatmap figure."""
    fig = go.Figure()

    grid = interpolate_to_grid(result, nx=150, ny=75)
    if grid["x"].size == 0:
        fig.update_layout(**THESIS_LAYOUT, title="Champs aérodynamiques — aucune donnée")
        return fig

    _var_map = {
        "Mach": ("mach", "Nombre de Mach"),
        "p/p\u2080": ("p", "p / p\u2080"),
        "T/T\u2080": ("T", "T / T\u2080"),
        "θ (deg)": ("theta", "θ (°)"),
    }
    field_key, colorbar_title = _var_map.get(variable, ("mach", "Mach"))
    z_data = grid[field_key]

    # Convert theta from radians to degrees for display
    if field_key == "theta":
        z_data = np.degrees(z_data)

    fig.add_trace(
        go.Heatmap(
            x=grid["x"],
            y=grid["y"],
            z=z_data,
            colorscale=MACH_COLORSCALE,
            colorbar=dict(title=colorbar_title, tickfont=dict(size=11)),
            hovertemplate=f"{colorbar_title} = %{{z:.4f}}<extra></extra>",
        )
    )

    # Shock overlay
    for tr in _shock_traces(result):
        fig.add_trace(tr)

    # Wall overlay
    for tr in _wall_traces(result):
        fig.add_trace(tr)

    fig.update_layout(
        **THESIS_LAYOUT,
        title=f"Champs aérodynamiques — {variable}",
        xaxis=thesis_axis(title_text="x"),
        yaxis=thesis_axis(
            title_text="y",
            scaleanchor="x",
            scaleratio=1,
        ),
        height=520,
    )
    return fig


def build_geometry_preview(
    upper_rows: list[dict],
    lower_rows: list[dict],
    geom_kind: str,
) -> go.Figure:
    """Preview geometry from custom wall editor data."""
    fig = go.Figure()

    upper_pts = [(r["x"], r["y"]) for r in upper_rows if r.get("x") is not None]
    lower_pts = [(r["x"], r["y"]) for r in lower_rows if r.get("x") is not None]

    if len(upper_pts) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in upper_pts],
                y=[p[1] for p in upper_pts],
                mode="lines+markers",
                line=dict(color=COLOR_WALL, width=2),
                marker=dict(size=6),
                name="Paroi supérieure",
            )
        )
    if len(lower_pts) >= 2:
        lower_label = "Axe de symétrie" if geom_kind == "axisymmetric" else "Paroi inférieure"
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in lower_pts],
                y=[p[1] for p in lower_pts],
                mode="lines+markers",
                line=dict(color=COLOR_WALL, width=2, dash="dot"),
                marker=dict(size=6),
                name=lower_label,
            )
        )

    fig.update_layout(
        **THESIS_LAYOUT,
        title="Aperçu géométrie",
        xaxis=thesis_axis(title_text="x"),
        yaxis=thesis_axis(title_text="y", scaleanchor="x", scaleratio=1),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Render result tabs (shared between predefined and custom modes)
# ---------------------------------------------------------------------------


def render_result_tabs(result: SolverResult) -> None:
    """Render tabs with characteristic network and aerodynamic fields."""
    tab1, tab2 = st.tabs(["Réseau de caractéristiques", "Champs aérodynamiques"])

    with tab1:
        show_mach_bg = st.toggle(
            "Afficher le fond Mach (heatmap)",
            value=False,
            key="mach_bg_toggle",
        )
        with st.spinner("Calcul du réseau de caractéristiques…"):
            fig1 = build_characteristic_figure(result, show_mach_bg=show_mach_bg)
        st.plotly_chart(fig1, use_container_width=True)

        n_pts = len(result.char_points)
        n_shocks = sum(len(l) for l in result.shock_lines)
        col1, col2, col3 = st.columns(3)
        col1.metric("Points caractéristiques", n_pts)
        col2.metric("Lignes C+", len(result.c_plus_lines))
        col3.metric("Lignes C\u2212", len(result.c_minus_lines))

    with tab2:
        variable = st.selectbox(
            "Variable à afficher",
            ["Mach", "p/p\u2080", "T/T\u2080", "θ (deg)"],
            key="field_variable",
        )
        with st.spinner("Interpolation sur grille…"):
            fig2 = build_field_figure(result, variable=variable)
        st.plotly_chart(fig2, use_container_width=True)

        if result.char_points:
            machs = [p.mach for p in result.char_points]
            st.caption(
                f"Mach min = {min(machs):.3f} | max = {max(machs):.3f} | "
                f"Temps calcul = {result.wall_time * 1000:.1f} ms"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

CASE_NAMES = list(CASE_REGISTRY.keys()) + ["Géométrie custom"]

with st.sidebar:
    st.title("MOC 2D")
    st.caption("Méthode des caractéristiques 2D")
    st.divider()

    case_name = st.selectbox("Cas de simulation", CASE_NAMES, index=0)

    st.subheader("Paramètres du gaz")
    gamma = st.slider("γ (gamma)", min_value=1.1, max_value=2.0, value=1.4, step=0.05)

    st.subheader("Paramètres d'entrée")

    is_nozzle = case_name in ("Tuyere plane", "Tuyere axisymetrique")
    is_custom = case_name == "Géométrie custom"

    if not is_nozzle:
        M_inf = st.slider(
            "M∞ (nombre de Mach entrant)",
            min_value=1.1,
            max_value=5.0,
            value=2.0,
            step=0.1,
        )
    else:
        M_inf = 1.5  # fixed for nozzle cases (sonic throat exit)
        st.info("M∞ fixé à 1.5 pour les tuyères (section col).")

    n_char = st.slider(
        "Nombre de lignes caractéristiques (n_char)",
        min_value=10,
        max_value=200,
        value=30,
        step=5,
    )

    # Angle sliders — only for ramp/expansion cases
    angle1 = 10.0
    angle2 = 10.0

    if case_name in ("Rampe simple", "Coin de detente"):
        st.subheader("Géométrie")
        angle1 = st.slider("Angle de la rampe (deg)", min_value=1, max_value=30, value=10)

    elif case_name == "Double rampe":
        st.subheader("Géométrie")
        angle1 = st.slider("Angle 1ère rampe (deg)", min_value=1, max_value=30, value=10)
        angle2 = st.slider("Angle 2ème rampe (deg)", min_value=1, max_value=20, value=10)

    # Custom geometry mode selector
    geom_kind = "planar"
    if is_custom:
        st.subheader("Type de géométrie")
        geom_kind = st.selectbox(
            "Symétrie",
            ["planar", "axisymmetric"],
            format_func=lambda k: "Plan" if k == "planar" else "Axisymétrique",
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header("Méthode des Caractéristiques 2D")

if not is_custom:
    # -----------------------------------------------------------------------
    # Predefined case mode
    # -----------------------------------------------------------------------
    with st.spinner(f'Résolution du cas "{case_name}"…'):
        result = cached_solve(
            case_name=case_name,
            M_inf=M_inf,
            gamma=gamma,
            n_char=n_char,
            angle1=float(angle1),
            angle2=float(angle2),
            geom_kind=geom_kind,
        )

    render_result_tabs(result)

else:
    # -----------------------------------------------------------------------
    # Custom geometry mode
    # -----------------------------------------------------------------------
    st.subheader("Éditeur de géométrie personnalisée")

    st.markdown("Définissez les points de la **paroi supérieure** et de la **paroi inférieure** (ou axe).")

    default_upper = [
        {"x": 0.0, "y": 2.0},
        {"x": 2.0, "y": 2.0},
        {"x": 5.0, "y": 2.0},
    ]
    default_lower = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 5.0, "y": 0.5},
    ]

    col_upper, col_lower = st.columns(2)

    with col_upper:
        st.markdown("**Paroi supérieure**")
        upper_data = st.data_editor(
            default_upper,
            num_rows="dynamic",
            key="upper_wall_editor",
            column_config={
                "x": st.column_config.NumberColumn("x", min_value=0.0, format="%.3f"),
                "y": st.column_config.NumberColumn("y", format="%.3f"),
            },
        )

    with col_lower:
        lower_label = "Axe de symétrie" if geom_kind == "axisymmetric" else "Paroi inférieure"
        st.markdown(f"**{lower_label}**")
        lower_data = st.data_editor(
            default_lower,
            num_rows="dynamic",
            key="lower_wall_editor",
            column_config={
                "x": st.column_config.NumberColumn("x", min_value=0.0, format="%.3f"),
                "y": st.column_config.NumberColumn("y", min_value=0.0, format="%.3f"),
            },
        )

    # Geometry preview
    upper_rows_clean = [r for r in (upper_data or []) if r.get("x") is not None]
    lower_rows_clean = [r for r in (lower_data or []) if r.get("x") is not None]

    if len(upper_rows_clean) >= 2 and len(lower_rows_clean) >= 2:
        st.plotly_chart(
            build_geometry_preview(upper_rows_clean, lower_rows_clean, geom_kind),
            use_container_width=True,
        )
    else:
        st.info("Ajoutez au moins 2 points par paroi pour afficher l'aperçu.")

    # Compute button
    if st.button("Lancer le calcul", type="primary", disabled=(
        len(upper_rows_clean) < 2 or len(lower_rows_clean) < 2
    )):
        upper_tuple = tuple(
            (float(r["x"]), float(r["y"]))
            for r in sorted(upper_rows_clean, key=lambda r: r["x"])
        )
        lower_tuple = tuple(
            (float(r["x"]), float(r["y"]))
            for r in sorted(lower_rows_clean, key=lambda r: r["x"])
        )

        with st.spinner("Résolution en cours…"):
            custom_result = cached_solve_custom(
                upper_pts=upper_tuple,
                lower_pts=lower_tuple,
                M_inf=M_inf,
                gamma=gamma,
                n_char=n_char,
                geom_kind=geom_kind,
            )

        st.session_state["custom_result"] = custom_result

    if "custom_result" in st.session_state:
        st.divider()
        st.subheader("Résultats — Géométrie personnalisée")
        render_result_tabs(st.session_state["custom_result"])
    elif len(upper_rows_clean) >= 2 and len(lower_rows_clean) >= 2:
        st.info("Cliquez sur « Lancer le calcul » pour démarrer la simulation.")
