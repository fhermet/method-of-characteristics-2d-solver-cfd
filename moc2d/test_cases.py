"""Predefined test cases for the MOC 2D solver."""

from __future__ import annotations

import math

from moc2d.config import (
    GasProperties,
    InletCondition,
    WallPoint,
    WallDefinition,
    GeometryType,
    SimulationConfig,
)


def simple_ramp(angle_deg=10.0, M_inf=2.0, n_char_lines=30):
    """Flat plate followed by a compressive ramp."""
    angle = math.radians(angle_deg)
    L = 5.0
    H = 2.0
    corner_x = 1.0

    upper = WallDefinition(
        points=(WallPoint(0, H), WallPoint(L, H)), name="upper",
    )
    lower = WallDefinition(
        points=(
            WallPoint(0, 0),
            WallPoint(corner_x, 0),
            WallPoint(L, (L - corner_x) * math.tan(angle)),
        ),
        name="lower",
    )
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        inlet=InletCondition(mach=M_inf),
        walls=(upper, lower),
        geometry_type=GeometryType(kind="planar"),
        n_char_lines=n_char_lines,
    )


def expansion_corner(angle_deg=10.0, M_inf=2.0, n_char_lines=30):
    """Flat plate followed by an expansion corner (Prandtl-Meyer fan)."""
    angle = math.radians(angle_deg)
    L = 5.0
    H = 2.0
    corner_x = 1.0

    upper = WallDefinition(
        points=(WallPoint(0, H), WallPoint(L, H)), name="upper",
    )
    lower = WallDefinition(
        points=(
            WallPoint(0, 0),
            WallPoint(corner_x, 0),
            WallPoint(L, -(L - corner_x) * math.tan(angle)),
        ),
        name="lower",
    )
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        inlet=InletCondition(mach=M_inf),
        walls=(upper, lower),
        geometry_type=GeometryType(kind="planar"),
        n_char_lines=n_char_lines,
    )


def double_ramp(angle1_deg=10.0, angle2_deg=10.0, M_inf=3.0, n_char_lines=30):
    """Two successive compressive ramps."""
    a1 = math.radians(angle1_deg)
    a2 = math.radians(angle1_deg + angle2_deg)
    L = 6.0
    H = 3.0
    x1 = 1.0
    x2 = 3.0
    y1 = 0.0
    y2 = y1 + (x2 - x1) * math.tan(a1)

    upper = WallDefinition(
        points=(WallPoint(0, H), WallPoint(L, H)), name="upper",
    )
    lower = WallDefinition(
        points=(
            WallPoint(0, 0),
            WallPoint(x1, y1),
            WallPoint(x2, y2),
            WallPoint(L, y2 + (L - x2) * math.tan(a2)),
        ),
        name="lower",
    )
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        inlet=InletCondition(mach=M_inf),
        walls=(upper, lower),
        geometry_type=GeometryType(kind="planar"),
        n_char_lines=n_char_lines,
    )


def planar_nozzle(half_angle_deg=15.0, throat_half_height=0.5, exit_half_height=1.5, length=4.0, n_char_lines=40):
    """Planar convergent-divergent nozzle with smooth cubic walls."""
    n_pts = 10
    xs = [i * length / (n_pts - 1) for i in range(n_pts)]
    ys_upper = [
        throat_half_height + (exit_half_height - throat_half_height) * (x / length) ** 1.5
        for x in xs
    ]

    upper = WallDefinition(
        points=tuple(WallPoint(x, y) for x, y in zip(xs, ys_upper)),
        name="upper",
        interpolation="cubic",
    )
    lower = WallDefinition(
        points=(WallPoint(0, 0), WallPoint(length, 0)), name="lower",
    )
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        inlet=InletCondition(mach=1.5),
        walls=(upper, lower),
        geometry_type=GeometryType(kind="planar"),
        n_char_lines=n_char_lines,
    )


def axisymmetric_nozzle(throat_radius=0.5, exit_radius=1.5, length=4.0, n_char_lines=40):
    """Axisymmetric nozzle (body of revolution) with smooth cubic wall."""
    n_pts = 10
    xs = [i * length / (n_pts - 1) for i in range(n_pts)]
    rs = [
        throat_radius + (exit_radius - throat_radius) * (x / length) ** 1.5
        for x in xs
    ]

    wall = WallDefinition(
        points=tuple(WallPoint(x, r) for x, r in zip(xs, rs)),
        name="upper",
        interpolation="cubic",
    )
    axis = WallDefinition(
        points=(WallPoint(0, 0), WallPoint(length, 0)), name="axis",
    )
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        inlet=InletCondition(mach=1.5),
        walls=(wall, axis),
        geometry_type=GeometryType(kind="axisymmetric"),
        n_char_lines=n_char_lines,
    )


CASE_REGISTRY = {
    "Rampe simple": simple_ramp,
    "Coin de detente": expansion_corner,
    "Double rampe": double_ramp,
    "Tuyere plane": planar_nozzle,
    "Tuyere axisymetrique": axisymmetric_nozzle,
}
