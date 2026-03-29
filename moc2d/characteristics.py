"""Method of Characteristics unit processes for 2D steady supersonic flow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moc2d.config import GasProperties, GeometryType, WallDefinition
from moc2d.gas import (
    mach_angle,
    prandtl_meyer,
    inverse_prandtl_meyer,
    pressure_ratio,
    temperature_ratio,
)
from moc2d.geometry import wall_y_at, wall_angle_at


@dataclass
class CharPoint:
    """A single point in the characteristic network."""
    x: float
    y: float
    mach: float
    theta: float
    nu: float
    p: float
    T: float
    kind: str = "interior"


def _compute_thermo(mach: float, gas: GasProperties, p0: float, T0: float) -> tuple[float, float]:
    p = p0 * pressure_ratio(mach, gas.gamma)
    T = T0 * temperature_ratio(mach, gas.gamma)
    return float(p), float(T)


def interior_point(
    p1: CharPoint,
    p2: CharPoint,
    gas: GasProperties,
    geom_type: GeometryType,
    p0: float = 1.0,
    T0: float = 1.0,
    n_iter: int = 2,
) -> CharPoint:
    """Compute an interior point from two known points.

    p1 is on the C- characteristic (above), p2 is on the C+ characteristic (below).
    Riemann invariants: K+ = theta + nu (const along C-), K- = theta - nu (const along C+).
    """
    K_plus = p1.theta + p1.nu
    K_minus = p2.theta - p2.nu

    theta3 = (K_plus + K_minus) / 2.0
    nu3 = (K_plus - K_minus) / 2.0

    M3 = inverse_prandtl_meyer(nu3, gas.gamma)
    mu3 = float(mach_angle(M3))

    mu1 = float(mach_angle(p1.mach))
    mu2 = float(mach_angle(p2.mach))

    # C- from p1 (right-running): slope = tan(theta1 - mu1)  [downward from upper point]
    # C+ from p2 (left-running):  slope = tan(theta2 + mu2)  [upward from lower point]
    slope_cm = math.tan(p1.theta - mu1)
    slope_cp = math.tan(p2.theta + mu2)

    if abs(slope_cm - slope_cp) < 1e-15:
        x3 = (p1.x + p2.x) / 2
        y3 = (p1.y + p2.y) / 2
    else:
        x3 = (p2.y - p1.y + slope_cm * p1.x - slope_cp * p2.x) / (slope_cm - slope_cp)
        y3 = p1.y + slope_cm * (x3 - p1.x)

    # Corrector iterations with averaged slopes
    for _ in range(n_iter - 1):
        avg_slope_cm = math.tan((p1.theta + theta3) / 2 - (mu1 + mu3) / 2)
        avg_slope_cp = math.tan((p2.theta + theta3) / 2 + (mu2 + mu3) / 2)

        if abs(avg_slope_cm - avg_slope_cp) > 1e-15:
            x3 = (p2.y - p1.y + avg_slope_cm * p1.x - avg_slope_cp * p2.x) / (
                avg_slope_cm - avg_slope_cp
            )
            y3 = p1.y + avg_slope_cm * (x3 - p1.x)

        # Axisymmetric correction
        if geom_type.kind == "axisymmetric" and abs(y3) > 1e-10:
            ds_cm = math.sqrt((x3 - p1.x) ** 2 + (y3 - p1.y) ** 2)
            ds_cp = math.sqrt((x3 - p2.x) ** 2 + (y3 - p2.y) ** 2)
            y_avg1 = (p1.y + y3) / 2
            y_avg2 = (p2.y + y3) / 2
            if abs(y_avg1) > 1e-10 and abs(y_avg2) > 1e-10:
                M_avg = (p1.mach + M3) / 2
                delta_plus = (
                    math.sqrt(M_avg**2 - 1) * math.sin((p1.theta + theta3) / 2) / y_avg1 * ds_cm
                )
                M_avg2 = (p2.mach + M3) / 2
                delta_minus = (
                    math.sqrt(M_avg2**2 - 1) * math.sin((p2.theta + theta3) / 2) / y_avg2 * ds_cp
                )
                K_plus_corr = p1.theta + p1.nu - delta_plus
                K_minus_corr = p2.theta - p2.nu + delta_minus
                theta3 = (K_plus_corr + K_minus_corr) / 2
                nu3 = (K_plus_corr - K_minus_corr) / 2
                M3 = inverse_prandtl_meyer(nu3, gas.gamma)
                mu3 = float(mach_angle(M3))

    p3, T3 = _compute_thermo(M3, gas, p0, T0)
    return CharPoint(x=x3, y=y3, mach=M3, theta=theta3, nu=nu3, p=p3, T=T3, kind="interior")


def wall_point(
    p1: CharPoint,
    wall: WallDefinition,
    gas: GasProperties,
    geom_type: GeometryType,
    side: str,
    p0: float = 1.0,
    T0: float = 1.0,
    n_iter: int = 3,
) -> CharPoint:
    """Compute a wall point from one known interior point and a wall constraint.

    side="upper": p1 sends a C+ characteristic to the upper wall.
    side="lower": p1 sends a C- characteristic to the lower wall.
    """
    mu1 = float(mach_angle(p1.mach))

    if side == "upper":
        # C+ (left-running) goes upward to the upper wall: slope = tan(theta + mu)
        slope = math.tan(p1.theta + mu1)
        K = p1.theta - p1.nu  # K- along C+
    else:
        # C- (right-running) goes downward to the lower wall: slope = tan(theta - mu)
        slope = math.tan(p1.theta - mu1)
        K = p1.theta + p1.nu  # K+ along C-

    x_w = p1.x + 0.5
    for _ in range(n_iter):
        y_w = wall_y_at(wall, x_w)
        x_w = p1.x + (y_w - p1.y) / slope if abs(slope) > 1e-15 else p1.x + 0.1
        theta_w = wall_angle_at(wall, x_w)
        if side == "upper":
            nu_w = theta_w - K
        else:
            nu_w = K - theta_w

        if nu_w > 0:
            M_w = inverse_prandtl_meyer(nu_w, gas.gamma)
            mu_w = float(mach_angle(M_w))
            if side == "upper":
                avg_slope = math.tan((p1.theta + theta_w) / 2 + (mu1 + mu_w) / 2)
            else:
                avg_slope = math.tan((p1.theta + theta_w) / 2 - (mu1 + mu_w) / 2)
            if abs(avg_slope) > 1e-15:
                x_w = p1.x + (wall_y_at(wall, x_w) - p1.y) / avg_slope
                slope = avg_slope

    y_w = wall_y_at(wall, x_w)
    theta_w = wall_angle_at(wall, x_w)

    if side == "upper":
        nu_w = theta_w - K
    else:
        nu_w = K - theta_w

    nu_w = max(nu_w, 1e-10)
    M_w = inverse_prandtl_meyer(nu_w, gas.gamma)
    p_w, T_w = _compute_thermo(M_w, gas, p0, T0)

    return CharPoint(x=x_w, y=y_w, mach=M_w, theta=theta_w, nu=nu_w, p=p_w, T=T_w, kind="wall")


def axis_point(
    p1: CharPoint,
    gas: GasProperties,
    p0: float = 1.0,
    T0: float = 1.0,
) -> CharPoint:
    """Compute a point on the symmetry axis (y=0, theta=0).

    p1 sends a C+ characteristic to the axis. K- = theta - nu is preserved.
    """
    K_minus = p1.theta - p1.nu
    theta_a = 0.0
    nu_a = -K_minus

    nu_a = max(nu_a, 1e-10)
    M_a = inverse_prandtl_meyer(nu_a, gas.gamma)

    mu1 = float(mach_angle(p1.mach))
    slope = math.tan(p1.theta - mu1)
    if abs(slope) > 1e-15:
        x_a = p1.x + (0.0 - p1.y) / slope
    else:
        x_a = p1.x + 0.1

    p_a, T_a = _compute_thermo(M_a, gas, p0, T0)

    return CharPoint(x=x_a, y=0.0, mach=M_a, theta=theta_a, nu=nu_a, p=p_a, T=T_a, kind="axis")
