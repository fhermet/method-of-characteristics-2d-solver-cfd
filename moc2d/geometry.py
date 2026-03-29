"""Wall geometry: interpolation, intersection, corner detection."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

from moc2d.config import WallDefinition


def _get_xy(wall: WallDefinition) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([p.x for p in wall.points])
    ys = np.array([p.y for p in wall.points])
    return xs, ys


def _find_segment(xs: np.ndarray, x: float) -> int:
    idx = int(np.searchsorted(xs, x, side="right")) - 1
    return max(0, min(idx, len(xs) - 2))


def wall_y_at(wall: WallDefinition, x: float) -> float:
    """Y-coordinate of the wall at position x."""
    xs, ys = _get_xy(wall)
    if wall.interpolation == "cubic" and len(xs) >= 3:
        cs = CubicSpline(xs, ys)
        return float(cs(x))
    i = _find_segment(xs, x)
    dx = xs[i + 1] - xs[i]
    if abs(dx) < 1e-15:
        return float(ys[i])
    t = (x - xs[i]) / dx
    return float(ys[i] + t * (ys[i + 1] - ys[i]))


def wall_angle_at(wall: WallDefinition, x: float) -> float:
    """Wall angle (radians) at position x."""
    xs, ys = _get_xy(wall)
    if wall.interpolation == "cubic" and len(xs) >= 3:
        cs = CubicSpline(xs, ys)
        return float(np.arctan(cs(x, 1)))
    i = _find_segment(xs, x)
    dx = xs[i + 1] - xs[i]
    dy = ys[i + 1] - ys[i]
    if abs(dx) < 1e-15:
        return 0.0
    return float(math.atan2(dy, dx))


def intersect_char_wall(x0: float, y0: float, angle: float, wall: WallDefinition) -> tuple[float, float]:
    """Intersection of a characteristic line with a wall. Newton iteration."""
    tan_a = math.tan(angle)
    x_guess = x0
    for _ in range(20):
        y_wall = wall_y_at(wall, x_guess)
        f_val = y0 + tan_a * (x_guess - x0) - y_wall
        dx = 1e-8
        dwall = (wall_y_at(wall, x_guess + dx) - wall_y_at(wall, x_guess - dx)) / (2 * dx)
        f_deriv = tan_a - dwall
        if abs(f_deriv) < 1e-15:
            break
        x_new = x_guess - f_val / f_deriv
        if abs(x_new - x_guess) < 1e-12:
            x_guess = x_new
            break
        x_guess = x_new
    y_result = wall_y_at(wall, x_guess)
    return float(x_guess), float(y_result)


def find_compressive_corners(wall: WallDefinition) -> list[tuple[float, float, float]]:
    """Detect compressive corners on a wall (for shock generation).

    Returns list of (x, y, delta_theta) where delta_theta > 0 means compression.
    """
    if wall.interpolation == "cubic":
        return []
    xs, ys = _get_xy(wall)
    corners = []
    is_lower = wall.name in ("lower", "axis")
    for i in range(1, len(xs) - 1):
        dx_before = xs[i] - xs[i - 1]
        dy_before = ys[i] - ys[i - 1]
        dx_after = xs[i + 1] - xs[i]
        dy_after = ys[i + 1] - ys[i]
        angle_before = math.atan2(dy_before, dx_before) if abs(dx_before) > 1e-15 else 0.0
        angle_after = math.atan2(dy_after, dx_after) if abs(dx_after) > 1e-15 else 0.0
        delta = angle_after - angle_before
        if is_lower and delta > 1e-8:
            corners.append((float(xs[i]), float(ys[i]), float(delta)))
        elif not is_lower and delta < -1e-8:
            corners.append((float(xs[i]), float(ys[i]), float(abs(delta))))
    return corners


def find_expansion_corners(wall: WallDefinition) -> list[tuple[float, float, float]]:
    """Detect expansion corners on a wall (for Prandtl-Meyer fan generation).

    An expansion corner is where the wall turns AWAY from the flow.
    Returns list of (x, y, delta_theta) where delta_theta > 0 is the expansion angle.
    """
    if wall.interpolation == "cubic":
        return []
    xs, ys = _get_xy(wall)
    corners = []
    is_lower = wall.name in ("lower", "axis")
    for i in range(1, len(xs) - 1):
        dx_before = xs[i] - xs[i - 1]
        dy_before = ys[i] - ys[i - 1]
        dx_after = xs[i + 1] - xs[i]
        dy_after = ys[i + 1] - ys[i]
        angle_before = math.atan2(dy_before, dx_before) if abs(dx_before) > 1e-15 else 0.0
        angle_after = math.atan2(dy_after, dx_after) if abs(dx_after) > 1e-15 else 0.0
        delta = angle_after - angle_before
        # Expansion: lower wall turns downward (delta < 0), upper wall turns upward (delta > 0)
        if is_lower and delta < -1e-8:
            corners.append((float(xs[i]), float(ys[i]), float(abs(delta))))
        elif not is_lower and delta > 1e-8:
            corners.append((float(xs[i]), float(ys[i]), float(delta)))
    return corners
