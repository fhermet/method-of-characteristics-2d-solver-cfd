"""Results post-processing: grid interpolation and field extraction."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from moc2d.solver import SolverResult


def interpolate_to_grid(result, nx=100, ny=50):
    """Interpolate characteristic network data onto a regular grid.
    Returns dict with keys: x, y, mach, p, T, theta.
    """
    pts = result.char_points
    if not pts:
        return {
            "x": np.array([]), "y": np.array([]),
            "mach": np.array([[]]), "p": np.array([[]]),
            "T": np.array([[]]), "theta": np.array([[]]),
        }

    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    points = np.column_stack([xs, ys])

    x_grid = np.linspace(xs.min(), xs.max(), nx)
    y_grid = np.linspace(ys.min(), ys.max(), ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    fields = {}
    for name, values in [
        ("mach", [p.mach for p in pts]),
        ("p", [p.p for p in pts]),
        ("T", [p.T for p in pts]),
        ("theta", [p.theta for p in pts]),
    ]:
        vals = np.array(values)
        fields[name] = griddata(points, vals, (X, Y), method="linear")

    return {"x": x_grid, "y": y_grid, **fields}


def extract_wall_data(result):
    """Extract data at wall points from the characteristic network."""
    wall_pts = [p for p in result.char_points if p.kind == "wall"]
    if not wall_pts:
        wall_pts = result.char_points[:10] if result.char_points else []

    wall_pts.sort(key=lambda p: p.x)
    return {
        "x": np.array([p.x for p in wall_pts]),
        "y": np.array([p.y for p in wall_pts]),
        "mach": np.array([p.mach for p in wall_pts]),
        "p": np.array([p.p for p in wall_pts]),
        "T": np.array([p.T for p in wall_pts]),
        "theta": np.array([p.theta for p in wall_pts]),
    }
