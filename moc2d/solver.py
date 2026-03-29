"""MOC 2D solver — x-marching orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from moc2d.config import SimulationConfig
from moc2d.gas import prandtl_meyer, pressure_ratio, temperature_ratio
from moc2d.geometry import find_compressive_corners
from moc2d.characteristics import CharPoint, interior_point, wall_point, axis_point
from moc2d.shocks import ShockPoint, create_shock, propagate_shock


@dataclass
class SolverResult:
    """Container for all solver output."""
    char_points: list[CharPoint] = field(default_factory=list)
    shock_points: list[ShockPoint] = field(default_factory=list)
    c_plus_lines: list[list[int]] = field(default_factory=list)
    c_minus_lines: list[list[int]] = field(default_factory=list)
    shock_lines: list[list[int]] = field(default_factory=list)
    config: SimulationConfig | None = None
    wall_time: float = 0.0


def _init_inlet(config: SimulationConfig) -> list[CharPoint]:
    """Generate initial characteristic points along the inlet section."""
    gas = config.gas
    inlet = config.inlet
    M = inlet.mach
    theta = inlet.theta
    nu = float(prandtl_meyer(M, gas.gamma))
    p = inlet.p0 * float(pressure_ratio(M, gas.gamma))
    T = inlet.T0 * float(temperature_ratio(M, gas.gamma))

    y_values = [wall.points[0].y for wall in config.walls]
    y_min = min(y_values)
    y_max = max(y_values)

    n = config.n_char_lines
    points = []
    # Points ordered from high y to low y: p1 (upper) before p2 (lower)
    for i in range(n):
        y = y_max - (y_max - y_min) * (i + 0.5) / n
        points.append(
            CharPoint(x=0.0, y=y, mach=M, theta=theta, nu=nu, p=p, T=T, kind="interior")
        )
    return points


def _find_upper_lower_walls(config: SimulationConfig):
    """Extract upper and lower wall definitions from config."""
    upper = None
    lower = None
    for wall in config.walls:
        if wall.name == "upper":
            upper = wall
        elif wall.name in ("lower", "axis"):
            lower = wall
    return upper, lower


def _build_lines(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Build connected lines from a list of (from, to) edge pairs."""
    if not edges:
        return []
    # Build adjacency: for each node, what comes next
    next_node: dict[int, int] = {}
    has_predecessor: set[int] = set()
    for a, b in edges:
        next_node[a] = b
        has_predecessor.add(b)

    # Find line starts (nodes with no predecessor in this edge set)
    starts = [a for a, _ in edges if a not in has_predecessor]
    # Fallback: if all nodes have predecessors (cycle), just use first edge start
    if not starts and edges:
        starts = [edges[0][0]]

    visited: set[int] = set()
    lines = []
    for s in starts:
        if s in visited:
            continue
        line = [s]
        visited.add(s)
        node = s
        while node in next_node and next_node[node] not in visited:
            node = next_node[node]
            line.append(node)
            visited.add(node)
        if len(line) > 1:
            lines.append(line)
    return lines


def solve(config: SimulationConfig) -> SolverResult:
    """Run the MOC 2D solver with x-marching."""
    t_start = time.perf_counter()

    gas = config.gas
    geom_type = config.geometry_type
    p0 = config.inlet.p0
    T0 = config.inlet.T0

    upper_wall, lower_wall = _find_upper_lower_walls(config)
    x_max = max(p.x for w in config.walls for p in w.points)

    corners_lower = find_compressive_corners(lower_wall) if lower_wall else []
    corners_upper = find_compressive_corners(upper_wall) if upper_wall else []
    processed_corners: set[tuple[float, float]] = set()

    result = SolverResult(config=config)

    # Edge lists for connectivity: (from_idx, to_idx)
    cp_edges: list[tuple[int, int]] = []  # C+ line edges
    cm_edges: list[tuple[int, int]] = []  # C- line edges

    current_layer = _init_inlet(config)
    # Store indices of current layer points in result.char_points
    layer_indices: list[int] = []
    for pt in current_layer:
        layer_indices.append(len(result.char_points))
        result.char_points.append(pt)

    active_shocks: list[list[ShockPoint]] = []
    shock_idx_trackers: list[list[int]] = []

    max_layers = config.n_char_lines * 4

    for layer_num in range(max_layers):
        if not current_layer or len(current_layer) < 2:
            break

        avg_x = sum(pt.x for pt in current_layer) / len(current_layer)
        if avg_x > x_max:
            break

        new_layer: list[CharPoint] = []
        new_indices: list[int] = []

        # Interior points — p1 is upper (on C-), p2 is lower (on C+)
        for i in range(len(current_layer) - 1):
            p1 = current_layer[i]
            p2 = current_layer[i + 1]
            try:
                p3 = interior_point(p1, p2, gas, geom_type, p0, T0)
                if p3.x <= p1.x or p3.x <= p2.x:
                    continue
                if p3.mach < 1.0:
                    continue
                idx = len(result.char_points)
                result.char_points.append(p3)
                new_layer.append(p3)
                new_indices.append(idx)

                # C+ connects p1 (upper) to p3 (C+ goes from lower-left to upper-right,
                # but in layer progression it connects the upper source to the new point)
                cp_edges.append((layer_indices[i], idx))
                # C- connects p2 (lower) to p3
                cm_edges.append((layer_indices[i + 1], idx))
            except (ValueError, ZeroDivisionError, RuntimeError):
                continue

        # Upper wall point
        if upper_wall and new_layer:
            try:
                pw = wall_point(new_layer[0], upper_wall, gas, geom_type, "upper", p0, T0)
                if pw.x > current_layer[0].x and pw.mach >= 1.0:
                    idx = len(result.char_points)
                    result.char_points.append(pw)
                    new_layer.insert(0, pw)
                    new_indices.insert(0, idx)
                    # C+ connects topmost interior point to wall
                    cp_edges.append((new_indices[1], idx))
            except (ValueError, ZeroDivisionError, RuntimeError):
                pass

        # Lower wall point or axis point
        if lower_wall and new_layer:
            if geom_type.kind == "axisymmetric" and lower_wall.name == "axis":
                try:
                    pa = axis_point(new_layer[-1], gas, p0, T0)
                    if pa.x > current_layer[-1].x and pa.mach >= 1.0:
                        idx = len(result.char_points)
                        result.char_points.append(pa)
                        new_layer.append(pa)
                        new_indices.append(idx)
                        cm_edges.append((new_indices[-2], idx))
                except (ValueError, ZeroDivisionError, RuntimeError):
                    pass
            else:
                try:
                    pw = wall_point(new_layer[-1], lower_wall, gas, geom_type, "lower", p0, T0)
                    if pw.x > current_layer[-1].x and pw.mach >= 1.0:
                        idx = len(result.char_points)
                        result.char_points.append(pw)
                        new_layer.append(pw)
                        new_indices.append(idx)
                        cm_edges.append((new_indices[-2], idx))
                except (ValueError, ZeroDivisionError, RuntimeError):
                    pass

        # Detect and create new shocks at compressive corners
        for corners in [corners_lower, corners_upper]:
            for cx, cy, d_theta in corners:
                key = (cx, cy)
                if key in processed_corners:
                    continue
                if avg_x >= cx:
                    processed_corners.add(key)
                    closest = min(current_layer, key=lambda pt: (pt.x - cx)**2 + (pt.y - cy)**2)
                    corner_pt = CharPoint(
                        x=cx, y=cy, mach=closest.mach, theta=closest.theta,
                        nu=closest.nu, p=closest.p, T=closest.T, kind="interior",
                    )
                    try:
                        sp = create_shock(corner_pt, d_theta, gas)
                        shock_idx = len(result.shock_points)
                        result.shock_points.append(sp)
                        active_shocks.append([sp])
                        shock_idx_trackers.append([shock_idx])
                    except (ValueError, ZeroDivisionError):
                        pass

        # Propagate active shocks forward one step
        for i, shock_line in enumerate(active_shocks):
            if not shock_line:
                continue
            last_sp = shock_line[-1]
            if not new_layer:
                continue
            upstream = min(
                new_layer,
                key=lambda pt: (pt.x - last_sp.x)**2 + (pt.y - last_sp.y)**2,
            )
            try:
                new_sp = propagate_shock(last_sp, upstream, gas, ds=0.1)
                if new_sp.x > last_sp.x:
                    shock_idx = len(result.shock_points)
                    result.shock_points.append(new_sp)
                    shock_line.append(new_sp)
                    shock_idx_trackers[i].append(shock_idx)
            except (ValueError, ZeroDivisionError):
                pass

        current_layer = new_layer
        layer_indices = new_indices

    # Build characteristic lines from edge lists
    result.c_plus_lines = _build_lines(cp_edges)
    result.c_minus_lines = _build_lines(cm_edges)
    result.shock_lines = [t for t in shock_idx_trackers if len(t) > 1]

    result.wall_time = time.perf_counter() - t_start
    return result
