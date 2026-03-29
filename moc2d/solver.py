"""MOC 2D solver — x-marching orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from moc2d.config import SimulationConfig
from moc2d.gas import prandtl_meyer, pressure_ratio, temperature_ratio
import math

from moc2d.gas import inverse_prandtl_meyer, mach_angle
from moc2d.geometry import find_compressive_corners, find_expansion_corners, wall_y_at
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


def solve(config: SimulationConfig) -> SolverResult:
    """Run the MOC 2D solver with x-marching.

    Connectivity tracking uses per-family lists:
    - cm_families[j] tracks the C- line emanating from inlet point j (going downward)
    - cp_families[j] tracks the C+ line emanating from inlet point j (going upward)

    Convention: points in a layer are ordered high-y to low-y.
    For pair (i, i+1): p_i is upper (on C-), p_{i+1} is lower (on C+).
    The interior point between them is on:
      - C- family of p_i (C- continues downward from the upper point)
      - C+ family of p_{i+1} (C+ continues upward from the lower point)
    """
    t_start = time.perf_counter()

    gas = config.gas
    geom_type = config.geometry_type
    p0 = config.inlet.p0
    T0 = config.inlet.T0

    upper_wall, lower_wall = _find_upper_lower_walls(config)
    x_max = max(p.x for w in config.walls for p in w.points)

    corners_lower = find_compressive_corners(lower_wall) if lower_wall else []
    corners_upper = find_compressive_corners(upper_wall) if upper_wall else []
    exp_corners_lower = find_expansion_corners(lower_wall) if lower_wall else []
    exp_corners_upper = find_expansion_corners(upper_wall) if upper_wall else []
    processed_corners: set[tuple[float, float]] = set()

    result = SolverResult(config=config)

    current_layer = _init_inlet(config)
    y_values_inlet = [wall.points[0].y for wall in config.walls]
    y_min_inlet = min(y_values_inlet)
    y_max_inlet = max(y_values_inlet)

    # Store indices of current layer points in result.char_points
    layer_indices: list[int] = []
    for pt in current_layer:
        layer_indices.append(len(result.char_points))
        result.char_points.append(pt)

    n = len(current_layer)

    # Per-family line tracking.
    # cm_family_of[j] = index into cm_families for the C- line passing through layer point j
    # cp_family_of[j] = index into cp_families for the C+ line passing through layer point j
    # At the inlet, each point starts its own C- and C+ family.
    cm_families: list[list[int]] = [[layer_indices[j]] for j in range(n)]
    cp_families: list[list[int]] = [[layer_indices[j]] for j in range(n)]
    cm_family_of: list[int] = list(range(n))  # point j is on C- family j
    cp_family_of: list[int] = list(range(n))  # point j is on C+ family j

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
        new_cm_family_of: list[int] = []
        new_cp_family_of: list[int] = []

        # Interior points — p1=layer[i] is upper (on C-), p2=layer[i+1] is lower (on C+)
        for i in range(len(current_layer) - 1):
            p1 = current_layer[i]
            p2 = current_layer[i + 1]
            try:
                p3 = interior_point(p1, p2, gas, geom_type, p0, T0)
                if p3.x <= min(p1.x, p2.x):
                    continue
                if p3.mach < 1.0:
                    continue

                # Boundary condition check: if point is outside the domain,
                # replace it with a wall point (proper BC, not clamping).
                y_upper = wall_y_at(upper_wall, p3.x) if upper_wall else 1e10
                y_lower = wall_y_at(lower_wall, p3.x) if lower_wall else -1e10

                if p3.y > y_upper and upper_wall:
                    # C+ from p2 has reached the upper wall — compute wall point
                    p3 = wall_point(p2, upper_wall, gas, geom_type, "upper", p0, T0)
                    if p3.x <= min(p1.x, p2.x) or p3.mach < 1.0:
                        continue
                    p3 = CharPoint(
                        x=p3.x, y=p3.y, mach=p3.mach, theta=p3.theta,
                        nu=p3.nu, p=p3.p, T=p3.T, kind="wall",
                    )
                elif p3.y < y_lower and lower_wall:
                    # C- from p1 has reached the lower wall — compute wall point
                    p3 = wall_point(p1, lower_wall, gas, geom_type, "lower", p0, T0)
                    if p3.x <= min(p1.x, p2.x) or p3.mach < 1.0:
                        continue
                    p3 = CharPoint(
                        x=p3.x, y=p3.y, mach=p3.mach, theta=p3.theta,
                        nu=p3.nu, p=p3.p, T=p3.T, kind="wall",
                    )

                idx = len(result.char_points)
                result.char_points.append(p3)
                new_layer.append(p3)
                new_indices.append(idx)

                # C- from p1 (upper) continues through p3
                cm_fam = cm_family_of[i]
                cm_families[cm_fam].append(idx)
                new_cm_family_of.append(cm_fam)

                # C+ from p2 (lower) continues through p3
                cp_fam = cp_family_of[i + 1]
                cp_families[cp_fam].append(idx)
                new_cp_family_of.append(cp_fam)

            except (ValueError, ZeroDivisionError, RuntimeError):
                continue

        # Upper wall point — C+ from topmost interior reaches the wall
        if upper_wall and new_layer:
            try:
                pw = wall_point(new_layer[0], upper_wall, gas, geom_type, "upper", p0, T0)
                if pw.x > min(p.x for p in current_layer) and pw.mach >= 1.0:
                    idx = len(result.char_points)
                    result.char_points.append(pw)
                    new_layer.insert(0, pw)
                    new_indices.insert(0, idx)

                    # The C+ line from the topmost interior terminates at the wall
                    cp_fam = new_cp_family_of[0] if new_cp_family_of else -1
                    if cp_fam >= 0:
                        cp_families[cp_fam].append(idx)

                    # Start a NEW C- family from the wall reflection
                    new_cm_fam = len(cm_families)
                    cm_families.append([idx])
                    new_cm_family_of.insert(0, new_cm_fam)

                    # The wall point starts a new C+ family (for tracking)
                    new_cp_fam = len(cp_families)
                    cp_families.append([idx])
                    new_cp_family_of.insert(0, new_cp_fam)
            except (ValueError, ZeroDivisionError, RuntimeError):
                pass

        # Lower wall point or axis point — C- from bottommost interior reaches the wall
        if lower_wall and new_layer:
            if geom_type.kind == "axisymmetric" and lower_wall.name == "axis":
                try:
                    pa = axis_point(new_layer[-1], gas, p0, T0)
                    if pa.x > min(p.x for p in current_layer) and pa.mach >= 1.0:
                        idx = len(result.char_points)
                        result.char_points.append(pa)
                        new_layer.append(pa)
                        new_indices.append(idx)

                        cm_fam = new_cm_family_of[-1] if new_cm_family_of else -1
                        if cm_fam >= 0:
                            cm_families[cm_fam].append(idx)
                        new_cp_fam = len(cp_families)
                        cp_families.append([idx])
                        new_cp_family_of.append(new_cp_fam)
                        new_cm_fam = len(cm_families)
                        cm_families.append([idx])
                        new_cm_family_of.append(new_cm_fam)
                except (ValueError, ZeroDivisionError, RuntimeError):
                    pass
            else:
                try:
                    pw = wall_point(new_layer[-1], lower_wall, gas, geom_type, "lower", p0, T0)
                    if pw.x > min(p.x for p in current_layer) and pw.mach >= 1.0:
                        idx = len(result.char_points)
                        result.char_points.append(pw)
                        new_layer.append(pw)
                        new_indices.append(idx)

                        # C- from bottommost interior terminates at wall
                        cm_fam = new_cm_family_of[-1] if new_cm_family_of else -1
                        if cm_fam >= 0:
                            cm_families[cm_fam].append(idx)

                        # Start new C+ family from wall reflection
                        new_cp_fam = len(cp_families)
                        cp_families.append([idx])
                        new_cp_family_of.append(new_cp_fam)

                        # Start new C- family
                        new_cm_fam = len(cm_families)
                        cm_families.append([idx])
                        new_cm_family_of.append(new_cm_fam)
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

        # Detect and generate Prandtl-Meyer expansion fans at expansion corners
        for exp_corners, is_lower_wall in [
            (exp_corners_lower, True),
            (exp_corners_upper, False),
        ]:
            for cx, cy, d_theta in exp_corners:
                key = ("exp", cx, cy)
                if key in processed_corners:
                    continue
                if avg_x >= cx:
                    processed_corners.add(key)
                    # Find the upstream state at the corner
                    closest = min(
                        current_layer,
                        key=lambda pt: (pt.x - cx)**2 + (pt.y - cy)**2,
                    )
                    M_up = closest.mach
                    theta_up = closest.theta
                    nu_up = closest.nu

                    # Generate n_fan points along the Prandtl-Meyer fan.
                    # Each fan ray is a C+ characteristic (for lower wall) or C-
                    # (for upper wall) emanating from the corner at a specific angle.
                    # Place each point at a small distance ds along its ray.
                    n_fan = max(10, config.n_char_lines // 2)
                    # ds scales with the domain height for proper spacing
                    ds = (y_max_inlet - y_min_inlet) / n_fan * 0.8

                    for j in range(n_fan):
                        frac = (j + 1) / (n_fan + 1)
                        delta = frac * d_theta
                        if is_lower_wall:
                            theta_fan = theta_up - delta
                        else:
                            theta_fan = theta_up + delta
                        nu_fan = nu_up + delta
                        M_fan = inverse_prandtl_meyer(float(nu_fan), gas.gamma)
                        if M_fan < 1.0:
                            continue
                        mu_fan = float(mach_angle(M_fan))
                        p_fan = p0 * float(pressure_ratio(M_fan, gas.gamma))
                        T_fan = T0 * float(temperature_ratio(M_fan, gas.gamma))

                        # Place point along the fan ray (C+ for lower wall, C- for upper)
                        if is_lower_wall:
                            ray_angle = theta_fan + mu_fan  # C+ goes upward from lower corner
                        else:
                            ray_angle = theta_fan - mu_fan  # C- goes downward from upper corner
                        x_fan = cx + ds * (j + 1) * math.cos(ray_angle)
                        y_fan = cy + ds * (j + 1) * math.sin(ray_angle)

                        fan_pt = CharPoint(
                            x=x_fan, y=y_fan, mach=M_fan, theta=float(theta_fan),
                            nu=float(nu_fan), p=p_fan, T=T_fan, kind="interior",
                        )
                        idx = len(result.char_points)
                        result.char_points.append(fan_pt)

                        if is_lower_wall:
                            new_layer.append(fan_pt)
                            new_indices.append(idx)
                            new_cm_fam = len(cm_families)
                            cm_families.append([idx])
                            new_cm_family_of.append(new_cm_fam)
                            new_cp_fam = len(cp_families)
                            cp_families.append([idx])
                            new_cp_family_of.append(new_cp_fam)
                        else:
                            new_layer.insert(0, fan_pt)
                            new_indices.insert(0, idx)
                            new_cm_fam = len(cm_families)
                            cm_families.append([idx])
                            new_cm_family_of.insert(0, new_cm_fam)
                            new_cp_fam = len(cp_families)
                            cp_families.append([idx])
                            new_cp_family_of.insert(0, new_cp_fam)

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
        cm_family_of = new_cm_family_of
        cp_family_of = new_cp_family_of

    # Build line lists from families (filter out single-point families)
    result.c_minus_lines = [fam for fam in cm_families if len(fam) > 1]
    result.c_plus_lines = [fam for fam in cp_families if len(fam) > 1]
    result.shock_lines = [t for t in shock_idx_trackers if len(t) > 1]

    result.wall_time = time.perf_counter() - t_start
    return result
