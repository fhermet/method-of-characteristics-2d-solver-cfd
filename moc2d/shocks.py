"""Shock fitting, propagation, and interaction for the MOC 2D solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

from moc2d.config import GasProperties
from moc2d.characteristics import CharPoint
from moc2d.gas import oblique_shock_beta, post_shock_state


@dataclass
class ShockPoint:
    """A point along a shock wave."""
    x: float
    y: float
    beta: float
    M_upstream: float
    M_downstream: float
    theta_downstream: float
    p_ratio: float


def create_shock(corner, delta_theta, gas):
    """Create a shock at a compressive corner."""
    M1 = corner.mach
    beta = oblique_shock_beta(M1, delta_theta, gas.gamma)
    M2, p_r, T_r, rho_r, theta = post_shock_state(M1, beta, gas.gamma)
    return ShockPoint(
        x=corner.x, y=corner.y, beta=beta,
        M_upstream=M1, M_downstream=M2,
        theta_downstream=corner.theta + delta_theta, p_ratio=p_r,
    )


def propagate_shock(shock_pt, upstream_state, gas, ds=0.1):
    """Propagate a shock one step using the local upstream state."""
    M1 = upstream_state.mach
    theta_up = upstream_state.theta
    delta_theta = shock_pt.theta_downstream - theta_up

    if delta_theta < 1e-10:
        return ShockPoint(
            x=shock_pt.x + ds,
            y=shock_pt.y + ds * math.tan(shock_pt.beta + theta_up),
            beta=float(math.asin(1.0 / M1)),
            M_upstream=M1, M_downstream=M1,
            theta_downstream=theta_up, p_ratio=1.0,
        )

    beta = oblique_shock_beta(M1, delta_theta, gas.gamma)
    M2, p_r, T_r, rho_r, _ = post_shock_state(M1, beta, gas.gamma)

    shock_angle = theta_up + beta
    x_new = shock_pt.x + ds * math.cos(shock_angle)
    y_new = shock_pt.y + ds * math.sin(shock_angle)

    return ShockPoint(
        x=x_new, y=y_new, beta=beta,
        M_upstream=M1, M_downstream=M2,
        theta_downstream=shock_pt.theta_downstream, p_ratio=p_r,
    )


def shock_shock_interaction(shock1, shock2, gas, tol=1e-6, max_iter=50):
    """Compute shock-shock interaction. Returns (out1, out2, slip_angle)."""
    M_r1 = shock1.M_downstream
    theta_r1 = shock1.theta_downstream
    p_r1 = shock1.p_ratio

    M_r2 = shock2.M_downstream
    theta_r2 = shock2.theta_downstream
    p_r2 = shock2.p_ratio

    delta3 = 0.01
    delta4 = 0.0
    for _ in range(max_iter):
        delta4 = (theta_r1 + delta3) - theta_r2
        if abs(delta3) < 1e-12 or abs(delta4) < 1e-12:
            break
        try:
            beta3 = oblique_shock_beta(M_r1, abs(delta3), gas.gamma)
            _, p3, _, _, _ = post_shock_state(M_r1, beta3, gas.gamma)
            beta4 = oblique_shock_beta(M_r2, abs(delta4), gas.gamma)
            _, p4, _, _, _ = post_shock_state(M_r2, beta4, gas.gamma)
        except (ValueError, ZeroDivisionError):
            break
        residual = p_r1 * p3 - p_r2 * p4
        if abs(residual) < tol:
            break
        dd = 1e-6
        try:
            beta3p = oblique_shock_beta(M_r1, abs(delta3 + dd), gas.gamma)
            _, p3p, _, _, _ = post_shock_state(M_r1, beta3p, gas.gamma)
            delta4p = (theta_r1 + delta3 + dd) - theta_r2
            beta4p = oblique_shock_beta(M_r2, abs(delta4p), gas.gamma)
            _, p4p, _, _, _ = post_shock_state(M_r2, beta4p, gas.gamma)
        except (ValueError, ZeroDivisionError):
            break
        dresidual = (p_r1 * p3p - p_r2 * p4p) - residual
        if abs(dresidual) < 1e-15:
            break
        delta3 -= residual / (dresidual / dd)

    beta3 = oblique_shock_beta(M_r1, abs(delta3), gas.gamma)
    M3, p3, _, _, _ = post_shock_state(M_r1, beta3, gas.gamma)
    beta4 = oblique_shock_beta(M_r2, abs(delta4), gas.gamma)
    M4, p4, _, _, _ = post_shock_state(M_r2, beta4, gas.gamma)

    slip_angle = theta_r1 + delta3

    out1 = ShockPoint(
        x=shock1.x, y=shock1.y, beta=beta3,
        M_upstream=M_r1, M_downstream=M3,
        theta_downstream=slip_angle, p_ratio=p3,
    )
    out2 = ShockPoint(
        x=shock2.x, y=shock2.y, beta=beta4,
        M_upstream=M_r2, M_downstream=M4,
        theta_downstream=slip_angle, p_ratio=p4,
    )
    return out1, out2, slip_angle


def shock_expansion_interaction(shock_pt, char_point, gas):
    """Weaken a shock when intersected by an expansion characteristic."""
    return propagate_shock(shock_pt, char_point, gas)
