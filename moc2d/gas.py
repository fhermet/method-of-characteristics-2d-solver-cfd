"""Perfect gas relations for supersonic flow.

All functions are pure, stateless, and accept scalars or NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def mach_angle(M):
    """Mach angle mu = arcsin(1/M)."""
    return np.arcsin(1.0 / M)


def prandtl_meyer(M, gamma):
    """Prandtl-Meyer angle nu(M) in radians."""
    gp = (gamma + 1) / (gamma - 1)
    Msq_m1 = M * M - 1.0
    return np.sqrt(gp) * np.arctan(np.sqrt(Msq_m1 / gp)) - np.arctan(np.sqrt(Msq_m1))


def inverse_prandtl_meyer(nu, gamma, tol=1e-10, max_iter=50):
    """Inverse Prandtl-Meyer: find M given nu(M) using Newton-Raphson."""
    if nu <= 0:
        return 1.0
    M = 1.0 + nu
    for _ in range(max_iter):
        nu_M = prandtl_meyer(M, gamma)
        Msq = M * M
        dnu_dM = np.sqrt(Msq - 1.0) / (1.0 + (gamma - 1.0) / 2.0 * Msq) / M
        dM = (nu - nu_M) / dnu_dM
        M = M + dM
        if abs(dM) < tol:
            break
    return float(M)


def pressure_ratio(M, gamma):
    """Isentropic pressure ratio p/p0."""
    return (1.0 + (gamma - 1.0) / 2.0 * M * M) ** (-gamma / (gamma - 1.0))


def temperature_ratio(M, gamma):
    """Isentropic temperature ratio T/T0."""
    return (1.0 + (gamma - 1.0) / 2.0 * M * M) ** (-1.0)


def density_ratio(M, gamma):
    """Isentropic density ratio rho/rho0."""
    return (1.0 + (gamma - 1.0) / 2.0 * M * M) ** (-1.0 / (gamma - 1.0))


def sound_speed(T, gamma, R=287.0):
    """Speed of sound a = sqrt(gamma * R * T)."""
    return np.sqrt(gamma * R * T)


def stagnation_from_static(M, p, T, gamma):
    """Compute stagnation conditions (p0, T0) from static state and Mach."""
    p0 = p / pressure_ratio(M, gamma)
    T0 = T / temperature_ratio(M, gamma)
    return float(p0), float(T0)


def oblique_shock_beta(M, theta, gamma, strong=False, tol=1e-10, max_iter=50):
    """Oblique shock wave angle beta from theta-beta-M relation.

    Returns weak shock solution by default, strong if strong=True.
    """
    if abs(theta) < 1e-12:
        return float(mach_angle(M))

    Msq = M * M

    beta_min = float(mach_angle(M)) + 1e-6
    beta_max = np.pi / 2 - 1e-6

    def theta_beta(beta):
        sb = np.sin(beta)
        cb = np.cos(beta)
        num = Msq * sb * sb - 1.0
        den = Msq * (gamma + np.cos(2 * beta)) + 2.0
        return np.arctan(2.0 * cb / sb * num / den)

    # Find beta at max deflection by bisection
    b_lo, b_hi = beta_min, beta_max
    for _ in range(60):
        b_mid = (b_lo + b_hi) / 2
        db = 1e-8
        dtheta = theta_beta(b_mid + db) - theta_beta(b_mid - db)
        if dtheta > 0:
            b_lo = b_mid
        else:
            b_hi = b_mid
    beta_max_defl = (b_lo + b_hi) / 2

    if strong:
        b_lo_search, b_hi_search = beta_max_defl, beta_max
    else:
        b_lo_search, b_hi_search = beta_min, beta_max_defl

    # Newton-Raphson with bisection fallback
    beta = (b_lo_search + b_hi_search) / 2
    for _ in range(max_iter):
        f_val = theta_beta(beta) - theta
        db = 1e-8
        f_deriv = (theta_beta(beta + db) - theta_beta(beta - db)) / (2 * db)
        if abs(f_deriv) < 1e-15:
            break
        d_beta = -f_val / f_deriv
        beta_new = beta + d_beta
        if beta_new < b_lo_search or beta_new > b_hi_search:
            beta_new = (b_lo_search + b_hi_search) / 2
        if abs(d_beta) < tol:
            beta = beta_new
            break
        if theta_beta(beta_new) - theta > 0:
            b_hi_search = beta_new
        else:
            b_lo_search = beta_new
        beta = beta_new
    return float(beta)


def post_shock_state(M1, beta, gamma):
    """Compute post-shock state from upstream Mach and shock angle.

    Returns: (M2, p2/p1, T2/T1, rho2/rho1, theta).
    """
    sb = np.sin(beta)
    cb = np.cos(beta)
    Mn1 = M1 * sb
    Mn1sq = Mn1 * Mn1

    p_ratio_val = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Mn1sq - 1.0)
    rho_ratio_val = (gamma + 1.0) * Mn1sq / ((gamma - 1.0) * Mn1sq + 2.0)
    T_ratio_val = p_ratio_val / rho_ratio_val
    Mn2sq = ((gamma - 1.0) * Mn1sq + 2.0) / (2.0 * gamma * Mn1sq - (gamma - 1.0))

    theta = np.arctan(
        2.0 * cb / sb * (Mn1sq - 1.0) / (M1 * M1 * (gamma + np.cos(2.0 * beta)) + 2.0)
    )

    M2 = np.sqrt(Mn2sq) / np.sin(beta - theta)

    return float(M2), float(p_ratio_val), float(T_ratio_val), float(rho_ratio_val), float(theta)


def max_deflection(M, gamma):
    """Maximum flow deflection angle for attached oblique shock."""
    mu = float(mach_angle(M))
    Msq = M * M

    def theta_beta(beta):
        sb = np.sin(beta)
        cb = np.cos(beta)
        num = Msq * sb * sb - 1.0
        den = Msq * (gamma + np.cos(2 * beta)) + 2.0
        return np.arctan(2.0 * cb / sb * num / den)

    a, b = mu + 1e-6, np.pi / 2 - 1e-6
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in range(100):
        if theta_beta(c) < theta_beta(d):
            a = c
        else:
            b = d
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return float(theta_beta((a + b) / 2))
