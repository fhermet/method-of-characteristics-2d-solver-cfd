"""Tests for moc2d.shocks — shock fitting and interactions."""

import math
import pytest
from moc2d.config import GasProperties
from moc2d.characteristics import CharPoint
from moc2d.gas import prandtl_meyer, oblique_shock_beta, post_shock_state
from moc2d.shocks import ShockPoint, create_shock, propagate_shock, shock_shock_interaction


GAMMA = 1.4
GAS = GasProperties(gamma=GAMMA)


def _make_point(x, y, mach, theta, p=1.0, T=1.0):
    nu = prandtl_meyer(mach, GAMMA)
    return CharPoint(x=x, y=y, mach=mach, theta=theta, nu=float(nu), p=p, T=T, kind="interior")


class TestCreateShock:
    def test_simple_ramp_10deg(self):
        corner = _make_point(1.0, 0.0, 2.0, 0.0)
        sp = create_shock(corner, math.radians(10), GAS)
        assert sp.x == pytest.approx(1.0)
        assert sp.y == pytest.approx(0.0)
        assert math.degrees(sp.beta) == pytest.approx(39.31, abs=0.2)
        assert sp.M_downstream == pytest.approx(1.641, abs=0.02)
        assert sp.theta_downstream == pytest.approx(math.radians(10), abs=0.02)

    def test_shock_pressure_increase(self):
        corner = _make_point(1.0, 0.0, 3.0, 0.0)
        sp = create_shock(corner, math.radians(15), GAS)
        assert sp.p_ratio > 1.0


class TestPropagateShock:
    def test_uniform_upstream(self):
        corner = _make_point(0.0, 0.0, 2.0, 0.0)
        sp = create_shock(corner, math.radians(10), GAS)
        upstream = _make_point(0.5, 0.5, 2.0, 0.0)
        sp2 = propagate_shock(sp, upstream, GAS)
        assert sp2.beta == pytest.approx(sp.beta, abs=1e-4)
        assert sp2.M_downstream == pytest.approx(sp.M_downstream, abs=1e-4)
        assert sp2.x > sp.x


class TestShockShockInteraction:
    def test_symmetric_interaction(self):
        corner1 = _make_point(0.0, 0.0, 3.0, 0.0)
        corner2 = _make_point(0.0, 2.0, 3.0, 0.0)
        sp1 = create_shock(corner1, math.radians(10), GAS)
        sp2 = create_shock(corner2, math.radians(10), GAS)
        sp1 = ShockPoint(
            x=1.0, y=1.0, beta=sp1.beta,
            M_upstream=sp1.M_upstream, M_downstream=sp1.M_downstream,
            theta_downstream=sp1.theta_downstream, p_ratio=sp1.p_ratio,
        )
        sp2 = ShockPoint(
            x=1.0, y=1.0, beta=sp2.beta,
            M_upstream=sp2.M_upstream, M_downstream=sp2.M_downstream,
            theta_downstream=-sp2.theta_downstream, p_ratio=sp2.p_ratio,
        )
        out1, out2, slip = shock_shock_interaction(sp1, sp2, GAS)
        assert out1.p_ratio * sp1.p_ratio == pytest.approx(
            out2.p_ratio * sp2.p_ratio, abs=0.1
        )
