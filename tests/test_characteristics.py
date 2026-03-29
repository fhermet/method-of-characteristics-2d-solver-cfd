"""Tests for moc2d.characteristics — unit processes."""

import math
import pytest
from moc2d.config import GasProperties, GeometryType, WallPoint, WallDefinition
from moc2d.characteristics import CharPoint, interior_point, wall_point, axis_point
from moc2d.gas import prandtl_meyer


GAMMA = 1.4
GAS = GasProperties(gamma=GAMMA)
PLANAR = GeometryType(kind="planar")


def _make_point(x, y, mach, theta, p=1.0, T=1.0, kind="interior"):
    nu = prandtl_meyer(mach, GAMMA)
    return CharPoint(x=x, y=y, mach=mach, theta=theta, nu=float(nu), p=p, T=T, kind=kind)


class TestInteriorPoint:
    def test_uniform_flow(self):
        p1 = _make_point(0, 1.0, 2.0, 0.0)
        p2 = _make_point(0, 0.5, 2.0, 0.0)
        p3 = interior_point(p1, p2, GAS, PLANAR)
        assert p3.mach == pytest.approx(2.0, abs=1e-4)
        assert p3.theta == pytest.approx(0.0, abs=1e-4)
        assert p3.x > 0

    def test_riemann_invariants_conserved(self):
        p1 = _make_point(0, 1.0, 2.5, math.radians(5))
        p2 = _make_point(0, 0.5, 2.0, math.radians(-3))
        p3 = interior_point(p1, p2, GAS, PLANAR)
        K_plus = p1.theta + p1.nu
        K_minus = p2.theta - p2.nu
        assert (p3.theta + p3.nu) == pytest.approx(K_plus, abs=1e-4)
        assert (p3.theta - p3.nu) == pytest.approx(K_minus, abs=1e-4)

    def test_position_downstream(self):
        p1 = _make_point(0, 1.0, 2.0, math.radians(5))
        p2 = _make_point(0, 0.5, 2.0, math.radians(-5))
        p3 = interior_point(p1, p2, GAS, PLANAR)
        assert p3.x > 0


class TestWallPoint:
    def test_flat_upper_wall(self):
        wall = WallDefinition(
            points=(WallPoint(0, 1), WallPoint(3, 1)), name="upper"
        )
        p1 = _make_point(0, 0.8, 2.0, 0.0)
        pw = wall_point(p1, wall, GAS, PLANAR, side="upper")
        assert pw.y == pytest.approx(1.0, abs=0.05)
        assert pw.theta == pytest.approx(0.0, abs=1e-4)
        assert pw.kind == "wall"


class TestAxisPoint:
    def test_theta_zero_on_axis(self):
        p1 = _make_point(0, 0.5, 2.0, math.radians(5))
        pa = axis_point(p1, GAS)
        assert pa.theta == pytest.approx(0.0, abs=1e-10)
        assert pa.y == pytest.approx(0.0, abs=1e-10)
        assert pa.kind == "axis"

    def test_riemann_invariant(self):
        p1 = _make_point(0, 0.5, 2.0, math.radians(5))
        pa = axis_point(p1, GAS)
        K_minus = p1.theta - p1.nu
        assert (pa.theta - pa.nu) == pytest.approx(K_minus, abs=1e-4)
