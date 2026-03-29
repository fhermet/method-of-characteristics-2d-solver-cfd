"""Tests for moc2d.config dataclasses."""

import math
import pytest
from moc2d.config import (
    GasProperties,
    WallPoint,
    WallDefinition,
    InletCondition,
    GeometryType,
    SimulationConfig,
)


class TestGasProperties:
    def test_default_gamma(self):
        gas = GasProperties()
        assert gas.gamma == 1.4

    def test_custom_gamma(self):
        gas = GasProperties(gamma=1.3)
        assert gas.gamma == 1.3

    def test_frozen(self):
        gas = GasProperties()
        with pytest.raises(AttributeError):
            gas.gamma = 1.3


class TestWallPoint:
    def test_creation(self):
        wp = WallPoint(x=0.0, y=1.0)
        assert wp.x == 0.0
        assert wp.y == 1.0
        assert wp.theta is None

    def test_with_theta(self):
        wp = WallPoint(x=0.0, y=1.0, theta=math.radians(10))
        assert wp.theta == pytest.approx(math.radians(10))

    def test_frozen(self):
        wp = WallPoint(x=0.0, y=1.0)
        with pytest.raises(AttributeError):
            wp.x = 1.0


class TestWallDefinition:
    def test_linear_default(self):
        pts = (WallPoint(0, 1), WallPoint(1, 1))
        wall = WallDefinition(points=pts, name="upper")
        assert wall.interpolation == "linear"
        assert wall.name == "upper"

    def test_cubic(self):
        pts = (WallPoint(0, 1), WallPoint(0.5, 1.2), WallPoint(1, 1.5))
        wall = WallDefinition(points=pts, name="upper", interpolation="cubic")
        assert wall.interpolation == "cubic"


class TestInletCondition:
    def test_defaults(self):
        inlet = InletCondition(mach=2.0)
        assert inlet.mach == 2.0
        assert inlet.theta == 0.0
        assert inlet.p0 == 1.0
        assert inlet.T0 == 1.0


class TestGeometryType:
    def test_planar(self):
        gt = GeometryType(kind="planar")
        assert gt.kind == "planar"

    def test_axisymmetric(self):
        gt = GeometryType(kind="axisymmetric")
        assert gt.kind == "axisymmetric"


class TestSimulationConfig:
    def test_creation(self):
        gas = GasProperties()
        inlet = InletCondition(mach=2.0)
        walls = (
            WallDefinition(
                points=(WallPoint(0, 1), WallPoint(1, 1)), name="upper"
            ),
        )
        cfg = SimulationConfig(
            gas=gas,
            inlet=inlet,
            walls=walls,
            geometry_type=GeometryType(kind="planar"),
        )
        assert cfg.n_char_lines == 50
        assert cfg.gas.gamma == 1.4

    def test_mutable(self):
        gas = GasProperties()
        inlet = InletCondition(mach=2.0)
        walls = (
            WallDefinition(
                points=(WallPoint(0, 1), WallPoint(1, 1)), name="upper"
            ),
        )
        cfg = SimulationConfig(
            gas=gas,
            inlet=inlet,
            walls=walls,
            geometry_type=GeometryType(kind="planar"),
        )
        cfg.n_char_lines = 100
        assert cfg.n_char_lines == 100
