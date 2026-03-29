"""Tests for moc2d.test_cases — predefined simulation cases."""

import pytest
from moc2d.test_cases import (
    simple_ramp,
    expansion_corner,
    double_ramp,
    planar_nozzle,
    axisymmetric_nozzle,
    CASE_REGISTRY,
)
from moc2d.config import SimulationConfig


class TestCaseRegistry:
    def test_five_cases_registered(self):
        assert len(CASE_REGISTRY) == 5

    def test_all_return_sim_config(self):
        for name, factory in CASE_REGISTRY.items():
            config = factory()
            assert isinstance(config, SimulationConfig), f"{name} did not return SimulationConfig"
            assert config.inlet.mach > 1.0, f"{name} inlet is not supersonic"


class TestSimpleRamp:
    def test_default(self):
        cfg = simple_ramp()
        assert cfg.inlet.mach == 2.0
        assert cfg.geometry_type.kind == "planar"
        assert len(cfg.walls) == 2

    def test_custom_angle(self):
        cfg = simple_ramp(angle_deg=15, M_inf=3.0)
        assert cfg.inlet.mach == 3.0


class TestExpansionCorner:
    def test_default(self):
        cfg = expansion_corner()
        assert cfg.inlet.mach == 2.0


class TestDoubleRamp:
    def test_default(self):
        cfg = double_ramp()
        assert len(cfg.walls) == 2


class TestPlanarNozzle:
    def test_default(self):
        cfg = planar_nozzle()
        assert cfg.geometry_type.kind == "planar"

    def test_cubic_walls(self):
        cfg = planar_nozzle()
        upper = [w for w in cfg.walls if w.name == "upper"][0]
        assert upper.interpolation == "cubic"


class TestAxisymmetricNozzle:
    def test_default(self):
        cfg = axisymmetric_nozzle()
        assert cfg.geometry_type.kind == "axisymmetric"
