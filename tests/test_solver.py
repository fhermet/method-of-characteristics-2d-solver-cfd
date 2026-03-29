"""Tests for moc2d.solver — x-marching orchestration."""

import math
import pytest
from moc2d.config import (
    GasProperties, InletCondition, WallDefinition, WallPoint,
    GeometryType, SimulationConfig,
)
from moc2d.solver import solve, SolverResult


class TestUniformFlow:
    """Uniform M=2 flow between two flat walls — trivial case."""

    def setup_method(self):
        self.config = SimulationConfig(
            gas=GasProperties(gamma=1.4),
            inlet=InletCondition(mach=2.0),
            walls=(
                WallDefinition(
                    points=(WallPoint(0, 1), WallPoint(3, 1)), name="upper"
                ),
                WallDefinition(
                    points=(WallPoint(0, 0), WallPoint(3, 0)), name="lower"
                ),
            ),
            geometry_type=GeometryType(kind="planar"),
            n_char_lines=10,
        )

    def test_returns_solver_result(self):
        result = solve(self.config)
        assert isinstance(result, SolverResult)
        assert len(result.char_points) > 0
        assert result.wall_time >= 0

    def test_uniform_mach(self):
        result = solve(self.config)
        for pt in result.char_points:
            assert pt.mach == pytest.approx(2.0, abs=0.05)

    def test_uniform_theta(self):
        result = solve(self.config)
        for pt in result.char_points:
            assert pt.theta == pytest.approx(0.0, abs=0.05)

    def test_has_characteristic_lines(self):
        result = solve(self.config)
        assert len(result.c_plus_lines) > 0
        assert len(result.c_minus_lines) > 0


class TestSimpleRamp:
    """10-degree ramp at M=2 should produce a shock."""

    def setup_method(self):
        angle = math.radians(10)
        self.config = SimulationConfig(
            gas=GasProperties(gamma=1.4),
            inlet=InletCondition(mach=2.0),
            walls=(
                WallDefinition(
                    points=(WallPoint(0, 2), WallPoint(5, 2)), name="upper"
                ),
                WallDefinition(
                    points=(
                        WallPoint(0, 0),
                        WallPoint(1, 0),
                        WallPoint(5, 4 * math.tan(angle)),
                    ),
                    name="lower",
                ),
            ),
            geometry_type=GeometryType(kind="planar"),
            n_char_lines=15,
        )

    def test_shock_detected(self):
        result = solve(self.config)
        assert len(result.shock_points) > 0

    def test_shock_lines(self):
        result = solve(self.config)
        assert len(result.shock_lines) > 0
