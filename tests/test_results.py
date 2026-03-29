"""Tests for moc2d.results — grid interpolation and field extraction."""

import pytest
import numpy as np
from moc2d.config import (
    GasProperties, InletCondition, WallDefinition, WallPoint,
    GeometryType, SimulationConfig,
)
from moc2d.solver import solve
from moc2d.results import interpolate_to_grid, extract_wall_data


class TestInterpolateToGrid:
    def test_uniform_flow_mach_field(self):
        config = SimulationConfig(
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
        result = solve(config)
        grid = interpolate_to_grid(result, nx=20, ny=10)
        assert "mach" in grid
        assert "x" in grid
        assert "y" in grid
        assert grid["x"].shape == (20,)
        assert grid["y"].shape == (10,)
        assert grid["mach"].shape == (10, 20)
        valid = ~np.isnan(grid["mach"])
        if valid.any():
            assert np.allclose(grid["mach"][valid], 2.0, atol=0.2)


class TestExtractWallData:
    def test_returns_arrays(self):
        config = SimulationConfig(
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
        result = solve(config)
        wall_data = extract_wall_data(result)
        assert "x" in wall_data
        assert "mach" in wall_data
        assert "p" in wall_data
