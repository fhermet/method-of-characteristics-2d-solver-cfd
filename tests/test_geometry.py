"""Tests for moc2d.geometry — wall interpolation and intersection."""

import math
import pytest
from moc2d.config import WallPoint, WallDefinition
from moc2d.geometry import (
    wall_y_at,
    wall_angle_at,
    intersect_char_wall,
    find_compressive_corners,
)


class TestLinearWall:
    def setup_method(self):
        self.flat = WallDefinition(
            points=(WallPoint(0, 1), WallPoint(2, 1)), name="upper"
        )
        self.ramp = WallDefinition(
            points=(
                WallPoint(0, 0),
                WallPoint(1, 0),
                WallPoint(2, math.tan(math.radians(10))),
            ),
            name="lower",
        )

    def test_flat_y(self):
        assert wall_y_at(self.flat, 0.5) == pytest.approx(1.0)
        assert wall_y_at(self.flat, 1.5) == pytest.approx(1.0)

    def test_flat_angle(self):
        assert wall_angle_at(self.flat, 0.5) == pytest.approx(0.0)

    def test_ramp_y_before_corner(self):
        assert wall_y_at(self.ramp, 0.5) == pytest.approx(0.0)

    def test_ramp_y_after_corner(self):
        y_at_1_5 = wall_y_at(self.ramp, 1.5)
        expected = 0.5 * math.tan(math.radians(10))
        assert y_at_1_5 == pytest.approx(expected, abs=1e-6)

    def test_ramp_angle_after_corner(self):
        angle = wall_angle_at(self.ramp, 1.5)
        assert angle == pytest.approx(math.radians(10), abs=1e-6)


class TestCubicWall:
    def setup_method(self):
        self.nozzle = WallDefinition(
            points=(
                WallPoint(0, 1.0),
                WallPoint(0.5, 0.8),
                WallPoint(1.0, 0.9),
                WallPoint(2.0, 1.5),
            ),
            name="upper",
            interpolation="cubic",
        )

    def test_passes_through_points(self):
        assert wall_y_at(self.nozzle, 0.0) == pytest.approx(1.0, abs=1e-6)
        assert wall_y_at(self.nozzle, 0.5) == pytest.approx(0.8, abs=1e-6)
        assert wall_y_at(self.nozzle, 2.0) == pytest.approx(1.5, abs=1e-6)

    def test_smooth_angle(self):
        angle = wall_angle_at(self.nozzle, 1.0)
        assert angle != 0.0


class TestIntersection:
    def test_horizontal_wall(self):
        wall = WallDefinition(
            points=(WallPoint(0, 1), WallPoint(2, 1)), name="upper"
        )
        x, y = intersect_char_wall(
            x0=0.0, y0=0.5, angle=math.radians(45), wall=wall
        )
        assert x == pytest.approx(0.5, abs=1e-6)
        assert y == pytest.approx(1.0, abs=1e-6)


class TestCompressiveCorners:
    def test_ramp_has_one_corner(self):
        ramp = WallDefinition(
            points=(
                WallPoint(0, 0),
                WallPoint(1, 0),
                WallPoint(2, math.tan(math.radians(10))),
            ),
            name="lower",
        )
        corners = find_compressive_corners(ramp)
        assert len(corners) == 1
        assert corners[0][0] == pytest.approx(1.0)
        assert corners[0][1] == pytest.approx(0.0)
        assert corners[0][2] > 0

    def test_expansion_corner_not_compressive(self):
        wall = WallDefinition(
            points=(
                WallPoint(0, 1),
                WallPoint(1, 1),
                WallPoint(2, 1 + math.tan(math.radians(10))),
            ),
            name="upper",
        )
        corners = find_compressive_corners(wall)
        assert len(corners) == 0

    def test_flat_wall_no_corners(self):
        wall = WallDefinition(
            points=(WallPoint(0, 1), WallPoint(2, 1)), name="upper"
        )
        corners = find_compressive_corners(wall)
        assert len(corners) == 0
