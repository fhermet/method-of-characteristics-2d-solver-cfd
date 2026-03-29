"""Integration tests — run full solver on each predefined case."""

import math
import pytest
from moc2d.test_cases import CASE_REGISTRY
from moc2d.solver import solve
from moc2d.results import interpolate_to_grid
from moc2d.gas import oblique_shock_beta, post_shock_state, prandtl_meyer, inverse_prandtl_meyer


class TestSimpleRampAnalytical:
    """Validate simple ramp against analytical oblique shock solution."""

    def test_post_shock_mach(self):
        from moc2d.test_cases import simple_ramp
        cfg = simple_ramp(angle_deg=10, M_inf=2.0, n_char_lines=20)
        result = solve(cfg)

        beta = oblique_shock_beta(2.0, math.radians(10), 1.4)
        M2_exact, _, _, _, _ = post_shock_state(2.0, beta, 1.4)

        downstream = [
            p for p in result.char_points
            if p.x > 2.0 and 0.3 < p.y < 1.5
        ]
        if downstream:
            avg_mach = sum(p.mach for p in downstream) / len(downstream)
            assert avg_mach == pytest.approx(M2_exact, abs=0.25)


class TestExpansionCornerAnalytical:
    """Validate expansion corner against Prandtl-Meyer solution."""

    def test_post_expansion_mach(self):
        from moc2d.test_cases import expansion_corner
        cfg = expansion_corner(angle_deg=10, M_inf=2.0, n_char_lines=20)
        result = solve(cfg)

        nu1 = prandtl_meyer(2.0, 1.4)
        nu2 = nu1 + math.radians(10)
        M2_exact = inverse_prandtl_meyer(nu2, 1.4)

        downstream = [p for p in result.char_points if p.x > 2.0]
        if downstream:
            avg_mach = sum(p.mach for p in downstream) / len(downstream)
            assert avg_mach == pytest.approx(M2_exact, abs=0.3)


class TestAllCasesRun:
    """All predefined cases must run without error and produce points."""

    @pytest.mark.parametrize("case_name", list(CASE_REGISTRY.keys()))
    def test_case_runs(self, case_name):
        factory = CASE_REGISTRY[case_name]
        cfg = factory()
        cfg.n_char_lines = 10
        result = solve(cfg)
        assert len(result.char_points) > 0

    @pytest.mark.parametrize("case_name", list(CASE_REGISTRY.keys()))
    def test_grid_interpolation(self, case_name):
        factory = CASE_REGISTRY[case_name]
        cfg = factory()
        cfg.n_char_lines = 10
        result = solve(cfg)
        grid = interpolate_to_grid(result, nx=10, ny=5)
        assert grid["mach"].shape == (5, 10)
