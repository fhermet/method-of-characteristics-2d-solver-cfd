"""Tests for moc2d.gas — perfect gas relations."""

import math
import numpy as np
import pytest
from moc2d.gas import (
    mach_angle,
    prandtl_meyer,
    inverse_prandtl_meyer,
    pressure_ratio,
    temperature_ratio,
    density_ratio,
    sound_speed,
    stagnation_from_static,
    oblique_shock_beta,
    post_shock_state,
    max_deflection,
)


class TestMachAngle:
    def test_mach_1(self):
        assert mach_angle(1.0) == pytest.approx(math.pi / 2)

    def test_mach_2(self):
        assert mach_angle(2.0) == pytest.approx(math.radians(30))

    def test_vectorized(self):
        M = np.array([1.0, 2.0])
        mu = mach_angle(M)
        assert mu[0] == pytest.approx(math.pi / 2)
        assert mu[1] == pytest.approx(math.radians(30))


class TestPrandtlMeyer:
    def test_mach_1(self):
        assert prandtl_meyer(1.0, 1.4) == pytest.approx(0.0, abs=1e-10)

    def test_mach_2(self):
        expected_deg = 26.3798
        assert prandtl_meyer(2.0, 1.4) == pytest.approx(math.radians(expected_deg), abs=1e-4)

    def test_mach_3(self):
        expected_deg = 49.7573
        assert prandtl_meyer(3.0, 1.4) == pytest.approx(math.radians(expected_deg), abs=1e-4)

    def test_vectorized(self):
        M = np.array([1.0, 2.0])
        nu = prandtl_meyer(M, 1.4)
        assert nu[0] == pytest.approx(0.0, abs=1e-10)
        assert nu[1] == pytest.approx(math.radians(26.3798), abs=1e-4)


class TestInversePrandtlMeyer:
    def test_roundtrip_mach_2(self):
        nu = prandtl_meyer(2.0, 1.4)
        M = inverse_prandtl_meyer(nu, 1.4)
        assert M == pytest.approx(2.0, abs=1e-6)

    def test_roundtrip_mach_3(self):
        nu = prandtl_meyer(3.0, 1.4)
        M = inverse_prandtl_meyer(nu, 1.4)
        assert M == pytest.approx(3.0, abs=1e-6)

    def test_nu_zero(self):
        M = inverse_prandtl_meyer(0.0, 1.4)
        assert M == pytest.approx(1.0, abs=1e-6)


class TestIsentropicRatios:
    def test_pressure_ratio_mach_1(self):
        assert pressure_ratio(1.0, 1.4) == pytest.approx(0.5283, abs=1e-4)

    def test_pressure_ratio_mach_2(self):
        assert pressure_ratio(2.0, 1.4) == pytest.approx(0.1278, abs=1e-4)

    def test_temperature_ratio_mach_2(self):
        assert temperature_ratio(2.0, 1.4) == pytest.approx(0.5556, abs=1e-4)

    def test_density_ratio_mach_2(self):
        assert density_ratio(2.0, 1.4) == pytest.approx(0.2300, abs=1e-4)

    def test_ratios_at_mach_1_consistency(self):
        gamma = 1.4
        M = 2.0
        T_r = temperature_ratio(M, gamma)
        p_r = pressure_ratio(M, gamma)
        rho_r = density_ratio(M, gamma)
        assert p_r == pytest.approx(T_r ** (gamma / (gamma - 1)), abs=1e-6)
        assert rho_r == pytest.approx(T_r ** (1 / (gamma - 1)), abs=1e-6)

    def test_vectorized(self):
        M = np.array([1.0, 2.0])
        p = pressure_ratio(M, 1.4)
        assert p[0] == pytest.approx(0.5283, abs=1e-4)
        assert p[1] == pytest.approx(0.1278, abs=1e-4)


class TestUtilities:
    def test_sound_speed(self):
        a = sound_speed(300.0, 1.4, 287.0)
        assert a == pytest.approx(347.189, abs=0.1)

    def test_stagnation_from_static(self):
        p0, T0 = stagnation_from_static(2.0, 1.0, 1.0, 1.4)
        assert p0 == pytest.approx(1.0 / 0.1278, abs=0.01)
        assert T0 == pytest.approx(1.0 / 0.5556, abs=0.01)


class TestObliqueShock:
    def test_beta_mach2_theta10(self):
        theta = math.radians(10)
        beta = oblique_shock_beta(2.0, theta, 1.4)
        assert math.degrees(beta) == pytest.approx(39.31, abs=0.2)

    def test_beta_mach3_theta20(self):
        theta = math.radians(20)
        beta = oblique_shock_beta(3.0, theta, 1.4)
        assert math.degrees(beta) == pytest.approx(37.76, abs=0.3)

    def test_beta_strong(self):
        theta = math.radians(10)
        beta_weak = oblique_shock_beta(2.0, theta, 1.4, strong=False)
        beta_strong = oblique_shock_beta(2.0, theta, 1.4, strong=True)
        assert beta_strong > beta_weak

    def test_normal_shock_at_theta_zero(self):
        beta = oblique_shock_beta(2.0, 0.0, 1.4)
        assert beta == pytest.approx(mach_angle(2.0), abs=1e-3)


class TestPostShockState:
    def test_mach2_beta_39(self):
        beta = math.radians(39.31)
        M2, p_r, T_r, rho_r, theta = post_shock_state(2.0, beta, 1.4)
        assert M2 == pytest.approx(1.641, abs=0.01)
        assert p_r == pytest.approx(1.707, abs=0.02)
        assert theta == pytest.approx(math.radians(10), abs=0.02)

    def test_normal_shock_limit(self):
        M2, p_r, T_r, rho_r, theta = post_shock_state(2.0, math.radians(90), 1.4)
        assert M2 == pytest.approx(0.5774, abs=0.001)
        assert p_r == pytest.approx(4.5, abs=0.01)
        assert theta == pytest.approx(0.0, abs=1e-6)


class TestMaxDeflection:
    def test_mach_2(self):
        theta_max = max_deflection(2.0, 1.4)
        assert math.degrees(theta_max) == pytest.approx(22.97, abs=0.2)

    def test_mach_3(self):
        theta_max = max_deflection(3.0, 1.4)
        assert math.degrees(theta_max) == pytest.approx(34.07, abs=0.2)
