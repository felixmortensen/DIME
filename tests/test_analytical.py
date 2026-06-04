"""
Tests for dime.analytical — Bessel roots, diffusion spectra, GPA signals.

All tests are self-contained (no data files required).
"""

import numpy as np
import pytest

from dime.analytical import (
    cylinder_bessel_kernels,
    sphere_bessel_kernels,
    Dw_restricted,
    signal_gpa_cylinder,
    signal_gpa_sphere,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

R  = 10e-6   # 10 µm radius
D0 = 2e-9    # 2 µm²/ms free diffusivity
dt = 46e-6   # 46 µs raster time


def _simple_ste_waveform(n=500, gmax=0.08):
    """
    Minimal 3-axis STE-like waveform for signal tests.
    x and y: bipolar pulses confined to the first half (balanced: q returns to 0).
    z: monopolar pulse spanning the middle (balanced across the rf flip).
    rf: +1 before midpoint, -1 after.
    """
    gwf = np.zeros((n, 3))
    q = n // 4
    gwf[:q,   0] =  gmax
    gwf[q:2*q, 0] = -gmax
    gwf[:q,   1] =  gmax
    gwf[q:2*q, 1] = -gmax
    gwf[q:3*q, 2] = gmax
    rf = np.ones(n)
    rf[n // 2:] = -1
    return gwf, rf, dt


# ---------------------------------------------------------------------------
# Bessel roots
# ---------------------------------------------------------------------------

class TestCylinderBesselKernels:
    def test_first_root_known_value(self):
        mu = cylinder_bessel_kernels(4)
        # Roots of J'_1(mu) = 0: 1.8412, 5.3314, 8.5363, 11.706
        assert abs(mu[0] - 1.8412) < 1e-3

    def test_second_root_known_value(self):
        mu = cylinder_bessel_kernels(4)
        assert abs(mu[1] - 5.3314) < 1e-3

    def test_roots_positive_and_increasing(self):
        mu = cylinder_bessel_kernels(10)
        assert np.all(mu > 0)
        assert np.all(np.diff(mu) > 0)

    def test_boundary_condition_satisfied(self):
        from scipy.special import jv
        mu = cylinder_bessel_kernels(5)
        residuals = jv(0, mu) - jv(1, mu) / mu
        assert np.all(np.abs(residuals) < 1e-10)


class TestSphereBesselKernels:
    def test_first_root_known_value(self):
        zk = sphere_bessel_kernels(4)
        # Roots of (z²-2)sin(z)+2z·cos(z)=0: 2.0816, 5.9404, 9.2058, 12.4044
        assert abs(zk[0] - 2.0816) < 1e-3

    def test_second_root_known_value(self):
        zk = sphere_bessel_kernels(4)
        assert abs(zk[1] - 5.9404) < 1e-3

    def test_roots_positive_and_increasing(self):
        zk = sphere_bessel_kernels(10)
        assert np.all(zk > 0)
        assert np.all(np.diff(zk) > 0)

    def test_boundary_condition_satisfied(self):
        zk = sphere_bessel_kernels(5)
        residuals = (zk**2 - 2) * np.sin(zk) + 2 * zk * np.cos(zk)
        assert np.all(np.abs(residuals) < 1e-10)


# ---------------------------------------------------------------------------
# Dw_restricted
# ---------------------------------------------------------------------------

class TestDwRestricted:
    @pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
    def test_zero_frequency_gives_zero(self, geometry):
        D = Dw_restricted(np.array([0.0]), R, D0, geometry)
        assert abs(D[0]) < 1e-20

    @pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
    def test_high_frequency_approaches_D0(self, geometry):
        # At very high omega, all Lorentzians saturate and D -> D0
        omega_high = np.array([1e8])
        D = Dw_restricted(omega_high, R, D0, geometry)
        assert abs(D[0] - D0) / D0 < 0.01

    @pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
    def test_spectrum_monotonically_increasing(self, geometry):
        omega = np.linspace(0, 1e5, 200)
        D = Dw_restricted(omega, R, D0, geometry)
        assert np.all(np.diff(D) >= -1e-25), f"{geometry} spectrum not monotone"

    @pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
    def test_bounded_by_D0(self, geometry):
        omega = np.linspace(0, 1e6, 100)
        D = Dw_restricted(omega, R, D0, geometry)
        assert np.all(D >= -1e-25)
        assert np.all(D <= D0 * 1.001)

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="geometry"):
            Dw_restricted(np.array([1.0]), R, D0, "cube")

    def test_larger_radius_higher_diffusivity_at_fixed_omega(self):
        # Larger R means less restriction at a given frequency
        omega = np.array([1e3])
        D_small = Dw_restricted(omega, 1e-6,  D0, "cylinder")
        D_large = Dw_restricted(omega, 50e-6, D0, "cylinder")
        assert D_large > D_small

    def test_alpha_sets_low_frequency_floor(self):
        alpha = 0.3
        D = Dw_restricted(np.array([0.0]), R, D0, "cylinder", alpha=alpha)
        assert abs(D[0] - D0 * alpha) / (D0 * alpha) < 1e-10


# ---------------------------------------------------------------------------
# GPA signals
# ---------------------------------------------------------------------------

class TestSignalGpaCylinder:
    def test_zero_gradient_gives_unit_signal(self):
        gwf, rf, _ = _simple_ste_waveform(gmax=0.0)
        sig, beta = signal_gpa_cylinder(gwf, rf, dt, R, D0)
        assert abs(sig - 1.0) < 1e-10
        assert abs(beta) < 1e-10

    def test_signal_between_zero_and_one(self):
        gwf, rf, _ = _simple_ste_waveform()
        sig, _ = signal_gpa_cylinder(gwf, rf, dt, R, D0)
        assert 0.0 < sig <= 1.0

    def test_higher_gmax_lower_signal(self):
        gwf_lo, rf, _ = _simple_ste_waveform(gmax=0.02)
        gwf_hi, rf, _ = _simple_ste_waveform(gmax=0.08)
        sig_lo, _ = signal_gpa_cylinder(gwf_lo, rf, dt, R, D0)
        sig_hi, _ = signal_gpa_cylinder(gwf_hi, rf, dt, R, D0)
        assert sig_hi < sig_lo

    def test_large_radius_approaches_free_diffusion(self):
        # At R=1 mm, restriction is negligible; signal ~ exp(-b*D0)
        from dime.waveform import to_qt
        gwf, rf, _ = _simple_ste_waveform()
        q = to_qt(gwf, rf, dt)
        b = np.sum(np.sum(q**2, axis=1)) * dt
        sig_restricted, _ = signal_gpa_cylinder(gwf, rf, dt, R=1e-3, D0=D0)
        sig_free = np.exp(-b * D0)
        assert abs(sig_restricted - sig_free) / max(sig_free, 1e-10) < 0.01


class TestSignalGpaSphere:
    def test_zero_gradient_gives_unit_signal(self):
        gwf, rf, _ = _simple_ste_waveform(gmax=0.0)
        sig, beta = signal_gpa_sphere(gwf, rf, dt, R, D0)
        assert abs(sig - 1.0) < 1e-10

    def test_signal_between_zero_and_one(self):
        gwf, rf, _ = _simple_ste_waveform()
        sig, _ = signal_gpa_sphere(gwf, rf, dt, R, D0)
        assert 0.0 < sig <= 1.0

    def test_cylinder_signal_lower_than_sphere_at_same_R(self):
        # At small R (strong restriction): cylinder z-axis diffuses freely at D0
        # (causing large z attenuation), while the sphere restricts all axes
        # (D_sph << D0). Net effect: sig_cyl < sig_sph at small R.
        gwf, rf, _ = _simple_ste_waveform()
        sig_cyl, _ = signal_gpa_cylinder(gwf, rf, dt, R, D0)
        sig_sph, _ = signal_gpa_sphere(gwf, rf, dt, R, D0)
        assert sig_cyl < sig_sph
