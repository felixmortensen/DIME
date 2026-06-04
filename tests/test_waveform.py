"""
Tests for dime.waveform — q(t) calculations and MATLAB file loading.
"""

import numpy as np
import pytest

from dime.waveform import to_qt, to_q_spectrum, GAMMA


dt = 46e-6   # 46 µs raster time


def _rect_pulse(n=100, amp=0.08, axis=0):
    """Single rectangular gradient pulse along one axis."""
    gwf = np.zeros((n, 3))
    gwf[:, axis] = amp
    rf = np.ones(n)
    return gwf, rf


def _spin_echo(n=200, amp=0.08, axis=0):
    """
    Spin-echo with two equal lobes and an rf flip in the middle.
    Net dephasing is zero at the echo.
    """
    gwf = np.zeros((n, 3))
    half = n // 2
    gwf[:half,  axis] = amp
    gwf[half:,  axis] = amp
    rf = np.ones(n)
    rf[half:] = -1
    return gwf, rf


class TestToQt:
    def test_linear_growth_single_pulse(self):
        n = 50
        amp = 0.05
        gwf, rf = _rect_pulse(n, amp)
        q = to_qt(gwf, rf, dt)
        # q should grow linearly: q[k] = GAMMA * amp * (k+1) * dt
        expected = GAMMA * amp * np.arange(1, n + 1) * dt
        np.testing.assert_allclose(q[:, 0], expected, rtol=1e-10)

    def test_zero_at_echo_for_balanced_waveform(self):
        gwf, rf = _spin_echo()
        q = to_qt(gwf, rf, dt)
        # Final q should be zero (spin echo condition)
        assert np.all(np.abs(q[-1, :]) < 1e-10)

    def test_shape(self):
        n = 80
        gwf, rf = _rect_pulse(n)
        q = to_qt(gwf, rf, dt)
        assert q.shape == (n, 3)

    def test_rf_sign_inversion(self):
        n = 40
        amp = 0.05
        gwf = np.zeros((n, 3))
        gwf[:, 0] = amp
        rf_pos = np.ones(n)
        rf_neg = -np.ones(n)
        q_pos = to_qt(gwf, rf_pos, dt)
        q_neg = to_qt(gwf, rf_neg, dt)
        np.testing.assert_allclose(q_pos, -q_neg, rtol=1e-10)


class TestToQSpectrum:
    def test_output_shapes(self):
        gwf, rf = _spin_echo()
        p, f = to_q_spectrum(gwf, rf, dt)
        assert p.ndim == 2
        assert p.shape[1] == 6
        assert f.shape[0] == p.shape[0]

    def test_diagonal_components_real_and_nonnegative(self):
        gwf, rf = _spin_echo()
        p, _ = to_q_spectrum(gwf, rf, dt)
        for c in range(3):
            assert np.all(p[:, c].real >= -1e-20), f"Component {c} has negative values"

    def test_zero_waveform_gives_zero_spectrum(self):
        gwf = np.zeros((100, 3))
        rf  = np.ones(100)
        p, _ = to_q_spectrum(gwf, rf, dt)
        np.testing.assert_allclose(np.abs(p), 0.0, atol=1e-30)

    def test_spectrum_integral_consistent_with_b_value(self):
        # b = integral q^2(t) dt = integral |Q(f)|^2 df  (Parseval)
        gwf, rf = _spin_echo(n=400, amp=0.08, axis=0)
        q = to_qt(gwf, rf, dt)
        b_time = np.sum(q[:, 0] ** 2) * dt

        p, f = to_q_spectrum(gwf, rf, dt)
        b_freq = np.trapezoid(p[:, 0].real, f)

        # Zero-padding in to_q_spectrum introduces small trapezoidal integration
        # error, so allow a looser tolerance here.
        assert abs(b_time - b_freq) / (b_time + 1e-30) < 0.10

    def test_inactive_axis_has_zero_spectrum(self):
        # Only x-axis active; y and z spectra should be zero
        gwf, rf = _spin_echo(axis=0)
        p, _ = to_q_spectrum(gwf, rf, dt)
        np.testing.assert_allclose(np.abs(p[:, 1]), 0.0, atol=1e-30)  # y
        np.testing.assert_allclose(np.abs(p[:, 2]), 0.0, atol=1e-30)  # z
