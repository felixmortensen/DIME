"""
Tests for dime.fitting — powder-averaged MD fitting.
"""

import numpy as np
import pytest

from dime.fitting import fit_md


bvals = np.array([0.0, 0.1, 0.5, 1.0, 2.0])   # ms/µm²


def _mono(b, D):
    """Pure monoexponential signal."""
    return np.exp(-b * D)


def _cumulant(b, D, V=0.0, S=0.0):
    """3-parameter cumulant signal (Eq. 18)."""
    return np.exp(-b*D + 0.5*b**2*V - (1/6)*b**3*S)


class TestFitMd:
    def test_recovers_D_from_monoexponential(self):
        D_true = 1.5e-3   # µm²/ms
        sig = _mono(bvals, D_true)
        D_ste, D_lte = fit_md(bvals, sig[np.newaxis, :], sig[np.newaxis, :])
        assert abs(D_ste[0] - D_true) / D_true < 0.01
        assert abs(D_lte[0] - D_true) / D_true < 0.01

    def test_ste_and_lte_can_differ(self):
        D_ste_true = 1.2e-3
        D_lte_true = 1.8e-3
        sig_ste = _mono(bvals, D_ste_true)
        sig_lte = _mono(bvals, D_lte_true)
        D_ste, D_lte = fit_md(bvals, sig_ste[np.newaxis, :], sig_lte[np.newaxis, :])
        assert abs(D_ste[0] - D_ste_true) / D_ste_true < 0.01
        assert abs(D_lte[0] - D_lte_true) / D_lte_true < 0.01

    def test_multiple_radii(self):
        D_vals = np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])
        sig = np.stack([_mono(bvals, D) for D in D_vals])   # (4, 5)
        D_ste, D_lte = fit_md(bvals, sig, sig)
        assert D_ste.shape == (4,)
        for i, D_true in enumerate(D_vals):
            assert abs(D_ste[i] - D_true) / D_true < 0.01

    def test_cumulant_signal_D_recovered(self):
        # Even with nonzero higher-order terms, D should still be recovered
        D_true = 1.5e-3
        sig = _cumulant(bvals, D_true, V=0.01e-6, S=0.001e-9)
        D_ste, _ = fit_md(bvals, sig[np.newaxis, :], sig[np.newaxis, :])
        assert abs(D_ste[0] - D_true) / D_true < 0.05

    def test_output_shapes(self):
        n_radii = 7
        sig = np.stack([_mono(bvals, 1e-3)] * n_radii)
        D_ste, D_lte = fit_md(bvals, sig, sig)
        assert D_ste.shape == (n_radii,)
        assert D_lte.shape == (n_radii,)

    def test_D_positive(self):
        sig = _mono(bvals, 1.0e-3)
        D_ste, D_lte = fit_md(bvals, sig[np.newaxis, :], sig[np.newaxis, :])
        assert D_ste[0] > 0
        assert D_lte[0] > 0
