"""
Signal fitting: powder-averaged mean diffusivity (MD) from DW signals.
"""

from pathlib import Path

import numpy as np
import scipy.optimize as opt


def fit_md(
    bvals: np.ndarray,
    sig_ste: np.ndarray,
    sig_lte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit powder-averaged MD from STE and LTE signal curves (Eq. 18).

    Uses the 4-parameter cumulant model:
        S(b) = S0 * exp(-b*D + 0.5*b^2*C2 - (1/6)*b^3*C3)

    Parameters
    ----------
    bvals   : b-values [ms/um^2], shape [n_b]
    sig_ste : powder-averaged STE signals, shape [n_radii x n_b]
    sig_lte : powder-averaged LTE signals, shape [n_radii x n_b]

    Returns
    -------
    D_ste : apparent MD from STE [um^2/ms], shape [n_radii]
    D_lte : apparent MD from LTE [um^2/ms], shape [n_radii]
    """
    bvals   = np.asarray(bvals, dtype=float)
    sig_ste = np.asarray(sig_ste, dtype=float)
    sig_lte = np.asarray(sig_lte, dtype=float)

    n_radii = sig_ste.shape[0]
    D_ste   = np.zeros(n_radii)
    D_lte   = np.zeros(n_radii)

    for j in range(n_radii):
        D_ste[j] = _fit_one(bvals, sig_ste[j])
        D_lte[j] = _fit_one(bvals, sig_lte[j])

    return D_ste, D_lte


def fit_md_from_npz(
    npz_path: str | Path,
    out_path: str | Path | None = None,
) -> dict:
    """
    Load a simulation .npz file, fit MD, and optionally save updated results.

    Parameters
    ----------
    npz_path : path to .npz file from dime.simulate.simulate()
    out_path : if given, save updated data here; otherwise overwrites npz_path

    Returns
    -------
    Dictionary with all original fields plus 'D_STE' and 'D_LTE'.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}

    bvals   = data["bvals"].astype(float)
    pow_ste = data["STE_powder"].astype(float)
    pow_lte = data["LTE_powder"].astype(float)

    data["D_STE"], data["D_LTE"] = fit_md(bvals, pow_ste, pow_lte)

    save_path = Path(out_path) if out_path is not None else npz_path.with_name(npz_path.stem)
    np.savez(save_path, **data)
    return data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dwsig(b, s0, D, c2, c3):
    return s0 * np.exp(-b * D + 0.5 * b ** 2 * c2 - (1 / 6) * b ** 3 * c3)


def _fit_one(bvals: np.ndarray, sig: np.ndarray) -> float:
    popt, _ = opt.curve_fit(_dwsig, bvals, sig, p0=[1.0, 1e-3, 0.0, 0.0])
    return float(popt[1])
