"""
Analytical diffusion spectra and GPA signals for cylinders and spheres (Appendix C).

Both geometries use the same Lorentzian sum (Eq. C1):

    D_res^(d)(omega) = D0 * sum_k [ a_k * B_k * omega^2 / (a_k^2 * D0^2 + omega^2) ]

with geometry-specific roots z_k and coefficients a_k, B_k (Eqs. C2–C9).

For an infinitely long cylinder aligned with z:
  - Restricted transverse spectrum D_perp(omega)  (Eqs. C4–C9)
  - Full tensor: diag(D_perp, D_perp, D0)

For a sphere:
  - Isotropic restricted spectrum D_sph(omega)  (Eq. C10, d=3)
  - Full tensor: D_sph * I

Reference: Appendix C; Nilsson et al. (2017) NMR Biomed; Lundell & Lasic (2020).
"""

import numpy as np
from scipy.special import jv
from scipy.optimize import fsolve, brentq

from .waveform import to_q_spectrum, GAMMA


# ---------------------------------------------------------------------------
# Bessel / root-finding helpers
# ---------------------------------------------------------------------------

def cylinder_bessel_kernels(n: int) -> np.ndarray:
    """
    Roots of J'_1(mu) = 0, i.e. J0(x) = J1(x)/x  (Eq. C5).

    These are the transverse eigenvalues for a cylinder with no-flux BC.

    Parameters
    ----------
    n : number of roots to return

    Returns
    -------
    roots : [n] array, all positive, increasing
    """
    roots = np.zeros(n, dtype=float)
    for k in range(n):
        x0 = 2.0 + k * np.pi
        roots[k] = fsolve(lambda x: jv(0, x) - jv(1, x) / x, x0)[0]
    return roots


def sphere_bessel_kernels(n: int) -> np.ndarray:
    """
    Roots of  (z^2 - 2)*sin(z) + 2*z*cos(z) = 0  (Eq. C3 with d=3).

    Equivalently: tan(z) = 2z / (2 - z^2).
    These are the eigenvalues for a sphere with no-flux BC.
    There is exactly one root per interval (k*pi, (k+1)*pi) for k = 0, 1, 2, ...

    Parameters
    ----------
    n : number of roots to return

    Returns
    -------
    roots : [n] array, all positive, increasing
            First few values: 2.082, 5.940, 9.206, 12.405, ...
    """
    def f(z):
        return (z ** 2 - 2.0) * np.sin(z) + 2.0 * z * np.cos(z)

    roots = np.zeros(n, dtype=float)
    for k in range(n):
        a = k * np.pi + 1e-10
        b = (k + 1) * np.pi - 1e-10
        roots[k] = brentq(f, a, b, xtol=1e-14)
    return roots


# ---------------------------------------------------------------------------
# Diffusion spectra
# ---------------------------------------------------------------------------

def Dw_cylinder(
    omega: float | np.ndarray,
    R: float,
    D0: float,
    alpha: float = 0.0,
    n: int = 100,
) -> np.ndarray:
    """
    Transverse frequency-dependent diffusivity for a cylinder (Eqs. C4–C9).

    D_perp(omega) = D0*alpha + D0*(1-alpha) * sum_k [ a_k*B_k * omega^2 / (a_k^2*D0^2 + omega^2) ]

    For a fully closed cylinder (no exchange) use alpha=0 (default), which
    gives D_perp(0)=0 and D_perp(inf)=D0.  The full 3-D cylinder tensor is
    diag(D_perp, D_perp, D0) — the z-axis contribution is handled separately
    in signal_gpa_cylinder.

    Parameters
    ----------
    omega : angular frequency [rad/s]
    R     : cylinder radius [m]
    D0    : free diffusivity [m^2/s]
    alpha : long-time fraction D_inf/D0 (default 0 for closed compartment)
    n     : number of Lorentzian terms

    Returns
    -------
    D : [len(omega)] diffusivity [m^2/s]
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    mu = cylinder_bessel_kernels(n)

    # Eq. C4: a_k = (mu_k/R)^2, B_k = 2*(R/mu_k)^2 / (mu_k^2 - 1)
    a = (mu / R) ** 2
    B = 2.0 * (R / mu) ** 2 / (mu ** 2 - 1.0)

    D = np.array([
        D0 * alpha + D0 * (1.0 - alpha) * np.sum(a * B * w ** 2 / (a ** 2 * D0 ** 2 + w ** 2))
        for w in omega
    ])
    return D


def Dw_sphere(
    omega: float | np.ndarray,
    R: float,
    D0: float,
    n: int = 100,
) -> np.ndarray:
    """
    Isotropic frequency-dependent diffusivity for a sphere (Eq. C1 with d=3).

    D_sph(omega) = D0 * sum_k [ a_k*B_k * omega^2 / (a_k^2*D0^2 + omega^2) ]

    with roots z_k from sphere_bessel_kernels and B_k = 2*(R/z_k)^2 / (z_k^2 - 2).

    D_sph(0) = 0, D_sph(inf) = D0.  The full tensor is D_sph * I  (Eq. C10).

    Parameters
    ----------
    omega : angular frequency [rad/s]
    R     : sphere radius [m]
    D0    : free diffusivity [m^2/s]
    n     : number of Lorentzian terms

    Returns
    -------
    D : [len(omega)] diffusivity [m^2/s]
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    zk = sphere_bessel_kernels(n)

    # Eq. C2 with d=3: a_k = (z_k/R)^2, B_k = 2*(R/z_k)^2 / (z_k^2 - 2)
    a = (zk / R) ** 2
    B = 2.0 * (R / zk) ** 2 / (zk ** 2 - 2.0)

    D = np.array([
        D0 * np.sum(a * B * w ** 2 / (a ** 2 * D0 ** 2 + w ** 2))
        for w in omega
    ])
    return D


# ---------------------------------------------------------------------------
# GPA signal computation — cylinders
# ---------------------------------------------------------------------------

def signal_gpa_cylinder(
    gwf: np.ndarray,
    rf: np.ndarray,
    dt: float,
    R: float,
    D0: float,
    n: int = 100,
    n_pad: int | None = None,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    """
    Analytical GPA signal for a cylinder aligned with the z-axis (Eq. 1).

    beta = integral[ (p_xx + p_yy)*D_perp(f) + p_zz*D0 ] df
    signal = exp(-beta)

    Parameters
    ----------
    gwf   : [n_t x 3] gradient waveform [T/m]
    rf    : [n_t] refocusing sign vector (+1/-1)
    dt    : raster time [s]
    R     : cylinder radius [m]
    D0    : free diffusivity [m^2/s]
    n     : number of Lorentzian terms
    n_pad : zero-padding length for FFT
    gamma : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    signal : float, signal magnitude
    beta   : float, log-attenuation
    """
    p, f = to_q_spectrum(gwf, rf, dt, n_pad=n_pad, gamma=gamma)
    omega = 2.0 * np.pi * f
    D_perp = Dw_cylinder(omega, R, D0, alpha=0.0, n=n)

    beta = np.trapz((p[:, 0].real + p[:, 1].real) * D_perp + p[:, 2].real * D0, x=f)
    return float(np.exp(-beta)), float(beta)


def signal_gpa_cylinder_rotations(
    gwfl: np.ndarray,
    rf: np.ndarray,
    dt: float,
    radii: np.ndarray,
    D0: float,
    n: int = 100,
    n_pad: int | None = None,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical GPA signal for a cylinder across multiple radii and rotations.

    The cylinder is assumed to be aligned with z before each waveform rotation,
    so q-spectra are precomputed once per rotation.

    Parameters
    ----------
    gwfl   : [n_t x 3 x n_rot] rotated gradient waveforms [T/m]
    rf     : [n_t] refocusing sign vector
    dt     : raster time [s]
    radii  : [n_radii] cylinder radii [m]
    D0     : free diffusivity [m^2/s]
    n      : number of Lorentzian terms
    n_pad  : zero-padding length for FFT
    gamma  : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    signals : [n_radii x n_rot] signal magnitudes
    betas   : [n_radii x n_rot] log-attenuations
    """
    gwfl   = np.asarray(gwfl, dtype=float)
    radii  = np.atleast_1d(np.asarray(radii, dtype=float))
    n_rot  = gwfl.shape[2]

    # Precompute q-spectra once per rotation
    P_xx, P_yy, P_zz, f_ref = _precompute_spectra(gwfl, rf, dt, n_rot, n_pad, gamma)
    omega = 2.0 * np.pi * f_ref

    signals = np.zeros((len(radii), n_rot), dtype=float)
    betas   = np.zeros((len(radii), n_rot), dtype=float)

    for i, R in enumerate(radii):
        D_perp = Dw_cylinder(omega, R, D0, alpha=0.0, n=n)
        beta = np.trapz(
            (P_xx + P_yy) * D_perp[None, :] + P_zz * D0,
            x=f_ref, axis=1,
        )
        betas[i]   = beta
        signals[i] = np.exp(-beta)

    return signals, betas


# ---------------------------------------------------------------------------
# GPA signal computation — spheres
# ---------------------------------------------------------------------------

def signal_gpa_sphere(
    gwf: np.ndarray,
    rf: np.ndarray,
    dt: float,
    R: float,
    D0: float,
    n: int = 100,
    n_pad: int | None = None,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    """
    Analytical GPA signal for a sphere (Eq. 1 with isotropic D_sph).

    beta = integral[ (p_xx + p_yy + p_zz) * D_sph(f) ] df
    signal = exp(-beta)

    Because D_sph is isotropic, the signal is rotation-invariant: all
    orientations of a given waveform yield the same signal in a sphere.
    STE and LTE can still differ because they have different encoding spectra.

    Parameters
    ----------
    gwf   : [n_t x 3] gradient waveform [T/m]
    rf    : [n_t] refocusing sign vector (+1/-1)
    dt    : raster time [s]
    R     : sphere radius [m]
    D0    : free diffusivity [m^2/s]
    n     : number of Lorentzian terms
    n_pad : zero-padding length for FFT
    gamma : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    signal : float, signal magnitude
    beta   : float, log-attenuation
    """
    p, f = to_q_spectrum(gwf, rf, dt, n_pad=n_pad, gamma=gamma)
    omega = 2.0 * np.pi * f
    D_sph = Dw_sphere(omega, R, D0, n=n)

    p_tot = p[:, 0].real + p[:, 1].real + p[:, 2].real
    beta  = np.trapz(p_tot * D_sph, x=f)
    return float(np.exp(-beta)), float(beta)


def signal_gpa_sphere_rotations(
    gwfl: np.ndarray,
    rf: np.ndarray,
    dt: float,
    radii: np.ndarray,
    D0: float,
    n: int = 100,
    n_pad: int | None = None,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical GPA signal for a sphere across multiple radii and rotations.

    Because the sphere spectrum is isotropic, signals are identical across
    rotations. This function is provided for API consistency with the cylinder
    version and to allow direct comparison with Monte Carlo results.

    Parameters
    ----------
    gwfl   : [n_t x 3 x n_rot] rotated gradient waveforms [T/m]
    rf     : [n_t] refocusing sign vector
    dt     : raster time [s]
    radii  : [n_radii] sphere radii [m]
    D0     : free diffusivity [m^2/s]
    n      : number of Lorentzian terms
    n_pad  : zero-padding length for FFT
    gamma  : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    signals : [n_radii x n_rot] signal magnitudes
    betas   : [n_radii x n_rot] log-attenuations
    """
    gwfl   = np.asarray(gwfl, dtype=float)
    radii  = np.atleast_1d(np.asarray(radii, dtype=float))
    n_rot  = gwfl.shape[2]

    P_xx, P_yy, P_zz, f_ref = _precompute_spectra(gwfl, rf, dt, n_rot, n_pad, gamma)
    omega = 2.0 * np.pi * f_ref

    signals = np.zeros((len(radii), n_rot), dtype=float)
    betas   = np.zeros((len(radii), n_rot), dtype=float)

    for i, R in enumerate(radii):
        D_sph = Dw_sphere(omega, R, D0, n=n)
        beta = np.trapz(
            (P_xx + P_yy + P_zz) * D_sph[None, :],
            x=f_ref, axis=1,
        )
        betas[i]   = beta
        signals[i] = np.exp(-beta)

    return signals, betas


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def alpha_from_bt(bt: np.ndarray) -> np.ndarray:
    """
    Long-time diffusivity fraction alpha = b_z / (b_x + b_y + b_z).

    Parameters
    ----------
    bt : [N x 3] b-tensor diagonal elements

    Returns
    -------
    alpha : [N] array
    """
    bt = np.asarray(bt, dtype=float)
    return bt[:, 2] / bt.sum(axis=1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _precompute_spectra(gwfl, rf, dt, n_rot, n_pad, gamma):
    """Precompute q-spectrum components for all rotations."""
    P_xx, P_yy, P_zz = [], [], []
    f_ref = None

    for k in range(n_rot):
        p, f = to_q_spectrum(gwfl[:, :, k], rf, dt, n_pad=n_pad, gamma=gamma)
        if f_ref is None:
            f_ref = f
        elif not np.allclose(f_ref, f):
            raise ValueError("Frequency grids differ across rotations.")
        P_xx.append(p[:, 0].real)
        P_yy.append(p[:, 1].real)
        P_zz.append(p[:, 2].real)

    return (
        np.asarray(P_xx, dtype=float),
        np.asarray(P_yy, dtype=float),
        np.asarray(P_zz, dtype=float),
        f_ref,
    )
