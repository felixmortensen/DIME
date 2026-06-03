"""
Analytical diffusion spectrum models (Appendix C).

Provides Lorentzian-type diffusion spectra for cylinders using the
General Phase Approximation (GPA).
"""

import numpy as np
from scipy.special import jv
from scipy.optimize import fsolve


def cylinder_bessel_kernels(n: int) -> np.ndarray:
    """
    Compute the first n roots of J0(x) = J1(x)/x (cylinder boundary condition).

    Parameters
    ----------
    n : number of roots

    Returns
    -------
    roots : [n] array of roots
    """
    roots = np.zeros(n, dtype=float)
    for l in range(n):
        x0 = 2.0 + l * np.pi
        roots[l] = fsolve(lambda x: jv(0, x) - jv(1, x) / x, x0)[0]
    return roots


def Dw_cylinder(
    omega: float | np.ndarray,
    R: float,
    D0: float,
    alpha: float,
    n: int = 60,
) -> np.ndarray:
    """
    Frequency-dependent apparent diffusivity for cylinders (GPA model).

    D(omega) = D0*alpha + D0*(1-alpha) * sum_l [ a_l * B_l * omega^2 / (a_l^2*D0^2 + omega^2) ]

    Parameters
    ----------
    omega : angular frequency [rad/s]
    R     : cylinder radius [m]
    D0    : free diffusivity [m^2/s]
    alpha : long-time diffusivity fraction (D_inf / D0)
    n     : number of Bessel terms

    Returns
    -------
    D : frequency-dependent diffusivity [m^2/s], same shape as omega
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    Kc = cylinder_bessel_kernels(n)

    B = 2.0 * (R / Kc) ** 2 / (Kc ** 2 - 1.0)
    a = (Kc / R) ** 2

    f = np.zeros_like(omega)
    for i in range(len(omega)):
        f[i] = D0 * alpha + D0 * (1.0 - alpha) * np.sum(
            a * B * omega[i] ** 2 / (a ** 2 * D0 ** 2 + omega[i] ** 2)
        )
    return f


def alpha_from_bt(bt: np.ndarray) -> np.ndarray:
    """
    Compute the long-time diffusivity fraction alpha from b-tensor principal values.

    alpha = b_z / (b_x + b_y + b_z)

    Parameters
    ----------
    bt : [N x 3] array of b-tensor diagonal [b_x, b_y, b_z]

    Returns
    -------
    alpha : [N] array
    """
    bt = np.asarray(bt, dtype=float)
    return bt[:, 2] / (bt[:, 0] + bt[:, 1] + bt[:, 2])
