"""
Waveform utilities: MATLAB file loading and q-space calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# MATLAB file I/O
# ---------------------------------------------------------------------------

class MatlabStruct:
    """Attribute-accessible container for MATLAB struct fields."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.__dict__)})"

    def keys(self):
        return self.__dict__.keys()

    def as_dict(self):
        return self.__dict__


def _convert(obj: Any) -> Any:
    """Recursively convert scipy-loaded MATLAB objects to Python types."""
    if hasattr(obj, "_fieldnames"):
        return MatlabStruct(**{f: _convert(getattr(obj, f)) for f in obj._fieldnames})
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        if obj.size == 1:
            return MatlabStruct(
                **{n: _convert(obj[n].item() if obj[n].size == 1 else obj[n])
                   for n in obj.dtype.names}
            )
        return [_convert(item) for item in obj]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 0:
            return _convert(obj.item())
        return [_convert(x) for x in obj.flat]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_mat(mat_file: str | Path, variable_name: str | None = None) -> MatlabStruct:
    """
    Load a .mat file into a MatlabStruct.

    Parameters
    ----------
    mat_file      : path to .mat file
    variable_name : if given, return only that variable; otherwise return all

    Returns
    -------
    MatlabStruct with all public variables as attributes.
    """
    data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    public = {k: v for k, v in data.items() if not k.startswith("__")}
    if not public:
        raise ValueError(f"No variables found in {mat_file}")
    if variable_name is not None:
        if variable_name not in public:
            raise KeyError(f"Variable '{variable_name}' not in {list(public)}")
        result = _convert(public[variable_name])
        return result if isinstance(result, MatlabStruct) else MatlabStruct(**{variable_name: result})
    return MatlabStruct(**{k: _convert(v) for k, v in public.items()})


# ---------------------------------------------------------------------------
# q-space calculations
# ---------------------------------------------------------------------------

GAMMA = 2.6751e8  # rad/s/T


def to_qt(gwf: np.ndarray, rf: np.ndarray, dt: float, gamma: float = GAMMA) -> np.ndarray:
    """
    Compute the dephasing vector q(t) = gamma * cumsum(g(t)*rf(t)) * dt.

    Parameters
    ----------
    gwf   : gradient waveform [n x 3] T/m
    rf    : refocusing sign vector [n] (+1/-1)
    dt    : raster time [s]
    gamma : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    qt : [n x 3] rad/m
    """
    gwf = np.asarray(gwf, dtype=float)
    rf  = np.asarray(rf,  dtype=float).reshape(-1, 1)
    return gamma * np.cumsum(gwf * rf, axis=0) * dt


def to_q_spectrum(
    gwf: np.ndarray,
    rf: np.ndarray,
    dt: float,
    n_pad: int | None = None,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT-based encoding power spectrum of the q-trajectory.

    Returns all 6 unique components of the b-tensor spectrum:
      p[:,0:2] = |F(qx)|^2, |F(qy)|^2, |F(qz)|^2
      p[:,3:5] = sqrt(2)*Re[F(qx)F*(qy)], ...(xz), ...(yz)

    Parameters
    ----------
    gwf   : [n x 3] T/m
    rf    : [n] refocusing vector
    dt    : raster time [s]
    n_pad : zero-padding length (default: 2*pi*n)

    Returns
    -------
    p : [N x 6] complex power spectrum
    f : [N] frequency axis [Hz]
    """
    gwf = np.asarray(gwf, dtype=float)
    rf  = np.asarray(rf,  dtype=float)
    if n_pad is None:
        n_pad = int(np.ceil(2 * np.pi * gwf.shape[0]))

    q = to_qt(gwf, rf, dt, gamma=gamma)
    q = np.vstack([np.zeros((n_pad, 3)), q, np.zeros((n_pad, 3))])

    N = q.shape[0]
    p = np.zeros((N, 6), dtype=complex)
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

    F = [None, None, None]
    for k in range(3):
        if np.any(q[:, k]):
            F[k] = np.fft.fft(q[:, k] * dt)
            p[:, k] = np.fft.fftshift(F[k] * np.conj(F[k]))

    pairs = [(0, 1, 3), (0, 2, 4), (1, 2, 5)]
    for i, j, c in pairs:
        if F[i] is not None and F[j] is not None:
            p[:, c] = np.fft.fftshift(F[i] * np.conj(F[j])) * np.sqrt(2)

    return p, f
