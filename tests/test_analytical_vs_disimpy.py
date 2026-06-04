"""
Validation: analytical GPA signals vs pre-computed Disimpy Monte Carlo signals.

Requires external data that lives outside the repository:
  data/waveforms/{200mTm,80mTm}/b1000/GWFL_*.mat
  data/Simulations/{200mTm,80mTm}/Disimpy/{Cylinders,Spheres}/GWFL_*_sig.npz

All tests are skipped automatically when the data folder is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Data root — set via DIME_DATA_DIR env var, or auto-detected from known paths
# ---------------------------------------------------------------------------

import os

def _find_data_root() -> Path | None:
    env = os.environ.get("DIME_DATA_DIR")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # Auto-detect: sibling Projects/IsoIso/data relative to this repo
    candidates = [
        Path(__file__).parents[3] / "Projects" / "IsoIso" / "data",
        Path(__file__).parents[2] / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

_DATA_ROOT = _find_data_root()
_WF_ROOT   = _DATA_ROOT / "waveforms"    if _DATA_ROOT else Path("_missing")
_SIM_ROOT  = _DATA_ROOT / "Simulations"  if _DATA_ROOT else Path("_missing")

DATA_MISSING = _DATA_ROOT is None
skip_no_data = pytest.mark.skipif(DATA_MISSING, reason="External data folder not found — set DIME_DATA_DIR")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wf(path: Path, n_rot: int = 100):
    from dime.waveform import load_mat
    wf   = load_mat(path)
    gwfl = np.asarray(wf.GWF[:, :, :n_rot], dtype=float)
    rf   = np.ones(gwfl.shape[0])
    dt   = float(wf.dt)
    return gwfl, rf, dt


def _load_npz_cyl(path: Path):
    """Cylinder .npz: single b-value, signal normalised by nWalkers."""
    d      = np.load(path, allow_pickle=True)
    radii  = np.asarray(d["radii"], dtype=float)
    n_w    = int(d["nWalkers"])
    pw     = np.asarray(d["STE_powder"], dtype=float)
    sg     = np.asarray(d["STE_sig"],    dtype=float)
    if pw.ndim == 2: pw = pw[:, 0]
    if sg.ndim == 3: sg = sg[:, 0, :]
    return radii, pw / n_w, sg / n_w


def _load_npz_sph(path: Path):
    """Sphere .npz: multi-b layout; pick b = 1 ms/µm²."""
    d      = np.load(path, allow_pickle=True)
    radii  = np.asarray(d["radii"], dtype=float)
    n_w    = int(d["nWalkers"])
    b_idx  = int(np.argmin(np.abs(d["bvals"] - 1.0)))
    pw     = np.asarray(d["STE_powder"], dtype=float)[:, b_idx]
    sg     = np.asarray(d["STE_sig"],    dtype=float)[:, b_idx, :]
    return radii, pw / n_w, sg / n_w


# ---------------------------------------------------------------------------
# Parametrised test cases
# ---------------------------------------------------------------------------

_WF_200   = _WF_ROOT  / "Cima"   / "b1000"
_WF_80    = _WF_ROOT  / "Prisma" / "b1000"
_CYL_200  = _SIM_ROOT / "Cima"   / "Disimpy" / "Cylinders"
_SPH_200  = _SIM_ROOT / "Cima"   / "Disimpy" / "Spheres"
_CYL_80   = _SIM_ROOT / "Prisma" / "Disimpy" / "Cylinders"
_SPH_80   = _SIM_ROOT / "Prisma" / "Disimpy" / "Spheres"

# Each entry: (id, geometry, waveform_mat, signal_npz, max_err_tol)
_CASES = [
    # 200 mT/m — cylinders
    ("200mTm_cyl_DIME",    "cyl", _WF_200/"GWFL_ii_ste_cima_b1000_v5.mat",            _CYL_200/"GWFL_ii_ste_cima_b1000_v5_sig.npz",            0.005),
    ("200mTm_cyl_NOW-EOP", "cyl", _WF_200/"GWFL_now_ste_cima_3d_efficient_b1000.mat", _CYL_200/"GWFL_now_ste_cima_3d_efficient_b1000_sig.npz",  0.005),
    ("200mTm_cyl_NOW-MTM", "cyl", _WF_200/"GWFL_now_ste_cima_3d_matched_b1000.mat",   _CYL_200/"GWFL_now_ste_cima_3d_matched_b1000_sig.npz",    0.005),
    ("200mTm_cyl_NOW-ETM", "cyl", _WF_200/"GWFL_now_ste_cima_3d_timed_b1000.mat",     _CYL_200/"GWFL_now_ste_cima_3d_timed_b1000_sig.npz",      0.005),
    # 200 mT/m — spheres
    ("200mTm_sph_DIME",    "sph", _WF_200/"GWFL_ii_ste_cima_b1000_v5.mat",            _SPH_200/"GWFL_ii_ste_cima_3d_v5_sig.npz",                0.005),
    ("200mTm_sph_NOW-EOP", "sph", _WF_200/"GWFL_now_ste_cima_3d_efficient_b1000.mat", _SPH_200/"GWFL_now_ste_cima_3d_efficient_sig.npz",         0.005),
    ("200mTm_sph_NOW-MTM", "sph", _WF_200/"GWFL_now_ste_cima_3d_matched_b1000.mat",   _SPH_200/"GWFL_now_ste_cima_3d_matched_sig.npz",           0.005),
    ("200mTm_sph_NOW-ETM", "sph", _WF_200/"GWFL_now_ste_cima_3d_timed_b1000.mat",     _SPH_200/"GWFL_now_ste_cima_3d_timed_sig.npz",             0.005),
    # 80 mT/m — cylinders
    ("80mTm_cyl_DIME",    "cyl", _WF_80/"GWFL_ii_ste_prisma_3d_b1000.mat",              _CYL_80/"GWFL_ii_ste_prisma_3d_b1000_sig.npz",             0.005),
    ("80mTm_cyl_NOW-EOP", "cyl", _WF_80/"GWFL_now_ste_prisma_3d_efficient_b1000.mat",   _CYL_80/"GWFL_now_ste_prisma_3d_efficient_b1000_sig.npz",  0.005),
    ("80mTm_cyl_NOW-MTM", "cyl", _WF_80/"GWFL_now_ste_prisma_3d_matched_b1000.mat",     _CYL_80/"GWFL_now_ste_prisma_3d_matched_b1000_sig.npz",    0.005),
    ("80mTm_cyl_NOW-ETM", "cyl", _WF_80/"GWFL_now_ste_prisma_3d_timed_b1000.mat",       _CYL_80/"GWFL_now_ste_prisma_3d_timed_b1000_sig.npz",      0.005),
    # 80 mT/m — spheres
    ("80mTm_sph_DIME",    "sph", _WF_80/"GWFL_ii_ste_prisma_3d_b1000.mat",              _SPH_80/"GWFL_ii_ste_prisma_3d_sig.npz",                   0.005),
    ("80mTm_sph_NOW-EOP", "sph", _WF_80/"GWFL_now_ste_prisma_3d_efficient_b1000.mat",   _SPH_80/"GWFL_now_ste_prisma_3d_efficient_sig.npz",         0.005),
    ("80mTm_sph_NOW-MTM", "sph", _WF_80/"GWFL_now_ste_prisma_3d_matched_b1000.mat",     _SPH_80/"GWFL_now_ste_prisma_3d_matched_sig.npz",           0.005),
    ("80mTm_sph_NOW-ETM", "sph", _WF_80/"GWFL_now_ste_prisma_3d_timed_b1000.mat",       _SPH_80/"GWFL_now_ste_prisma_3d_timed_sig.npz",             0.005),
]


@skip_no_data
@pytest.mark.parametrize("case_id,geo,wf_path,npz_path,tol", _CASES, ids=[c[0] for c in _CASES])
def test_analytical_matches_disimpy(case_id, geo, wf_path, npz_path, tol):
    """Powder-averaged analytical GPA signal must match Disimpy within `tol`."""
    from dime.analytical import signal_gpa_cylinder_rotations, signal_gpa_sphere_rotations

    if not wf_path.exists():
        pytest.skip(f"Waveform file not found: {wf_path.name}")
    if not npz_path.exists():
        pytest.skip(f"Simulation file not found: {npz_path.name}")

    gwfl, rf, dt = _load_wf(wf_path)

    if geo == "cyl":
        radii, powder_num, _ = _load_npz_cyl(npz_path)
        sigs_ana, _          = signal_gpa_cylinder_rotations(gwfl, rf, dt, radii=radii, D0=2e-9)
    else:
        radii, powder_num, _ = _load_npz_sph(npz_path)
        sigs_ana, _          = signal_gpa_sphere_rotations(gwfl, rf, dt, radii=radii, D0=2e-9)

    powder_ana = np.mean(sigs_ana, axis=1)
    max_err    = float(np.max(np.abs(powder_ana - powder_num)))

    assert max_err < tol, (
        f"Max |analytical − Disimpy| = {max_err:.4f} exceeds tolerance {tol} "
        f"(worst radius ≈ {radii[np.argmax(np.abs(powder_ana - powder_num))]*1e6:.1f} µm)"
    )
