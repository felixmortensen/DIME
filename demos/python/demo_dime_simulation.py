"""
demo_dime_simulation.py
=======================

End-to-end Python pipeline for the DIME waveform:

  1. Load the pre-built DIME GWFL waveform (waveforms/80/3D/GWFL_DIME_80_3d.mat)
  2. Run Disimpy Monte Carlo simulations in cylinders for STE and LTE
     across a small set of radii and b-values
  3. Compute the same signals analytically via the GPA model
  4. Plot signal vs b-value for both STE and LTE, comparing
     analytical (lines) and numerical (markers)

Run from the repository root:
    python demos/python/demo_dime_simulation.py

Requirements: disimpy, numpy, matplotlib, scipy  (see requirements.txt)

Note on Disimpy
---------------
Disimpy uses CUDA for GPU-accelerated Monte Carlo.  A CUDA-capable GPU and
the CUDA toolkit are required.  If you do not have a GPU, comment out the
"Monte Carlo simulation" section and use only the analytical part of this
script (the analytical computation is CPU-only and runs in seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from disimpy import simulations, substrates

# Make the dime package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parents[2] / "python"))

from dime.analytical import signal_gpa_cylinder_rotations
from dime.waveform import load_mat

# ── Configuration ─────────────────────────────────────────────────────────────

GWFL_FILE = Path(__file__).parents[2] / "waveforms" / "80" / "3D" / "GWFL_DIME_80_3d.mat"

# Cylinder radii for the demo (3 values covering small → large restriction)
RADII_UM = [3.0, 12.0, 36.0]          # µm
RADII    = np.array(RADII_UM) * 1e-6  # m

D0          = 2e-9   # Free diffusivity [m²/s]
N_WALKERS   = 2_000  # Walkers per simulation (increase for smoother curves)
N_ROT       = 10     # Rotations to use (subset of the 100 in the file)
SEED        = 42

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # one per radius

# ── Load waveform ──────────────────────────────────────────────────────────────

print(f"Loading waveform from {GWFL_FILE.name} ...")
wf     = load_mat(GWFL_FILE)
bvals  = np.asarray(wf.bval,  dtype=float).ravel()   # [ms/µm²]
n_b    = len(bvals)
n_rot  = int(wf.nrot)
dt     = float(wf.dt)                                 # s

# Split STE / LTE using wf_ind (1 = STE, 2 = LTE)
wf_ind = np.asarray(wf.wf_ind).ravel()
GWF    = np.asarray(wf.GWF, dtype=float)             # (2371, 3, 1000)

# Reshape to (time, xyz, n_wf=2, n_b, n_rot) following gwf2gwfl loop order
#   outer loop: wf (STE/LTE), middle: bval, inner: rotation
GWF_ste = GWF[:, :, wf_ind == 1].reshape(-1, 3, n_b, n_rot)  # (T, 3, n_b, n_rot)
GWF_lte = GWF[:, :, wf_ind == 2].reshape(-1, 3, n_b, n_rot)

# Use only first N_ROT rotations for the demo
GWF_ste = GWF_ste[:, :, :, :N_ROT]
GWF_lte = GWF_lte[:, :, :, :N_ROT]

rf = np.ones(GWF_ste.shape[0])  # rf absorbed into GWF by gwf2simfmt

print(f"  dt = {dt*1e6:.2f} µs,  b-values = {bvals} ms/µm²,  using {N_ROT}/{n_rot} rotations")

# ── Monte Carlo simulation ─────────────────────────────────────────────────────

print("\nRunning Disimpy Monte Carlo simulations ...")
print(f"  {len(RADII_UM)} radii × {n_b} b-values × 2 encodings × {N_ROT} rotations")
print(f"  n_walkers = {N_WALKERS}  (increase for production runs)\n")

# sig_*[r_idx, b_idx, rot_idx]
sig_ste_num = np.zeros((len(RADII), n_b, N_ROT))
sig_lte_num = np.zeros((len(RADII), n_b, N_ROT))

for r_idx, (R, R_um) in enumerate(zip(RADII, RADII_UM)):
    substrate = substrates.cylinder(
        radius=R,
        orientation=np.array([0.0, 0.0, 1.0]),
    )
    for b_idx in range(n_b):
        grad_ste = np.transpose(GWF_ste[:, :, b_idx, :], (2, 0, 1))  # (N_ROT, T, 3)
        grad_lte = np.transpose(GWF_lte[:, :, b_idx, :], (2, 0, 1))

        sig_ste_num[r_idx, b_idx] = simulations.simulation(
            n_walkers=N_WALKERS, diffusivity=D0,
            gradient=grad_ste, dt=dt, substrate=substrate,
            seed=SEED, quiet=True,
        ) / N_WALKERS

        sig_lte_num[r_idx, b_idx] = simulations.simulation(
            n_walkers=N_WALKERS, diffusivity=D0,
            gradient=grad_lte, dt=dt, substrate=substrate,
            seed=SEED, quiet=True,
        ) / N_WALKERS

        print(f"  R={R_um:.0f} µm, b={bvals[b_idx]:.1f}: "
              f"STE={sig_ste_num[r_idx,b_idx].mean():.3f}  "
              f"LTE={sig_lte_num[r_idx,b_idx].mean():.3f}")

# Powder average over rotations
powder_ste_num = sig_ste_num.mean(axis=2)  # (n_radii, n_b)
powder_lte_num = sig_lte_num.mean(axis=2)

# ── Analytical GPA signals ─────────────────────────────────────────────────────
#
# The GPA signal scales with gradient amplitude as S(b) = exp(-b * beta_1),
# where beta_1 is the log-attenuation at b = 1 ms/µm².  We extract the b = 1
# waveforms (index 3 in bvals = [0, 0.1, 0.5, 1.0, 2.0]) and compute beta_1
# analytically, then evaluate at all b-values.

b1_idx = int(np.argmin(np.abs(bvals - 1.0)))

print(f"\nComputing analytical GPA signals (b=1 waveform, {N_ROT} rotations) ...")

gwfl_ste_b1 = GWF_ste[:, :, b1_idx, :]  # (T, 3, N_ROT)
gwfl_lte_b1 = GWF_lte[:, :, b1_idx, :]

_, betas_ste = signal_gpa_cylinder_rotations(gwfl_ste_b1, rf, dt, radii=RADII, D0=D0)
_, betas_lte = signal_gpa_cylinder_rotations(gwfl_lte_b1, rf, dt, radii=RADII, D0=D0)
# betas_*: (n_radii, N_ROT) — log-attenuation at b=1 per rotation

# Powder-averaged log-attenuation at b=1
beta_ste_mean = betas_ste.mean(axis=1)  # (n_radii,)
beta_lte_mean = betas_lte.mean(axis=1)

# Scale to all b-values: S(b) = exp(-b * beta_1)
b_fine = np.linspace(0, bvals.max(), 100)
powder_ste_ana = np.exp(-np.outer(beta_ste_mean, b_fine))  # (n_radii, 100)
powder_lte_ana = np.exp(-np.outer(beta_lte_mean, b_fine))

print("  Done.\n")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, title, powder_num, powder_ana in zip(
    axes,
    ["STE (spherical tensor)", "LTE (linear tensor)"],
    [powder_ste_num, powder_lte_num],
    [powder_ste_ana, powder_lte_ana],
):
    for r_idx, (R_um, color) in enumerate(zip(RADII_UM, COLORS)):
        label = f"R = {R_um:.0f} µm"
        ax.plot(b_fine, powder_ana[r_idx], color=color, lw=2, label=label)
        ax.scatter(bvals, powder_num[r_idx], color=color, s=50, zorder=5,
                   marker="o", label="_")

    ax.set_xlabel("b-value  [ms/µm²]", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, bvals.max())
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Powder-averaged signal", fontsize=12)
axes[0].legend(title="Analytical (line)\nNumerical (dot)", fontsize=10)

fig.suptitle("DIME signal vs b-value — cylinders  (D₀ = 2 µm²/ms)", fontsize=13)
fig.tight_layout()

out_fig = Path(__file__).parent / "demo_dime_simulation.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_fig}")
plt.show()
