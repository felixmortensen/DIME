"""
DIME simulation demo — signal vs b-value for cylinders.
Requires: disimpy (CUDA GPU), numpy, matplotlib
Run from the repository root:  python demos/python/demo_dime_simulation.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from disimpy import simulations, substrates

sys.path.insert(0, str(Path(__file__).parents[2] / "python"))
from dime.waveform import load_mat
from dime.analytical import signal_gpa_cylinder_rotations

# --- Settings ---
GWFL    = Path("waveforms/80/3D/GWFL_DIME_80_3d.mat")
RADII   = np.array([3.0, 12.0, 36.0]) * 1e-6   # m
D0      = 2e-9                                   # m²/s
N_ROT   = 10     # rotations to use (100 available; increase for smoother curves)
N_WALK  = 2_000  # walkers per simulation        (increase for less noise)

# --- Load waveform ---
wf     = load_mat(GWFL)
bvals  = np.asarray(wf.bval, dtype=float).ravel()
dt     = float(wf.dt)
n_b, n_rot = len(bvals), int(wf.nrot)

wf_ind  = np.asarray(wf.wf_ind).ravel()
GWF     = np.asarray(wf.GWF, dtype=float)
GWF_ste = GWF[:, :, wf_ind == 1].reshape(-1, 3, n_b, n_rot)[:, :, :, :N_ROT]
GWF_lte = GWF[:, :, wf_ind == 2].reshape(-1, 3, n_b, n_rot)[:, :, :, :N_ROT]
rf      = np.ones(GWF_ste.shape[0])  # rf already absorbed into GWF

# --- Monte Carlo simulation ---
sig_ste = np.zeros((len(RADII), n_b, N_ROT))
sig_lte = np.zeros_like(sig_ste)

for i, R in enumerate(RADII):
    sub = substrates.cylinder(radius=R, orientation=np.array([0., 0., 1.]))
    for j in range(n_b):
        sig_ste[i, j] = simulations.simulation(N_WALK, D0,
            np.transpose(GWF_ste[:, :, j, :], (2, 0, 1)), dt, sub, seed=42, quiet=True) / N_WALK
        sig_lte[i, j] = simulations.simulation(N_WALK, D0,
            np.transpose(GWF_lte[:, :, j, :], (2, 0, 1)), dt, sub, seed=42, quiet=True) / N_WALK

powder_ste_num = sig_ste.mean(axis=2)
powder_lte_num = sig_lte.mean(axis=2)

# --- Analytical GPA signals ---
# Compute log-attenuation at b=1, then scale: S(b) = exp(-b * beta_1)
b1 = int(np.argmin(np.abs(bvals - 1.0)))
_, betas_ste = signal_gpa_cylinder_rotations(GWF_ste[:, :, b1, :], rf, dt, RADII, D0)
_, betas_lte = signal_gpa_cylinder_rotations(GWF_lte[:, :, b1, :], rf, dt, RADII, D0)

b_fine = np.linspace(0, bvals.max(), 200)
powder_ste_ana = np.exp(-np.outer(betas_ste.mean(axis=1), b_fine))
powder_lte_ana = np.exp(-np.outer(betas_lte.mean(axis=1), b_fine))

# --- Plot ---
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
labels = [f"R = {r*1e6:.0f} µm" for r in RADII]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for i, (color, label) in enumerate(zip(colors, labels)):
    for ax, ana, num in [(ax1, powder_ste_ana, powder_ste_num),
                         (ax2, powder_lte_ana, powder_lte_num)]:
        ax.plot(b_fine, ana[i], color=color, lw=2, label=label)
        ax.scatter(bvals, num[i], color=color, s=40, zorder=5)

for ax, title in [(ax1, "STE"), (ax2, "LTE")]:
    ax.set(xlabel="b  [ms/µm²]", title=title, xlim=(0, bvals.max()), ylim=(0, 1.05))
    ax.grid(alpha=0.3)

ax1.set_ylabel("Powder-averaged signal")
ax1.legend(title="— analytical\n● numerical", fontsize=9)
fig.suptitle("DIME signal vs b — cylinders (D₀ = 2 µm²/ms)")
fig.tight_layout()
plt.savefig(Path(__file__).parent / "demo_dime_simulation.png", dpi=150, bbox_inches="tight")
plt.show()
