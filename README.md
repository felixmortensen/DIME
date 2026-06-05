# DIME — Double-Isotropic Matched Encoding

A MATLAB and Python toolbox for designing gradient waveforms for tensor-valued diffusion MRI under time-dependent diffusion.

Repository based on the paper:

[Optimized gradient waveforms for tensor-valued diffusion MRI under time-dependent diffusion using Double-Isotropic Matched Encoding (DIME)](https://doi.org/),

by Felix Mortensen, Viktor Olsson, Athanasios Grigoriou, Samo Lasič, Malwina Molendowska, Ronnie Wirestam, and Filip Szczepankiewicz

---

# Citation

If you use this toolbox in your work, please cite:

> Mortensen et al., "Optimized gradient waveforms for tensor-valued diffusion MRI under time-dependent diffusion using DIME".

## Overview

Spherical b-tensor encoding (STE) enables rotationally invariant diffusion measurements, and STE/LTE comparisons are sensitive to microscopic diffusion anisotropy. Under time-dependent diffusion, however, this invariance can be compromised when restriction weighting is directionally uneven.

**DIME** addresses this by designing gradient waveforms that are simultaneously isotropic in both diffusion weighting (b-value) and low-frequency restriction weighting (m-value), with an LTE matched to the STE in m/b. The design enforces:

- **Double isotropy** — both the b-tensor B and the m-tensor M are spherical (B = bI, M = mI)
- **Spectral matching** — STE and LTE share the same m/b (Eq. 11), so they report the same apparent diffusivity under time-dependent diffusion
- **Gradient balance** with concomitant-gradient compensation (K-nulling)
- **Nerve stimulation compliance** — PNS and CNS via the SAFE model

---

## Repository structure

```
DIME/
├── matlab/+dime/
│   ├── +optimize/      optimize.m, options.m, demo.m
│   ├── +waveform/      par2gwf, par2bval, par2mval, ste2lte, ste2lte_matched,
│   │                   ana2num, gwf2simfmt, gwf2gwfl
│   ├── +histosim/      gwf2sig.m, loadtraj.m
│   ├── +analysis/      fitSignals.m
│   └── +plot/          gwfSetAndStim, spiderplot, pns_cns, bm_glyphs,
│                       wf_spectra, con_and_q, fit_histosim
├── python/dime/
│   ├── waveform.py     load_mat, to_qt, to_q_spectrum
│   ├── simulate.py     simulate() — Disimpy Monte Carlo
│   ├── analytical.py   Dw_restricted, GPA signals for cylinders and spheres
│   ├── fitting.py      fit_md, fit_md_from_npz
│   └── plot.py         CV vs radius, MD vs radius, efficiency figures
├── waveforms/          Pre-optimized GWFL .mat files
│   ├── 80/2D|3D/       GWFL_{DIME,NOWOE,NOWRM,NOWET}_80_{2d,3d}.mat
│   └── 200/2D|3D/      GWFL_{DIME,NOWOE,NOWRM,NOWET}_200_{2d,3d}.mat
├── rotmatrix/          gfod2_050.mat, gfod2_100.mat
├── demos/
│   ├── matlab/         demo_dime_waveform.m
│   └── python/         demo_dime_simulation.py
├── tests/              Python unit tests and Disimpy validation tests
├── data/raw/           Read-only reference data (not distributed)
└── requirements.txt    Python dependencies
```

---

## Quick start

### MATLAB — design a DIME waveform

Add the `matlab/` folder to the MATLAB path, then run the demo:

```matlab
addpath(genpath('matlab'))
dime.optimize.demo()
```

Or step through the full pipeline manually:

```matlab
addpath(genpath('matlab'))

% Scanner parameters
dur  = 40;   % ms — encoding duration per lobe
tp   = 8;    % ms — pause around refocusing pulse
gmax = 0.08; % T/m
smax = 200;  % T/m/s
mode = 7;    % stimulation constraint mode (see table below)

% Load SAFE hardware model for your scanner (requires safe_pns_prediction on path)
hw = safe_example_hw_peripheral;  % replace with your scanner hw struct

% Optimize, rasterise, and plot
opt           = dime.optimize.options(gmax, smax, dur);
[gwf, t]      = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);
[gwf, rf, dt] = dime.waveform.ana2num(gwf, t/1000, 46e-6);
dime.plot.gwfSetAndStim(gwf, rf, dt, hw)
```

For a complete pipeline from optimization through to a simulation-ready GWFL file (including LTE construction and rotation), see `demos/matlab/demo_dime_waveform.m`.

### Python — simulation and analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Pre-optimized waveforms for 80 mT/m and 200 mT/m systems are provided in `waveforms/`. To run Monte Carlo simulations with Disimpy (requires a CUDA GPU):

```python
from dime.simulate import simulate
import numpy as np

radii = np.linspace(3, 90, 30) * 1e-6  # m
simulate(folder='waveforms/80/3D', geometry='cylinder', radii=radii, n_walkers=int(1e6))
```

To compute the equivalent GPA analytical signal:

```python
from dime.waveform import load_mat
from dime.analytical import signal_gpa_cylinder_rotations
import numpy as np

wf   = load_mat('waveforms/80/3D/GWFL_DIME_80_3d.mat')
gwfl = np.asarray(wf.GWF[:, :, :100])   # first 100 rotations (STE)
rf   = np.ones(wf.rf[:100])
radii = np.array([5, 10, 20]) * 1e-6

signals, betas = signal_gpa_cylinder_rotations(gwfl, rf, float(wf.dt), radii, D0=2e-9)
```

For a complete end-to-end example including both simulation and analytical comparison with a figure, see `demos/python/demo_dime_simulation.py`.

---

## Stimulation constraint modes

The `mode` argument controls which SAFE stimulation constraints are applied during optimization.

| Mode | Constraint applied |
|------|--------------------|
| 0    | No stimulation constraints |
| 1    | Max PNS per axis (STE) |
| 2    | Global max PNS (STE) |
| 3    | L2-norm of PNS (STE) |
| 4    | Per-axis + L2-norm (STE) |
| 5    | LTE along y only |
| 6    | Max PNS for STE and worst-case LTE |
| 7    | Max PNS for STE + LTE + L2-norm of STE |

---

## Key notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| g(t)   | Gradient waveform | T/m |
| q(t)   | Dephasing vector | rad/m |
| B      | b-tensor (3×3) | s/m² |
| M      | m-tensor (3×3) | m⁻²s⁻¹ |
| b      | trace(B) | ms/µm² |
| m      | trace(M) | m⁻²s⁻¹ |
| tau    | Encoding duration | ms |
| gmax   | Max gradient amplitude | T/m |
| smax   | Max slew rate | T/m/s |
| tp     | Pause around refocusing pulse | ms |
| dt     | Waveform raster time | ms |

---

## Dependencies

**MATLAB (R2025b)**
- Optimization Toolbox — `fmincon`, `GlobalSearch`, `MultiStart`
- [safe_pns_prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction)
- [fwf_seq_tools](https://github.com/filip-szczepankiewicz/fwf_seq_tools)

**Python**
- See `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `disimpy`)
- Disimpy requires a CUDA-capable GPU

---
