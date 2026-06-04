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
├── matlab/
│   └── +dime/
│       ├── +optimize/          % DIME waveform optimization (fmincon, SAFE, multi-start)
│       │   ├── optimize.m      % Main optimizer
│       │   ├── options.m       % Optimizer configuration struct
│       │   └── demo.m          % Demo for 80 and 200 mT/m gradient systems
│       ├── +waveform/          % Trapezoidal pulse construction and tensor calculations
│       │   ├── par2gwf.m       % 7 parameters -> gradient waveform + time vector
│       │   ├── par2bval.m      % Closed-form b-value from parameters
│       │   ├── par2mval.m      % Closed-form m-value from parameters
│       │   ├── ste2lte.m       % Project STE -> LTE (Eq. 15)
│       │   └── ana2num.m       % Rasterize piecewise-linear waveform
│       ├── +histosim/          % Signal computation from Histo-uSim trajectories
│       │   ├── gwf2sig.m       % Noise-free MRI signal from particle trajectories
│       │   └── loadtraj.m      % Load binary .traj trajectory files
│       ├── +analysis/          % Signal fitting and diffusivity analysis
│       │   └── fitSignals.m    % Cumulant fits and per-rotation MD from Histo-uSim data
│       └── +plot/              % Figures
│           └── gwfSetAndStim.m % Waveform + nerve stimulation figure
├── python/
│   └── dime/
│       ├── simulate.py         % Disimpy Monte Carlo (cylinders & spheres)
│       ├── analytical.py       % Analytical diffusion spectra via GPA (Appendix C)
│       ├── fitting.py          % Powder-averaged MD fitting (Eq. 18)
│       ├── waveform.py         % q(t) calculations and MATLAB file loading
│       └── plot.py             % CV and MD-vs-radius figures
├── data/
│   └── raw/                    % Read-only reference data (not distributed)
├── figures/                    % Saved output figures
└── requirements.txt            % Python dependencies
```

---

## Quick start

### MATLAB — design a DIME waveform

Add the `matlab/` folder to the MATLAB path, then:

```matlab
addpath(genpath('matlab'))

% Scanner parameters
dur  = 40;   % ms — encoding duration per lobe
tp   = 8;    % ms — pause around refocusing pulse
gmax = 0.08; % T/m
smax = 200;  % T/m/s
mode = 7;    % stimulation constraint mode (see table below)

% Load SAFE hardware model for your scanner (requires safe_pns_prediction on path)
% Replace with your scanner-specific hw struct, e.g. safe_hw_<yourscanner>
hw = safe_example_hw_peripheral;

% Optimize
opt        = dime.optimize.options(gmax, smax, dur);
[gwf, t]   = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);

% Rasterize to 46 µs grid and plot
[gwf, rf, dt] = dime.waveform.ana2num(gwf, t/1000, 46e-6);
dime.plot.gwfSetAndStim(gwf, rf, dt, hw)
```

To run the built-in demo for two scanner scenarios (80 and 200 mT/m):

```matlab
dime.optimize.demo()
```

### Python — Monte Carlo simulation and MD fitting

Install dependencies:

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from dime.simulate import simulate
from dime.fitting import fit_md_from_npz
from dime.plot import load_md_from_npz_files, plot_md_vs_radius

# Run Disimpy Monte Carlo for a folder of waveform .mat files
radii = np.linspace(3, 90, 30) * 1e-6  # m
simulate(folder='path/to/waveforms', geometry='cylinder', radii=radii, n_walkers=int(1e6))

# Fit powder-averaged MD from the simulated signals
fit_md_from_npz('path/to/signals.npz')
```

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
| 7    | Max PNS for STE + LTE + L2-norm of STE **(recommended)** |

---

## Key notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| g(t)   | Gradient waveform | mT/m |
| q(t)   | Dephasing vector | rad/m |
| B      | b-tensor (3×3) | ms/µm² |
| M      | m-tensor (3×3) | µs⁻² |
| b      | trace(B) | ms/µm² |
| m      | trace(M) | µs⁻² |
| tau    | Encoding duration | ms |
| gmax   | Max gradient amplitude | mT/m |
| smax   | Max slew rate | T/m/s |
| tp     | Pause around refocusing pulse | ms |
| dt     | Waveform raster time | µs (46 µs standard) |

---

## Dependencies

**MATLAB (R2025b)**
- Optimization Toolbox — `fmincon`, `GlobalSearch`, `MultiStart`
- [safe_pns_prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction)
- [fwf_seq_tools](https://github.com/filip-szczepankiewicz/fwf_seq_tools)

**Python**
- See `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `disimpy`)

---
