# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DIME (Double-Isotropic Matched Encoding) — gradient waveform design for tensor-valued diffusion MRI under time-dependent diffusion. Companion code for: Mortensen et al., "Optimized gradient waveforms for tensor-valued diffusion MRI under time-dependent diffusion using DIME" (in submission).

## Commands

### MATLAB

Run the demo (Prisma 3T and CIMA.X scenarios):
```matlab
addpath(genpath('matlab'))   % add +dime package to path
dime.optimize.demo()
```

Run the optimizer directly:
```matlab
opt = dime.optimize.options(gmax, smax, dur);
[gwf, t, x] = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);
[gwf, rf, dt] = dime.waveform.ana2num(gwf, t/1000, 46e-6);
```

### Python

Install dependencies:
```bash
pip install -r requirements.txt
```

Run Monte Carlo simulation:
```python
from dime.simulate import simulate
simulate(folder='path/to/waveforms', geometry='cylinder', radii=radii)
```

Fit MD from simulation results:
```python
from dime.fitting import fit_md_from_npz
fit_md_from_npz('path/to/signals.npz')
```

## Architecture

### Repository layout

```
DIME/
├── matlab/+dime/       ← MATLAB package
│   ├── +optimize/      ← optimizer (fmincon, SAFE, multi-start)
│   ├── +waveform/      ← trapezoidal pulse construction, b/m tensor calc, LTE projection
│   ├── +histosim/      ← signal from Histo-µSim trajectories
│   ├── +analysis/      ← MD fitting, CV, ΔMD across rotations
│   └── +plot/          ← figures
├── python/dime/        ← Python package
│   ├── simulate.py     ← Disimpy Monte Carlo (cylinders & spheres)
│   ├── analytical.py   ← analytical signal via GPA model (Appendix C)
│   ├── fitting.py      ← powder-averaged MD fitting (Eq. 18)
│   ├── waveform.py     ← q(t) calculations, MATLAB file loading
│   └── plot.py         ← CV and MD-vs-radius figures
├── data/raw/           ← read-only (Histo-µSim trajectories, rotation matrices)
└── figures/            ← saved output figures
```

### MATLAB package (`matlab/+dime/`)

**Calling convention:** `dime.<subpackage>.<function>()`

| Function | Description |
|---|---|
| `dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt)` | Main DIME optimizer |
| `dime.optimize.options(gmax, smax, dur)` | Build optimizer config struct |
| `dime.optimize.demo()` | Run demo on Prisma and CIMA.X scenarios |
| `dime.waveform.par2gwf(x, dur, tp)` | 7 parameters → gradient waveform + time vector |
| `dime.waveform.par2bval(amp, tru, trd, ftt, tp)` | Closed-form b-value |
| `dime.waveform.par2mval(amp, tru, trd, ftt)` | Closed-form m-value |
| `dime.waveform.ste2lte(gwf)` | Project STE → LTE via u = (-1,+1,+1)/√3 (Eq. 15) |
| `dime.waveform.ana2num(gwf, t, dt)` | Rasterize piecewise-linear waveform, apply rf sign |
| `dime.histosim.gwf2sig(Nsteps, Tdur, traj, gwf)` | Signal from Histo-µSim trajectories |
| `dime.histosim.loadtraj(Nsteps, filename)` | Load binary .traj trajectory file |
| `dime.analysis.fitSignals(data_dir, wf_names, n_rot, do_2d)` | Cumulant fit of Histo-µSim signals |
| `dime.plot.gwfSetAndStim(gwf, rf, dt, hw)` | 4-panel waveform + stimulation figure |

**Optimization parameterization:** The waveform is described by 7 parameters — `alpha` (xy amplitude fraction), `beta` (timing split), four ramp times, and a global amplitude scale. `dime.waveform.par2gwf` expands these into the full 3-axis control-point waveform; `dime.waveform.ana2num` rasterizes it.

**Stimulation modes (mode argument):** 0 = no constraints; 1–7 = progressively stricter combinations of per-axis PNS, L2-norm of STE, and worst-case LTE across physical axes.

### Python package (`python/dime/`)

| Module | Key functions |
|---|---|
| `waveform` | `load_mat()`, `to_qt()`, `to_q_spectrum()` |
| `simulate` | `simulate(folder, geometry, radii, ...)` |
| `fitting`  | `fit_md(bvals, sig_ste, sig_lte)`, `fit_md_from_npz(path)` |
| `analytical` | `Dw_cylinder(omega, R, D0, alpha)`, `cylinder_bessel_kernels(n)` |
| `plot`     | `plot_cv_vs_radius()`, `boxplot_cv()`, `plot_md_vs_radius()` |

**Data flow:** MATLAB optimizer → .mat waveform files → `dime.simulate.simulate()` → .npz signal files → `dime.fitting.fit_md_from_npz()` → .npz with D_STE/D_LTE → `dime.plot.*`.

## Key physics and notation

| Symbol | Meaning | Units |
|---|---|---|
| `g(t)` | Gradient waveform | mT/m |
| `q(t)` | Dephasing vector | rad/m |
| `B` | b-tensor (3×3) | ms/µm² |
| `M` | m-tensor (3×3) | µs⁻² |
| `b` | trace(B) | ms/µm² |
| `m` | trace(M) | µs⁻² |
| `tau` | Encoding duration | ms |
| `gmax` | Max gradient amplitude | mT/m |
| `smax` | Max slew rate | T/m/s |
| `tp` | Pause around refocusing pulse | ms |
| `dt` | Waveform raster time | µs (46 µs standard) |

DIME enforces B = bI and M = mI (both isotropic) with equal m/b ratio across STE and LTE.

## Constraints and tolerances

- b-tensor isotropy: `|bxy − bz| / 2b < 0.01`
- m-tensor isotropy: `|mxy − mz| / 2m < 0.01`
- SAFE stimulation: PNS and CNS below 95% of hardware limit
- Stimulation assessed on STE norm and all three physical axes for LTE

## External dependencies

**MATLAB (must be on path):**
- Optimization Toolbox — `fmincon`, `GlobalSearch`, `MultiStart`
- [safe_pns_prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction) — `safe_gwf_to_pns`, `safe_hw_*`, `safe_plot`
- [fwf toolbox](https://github.com/filip-szczepankiewicz/fwf_seq_tools) — used in `dime.plot.gwfSetAndStim`

**Python:** see `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `disimpy`)

## What NOT to do

- Do not reorganize the directory structure without confirming with the owner
- Do not change unit conventions silently — flag any ambiguity
- Do not modify anything in `data/raw/` (read-only reference data)
- Do not hardcode `gmax`, `smax`, or file paths — always parameterize
- Do not present untested numerics as validated results
