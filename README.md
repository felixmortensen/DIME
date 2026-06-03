# DIME — Double-Isotropic Matched Encoding

Gradient waveform design for tensor-valued diffusion MRI under time-dependent diffusion.
Companion code for: Mortensen et al., "Optimized gradient waveforms for tensor-valued
diffusion MRI under time-dependent diffusion using DIME" (in submission).

Public release. Intended for researchers in diffusion MRI.

## Repository structure

```
DIME/
├── CLAUDE.md
├── README.md
├── matlab/
│   └── +dime/
│       ├── +optimize/     ← DIME & NOW waveform optimization (fmincon, SAFE, multi-start)
│       ├── +waveform/     ← trapezoidal pulse construction, LTE projection, b/m tensor calc
│       ├── +histosim/     ← signal computation from Histo-µSim trajectories
│       ├── +analysis/     ← MD fitting, CV, ΔMD across rotations
│       └── +plot/         ← figures
├── python/
│   └── dime/
│       ├── simulate.py    ← Disimpy Monte Carlo (cylinders & spheres)
│       ├── analytical.py  ← analytical signal computation (Appendix C, Lorentzian spectra)
│       ├── fitting.py     ← powder-averaged MD fitting (Eq. 18), signal analysis
│       └── plot.py        ← figures
├── data/
│   └── raw/              ← read-only: Histo-µSim trajectories, GFO rotation matrices
└── figures/              ← saved output figures
```

## Languages and packages

**MATLAB (R2025b)**
- Package namespace: `+dime/` with subpackages `+optimize`, `+simulate`,
  `+plot`, `+waveform`, `+safe`
- Optimization via `fmincon` (SQP), multi-start scheme
- SAFE model via `safe_pns_prediction` (Szczepankiewicz, GitHub)
- NOW toolbox for reference waveforms

**Python**
- Package: `dime/` with modules `simulate.py`, `fitting.py`, `waveform.py`, `plot.py`
- Monte Carlo via Disimpy
- Analytical signal computation (cylinders/spheres, Appendix C)
- Signal fitting: powder-averaged MD via Eq. (18), scipy curve_fit

## Key physics and notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| `g(t)` | Gradient waveform vector | mT/m |
| `q(t)` | Dephasing vector (integral of g) | rad/m |
| `B` | b-tensor (3×3) | ms/µm² |
| `M` | m-tensor / restriction-weighting tensor (3×3) | µs⁻² |
| `b` | trace(B), scalar b-value | ms/µm² |
| `m` | trace(M), scalar restriction weighting | µs⁻² |
| `kappa` | Waveform encoding efficiency (dimensionless) | — |
| `tau` | Diffusion encoding duration | ms |
| `gmax` | Maximum gradient amplitude | mT/m |
| `smax` | Maximum slew rate | T/m/s |
| `tp` | Pause time around refocusing pulse | ms |
| `dt` | Waveform time resolution (rasterization) | µs (46 µs standard) |

STE = spherical b-tensor encoding, LTE = linear b-tensor encoding.
DIME enforces B = bI and M = mI (both tensors isotropic) and equal m/b across STE/LTE.
LTE is constructed by projecting STE onto u = (±1,±1,±1)/√3 (Eq. 15 in paper).

## Constraints and safety

- b-tensor isotropy tolerance: |bxy − bz| / 2b < 0.01
- m-tensor isotropy tolerance: |mxy − mz| / 2m < 0.01
- Nerve stimulation (PNS and CNS) via SAFE model: must stay below 95% of limit
- Stimulation assessed on STE norm and all three physical axes for LTE

## Coding conventions

- MATLAB: one class or coherent function group per file, inside the `+dime` package namespace
- Python: module-level functions grouped by role; no globals; type hints on public functions
- Units must be explicit in variable names or docstrings — never mix ms and s, or mT/m and T/m
- Do not modify anything in `data/raw/` (read-only reference data)

## What NOT to do

- Do not reorganize the directory structure without confirming with the owner first
- Do not change unit conventions silently — flag any ambiguity
- Do not present untested numerics as validated results
