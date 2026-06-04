% demo_dime_waveform.m
%
% End-to-end pipeline for generating a DIME gradient waveform:
%   1. Optimize the DIME STE waveform with fmincon / MultiStart
%   2. Convert control points to a rasterised numerical waveform
%   3. Prepare for simulation (resample, balance, pad, absorb rf)
%   4. Construct matched DIME LTE via Eq. 15 projection
%   5. Build a GWFL file with rotations and b-values
%
% Requirements
% ------------
%   * DIME matlab package on the path:  addpath(genpath('matlab'))
%   * fwf toolbox (for fwf.gwf.* utilities inside the package functions)
%   * safe_pns_prediction toolbox for stimulation assessment
%     (https://github.com/filip-szczepankiewicz/safe_pns_prediction)
%     Set mode = 0 below to skip PNS constraints if unavailable.
%
% The rotation file rotmatrix/gfod2_100.mat contains 100 evenly distributed
% 3D orientations and is distributed with this repository.

clear; clc

addpath(genpath('matlab'))

% ── Scanner parameters ───────────────────────────────────────────────────────
gmax = 0.08;   % Maximum gradient amplitude [T/m]  (80 mT/m system)
smax = 200;    % Maximum slew rate [T/m/s]

% ── Timing ───────────────────────────────────────────────────────────────────
% dur: encoding duration per lobe [ms].  tp: pause around refocusing pulse [ms].
% These values give b ≈ 1 ms/µm² at 80 mT/m.
dur  = 40;    % ms
tp   = 8;     % ms
dt   = 46.41e-6;  % Raster time for numerical waveform [s]

% ── PNS model ────────────────────────────────────────────────────────────────
% mode = 0  → no stimulation constraints (fastest, no SAFE toolbox required)
% mode = 7  → combined PNS + CNS constraints via SAFE model
%   Replace safe_example_hw_peripheral with your scanner hw struct.
mode = 0;
hw   = [];
if mode > 0
    hw = safe_example_hw_peripheral;  % <-- replace with your scanner model
    hw.gmax = gmax;
    hw.smax = smax;
    hw.lim  = 95;  % Stimulation limit [%]
end

% ── GWFL parameters ──────────────────────────────────────────────────────────
bval     = [0  0.1  0.5  1.0  2.0];  % Target b-values [ms/µm²]
nrot     = 100;                        % Number of orientations
rot_file = fullfile('rotmatrix', 'gfod2_100.mat');  % Rotation matrix file
onam     = fullfile('demos', 'matlab', 'GWFL_DIME_demo');  % Output filename

% ─────────────────────────────────────────────────────────────────────────────
% Step 1 — Optimize DIME waveform
% ─────────────────────────────────────────────────────────────────────────────
fprintf('Step 1: Optimising DIME waveform (dur=%.0f ms, tp=%.0f ms, gmax=%.0f mT/m)...\n', ...
    dur, tp, gmax*1000)

opt = dime.optimize.options(gmax, smax, dur);

% For a quick demo, reduce the number of MultiStart points.
% Remove these two lines to use the defaults (slower but more thorough).
opt.numStartPts   = 5;
opt.numInitialPts = 20;

[gwf_ana, t_ana] = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);

fprintf('  Done. Waveform has %d control points per axis.\n', size(gwf_ana, 1))

% ─────────────────────────────────────────────────────────────────────────────
% Step 2 — Convert control points to rasterised numerical waveform
% ─────────────────────────────────────────────────────────────────────────────
fprintf('Step 2: Rasterising to %.2f µs steps...\n', dt*1e6)

[gwfI, rf, dtI] = dime.waveform.ana2num(gwf_ana, t_ana/1000, dt);

fprintf('  Done. Waveform: %d steps x 3 axes, dt = %.2f µs.\n', size(gwfI,1), dtI*1e6)

% ─────────────────────────────────────────────────────────────────────────────
% Step 3 — Prepare for simulation (resample to 46.41 µs, pad to 2371 steps)
% ─────────────────────────────────────────────────────────────────────────────
fprintf('Step 3: Converting to simulation format (balance, pad to 2371 steps)...\n')

[gwfS, rfS, dtS] = dime.waveform.gwf2simfmt(gwfI, rf, dtI);

fprintf('  Done. Sim waveform: %d steps, dt = %.2f µs.\n', size(gwfS,1), dtS*1e6)

% Plot STE waveform
figure(1); clf
dime.plot.gwfSetAndStim(gwfS, rfS, dtS, hw)
title('DIME STE — simulation format')

% ─────────────────────────────────────────────────────────────────────────────
% Step 4 — Construct matched DIME LTE (Eq. 15: u = [-1,+1,+1]/sqrt(3))
% ─────────────────────────────────────────────────────────────────────────────
fprintf('Step 4: Constructing matched DIME LTE via Eq. 15 projection...\n')

gwfL = dime.waveform.ste2lte(gwfS);

% Plot LTE waveform
figure(2); clf
dime.plot.gwfSetAndStim(gwfL, rfS, dtS, hw)
title('DIME LTE — simulation format')

% ─────────────────────────────────────────────────────────────────────────────
% Step 5 — Build GWFL with rotations and b-values
% ─────────────────────────────────────────────────────────────────────────────
fprintf('Step 5: Building GWFL (%d rotations, %d b-values, STE+LTE)...\n', nrot, numel(bval))

gwf_pair = cat(3, gwfS, gwfL);  % [2371 x 3 x 2]: STE in slice 1, LTE in slice 2

GWF = dime.waveform.gwf2gwfl(gwf_pair, rfS, dtS, bval, nrot, rot_file, onam);

fprintf('  Saved GWFL to %s.mat\n', onam)
fprintf('  GWF array: %d x %d x %d  (time x xyz x encodings)\n', size(GWF,1), size(GWF,2), size(GWF,3))
fprintf('  Total encodings: %d waveforms × %d b-values × %d rotations = %d\n', ...
    size(gwf_pair,3), numel(bval), nrot, size(GWF,3))

fprintf('\nPipeline complete.\n')
fprintf('Pass %s.mat to dime.simulate.simulate() in Python for Monte Carlo simulation.\n', onam)
