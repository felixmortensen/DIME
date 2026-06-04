% DIME waveform generation demo
% Requires: DIME matlab package, fwf toolbox, safe_pns_prediction toolbox
% Run from the repository root.

addpath(genpath('matlab'))

% Scanner and timing parameters
gmax = 0.08;  % T/m  (80 mT/m system)
smax = 200;   % T/m/s
dur  = 40;    % ms — encoding duration per lobe
tp   = 8;     % ms — pause around refocusing pulse

% Set mode = 0 to skip PNS constraints (no SAFE toolbox needed).
% Set mode = 7 and provide your hw struct to enforce PNS/CNS limits.
mode = 0;
hw   = [];

% --- Step 1: Optimize DIME waveform ---
opt = dime.optimize.options(gmax, smax, dur);
opt.numStartPts = 5;  % reduce for speed; remove for full optimization

[gwf, t] = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);

% --- Step 2: Rasterise control points to numerical waveform ---
[gwfI, rf, dt] = dime.waveform.ana2num(gwf, t/1000, 46.41e-6);

figure(1); dime.plot.gwfSetAndStim(gwfI, rf, dt, hw); title('DIME STE')

% --- Step 3: Prepare for simulation (resample, pad, absorb rf) ---
[gwfS, rfS, dtS] = dime.waveform.gwf2simfmt(gwfI, rf, dt);

% --- Step 4: Construct matched DIME LTE (Eq. 15 projection) ---
gwfL = dime.waveform.ste2lte(gwfS);

figure(2); dime.plot.gwfSetAndStim(gwfL, rfS, dtS, hw); title('DIME LTE')

% --- Step 5: Build GWFL with rotations and b-values ---
bval     = [0  0.1  0.5  1.0  2.0];          % ms/µm²
nrot     = 100;
rot_file = fullfile('rotmatrix', 'gfod2_100.mat');
onam     = fullfile('demos', 'matlab', 'GWFL_DIME_demo');

gwf_pair = cat(3, gwfS, gwfL);
dime.waveform.gwf2gwfl(gwf_pair, rfS, dtS, bval, nrot, rot_file, onam);
