function [gwf_out, rf_out, dt_out] = gwf2simfmt(gwf, rf, dt, dt_target, n_total)
% function [gwf_out, rf_out, dt_out] = dime.waveform.gwf2simfmt(gwf, rf, dt, dt_target, n_total)
%
% Prepare a gradient waveform for signal simulation by resampling to a fixed
% temporal resolution, enforcing gradient balance, and zero-padding to a fixed
% total length.  The rf sign is absorbed into the waveform (gwf_out = gwf.*rf)
% so that rf_out is all ones — the format expected by histosim and Disimpy.
%
%   Inputs
%   ------
%   gwf       - [n x 3] gradient waveform [T/m]
%   rf        - [n x 1] refocusing sign vector (+1/-1)
%   dt        - current raster time [s]
%   dt_target - target raster time [s] (default: 46.41e-6 s)
%   n_total   - total number of time steps after padding (default: 2371)
%
%   Outputs
%   -------
%   gwf_out - [n_total x 3] resampled, balanced, padded waveform with rf baked in [T/m]
%   rf_out  - [n_total x 1] vector of ones
%   dt_out  - raster time [s] (equals dt_target)

if nargin < 4 || isempty(dt_target), dt_target = 46.41e-6; end  % s
if nargin < 5 || isempty(n_total),   n_total   = 2371;     end

% Resample to target temporal resolution
n_samp = round((size(gwf, 1) - 1) * dt / dt_target);
[gwf, rf, dt_out] = fwf.gwf.toInterpolated(gwf, rf, dt, n_samp);

% Enforce gradient balance
[gwf, rf, dt_out] = fwf.gwf.force.balance_v2(gwf, rf, dt_out);

% Zero-pad to n_total steps (symmetric)
ncur = size(gwf, 1);
nza  = round((n_total - ncur) / 2);
nzb  = n_total - ncur - nza;

gwf = [zeros(nza, 3); gwf; zeros(nzb, 3)];
rf  = [ones(nza,  1) * rf(1); rf; ones(nzb, 1) * rf(end)];

% Absorb rf into waveform
gwf_out = gwf .* rf;
rf_out  = ones(size(rf));
