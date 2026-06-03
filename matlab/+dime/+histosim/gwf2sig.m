function mri_sig = gwf2sig(Nsteps, Tdur, traj, gwf, gamma)
% function mri_sig = dime.histosim.gwf2sig(Nsteps, Tdur, traj, gwf, gamma)
% By Filip Szczepankiewicz, Lund University
% Inspired by code from Athanasios Grigoriou.
%
% Compute the noise-free MRI signal from a set of particle trajectories and
% gradient waveforms.  The code is general to 3D and uses fast matrix
% multiplication.
%
%   Inputs
%   ------
%   Nsteps - Number of time steps in the simulation.
%   Tdur   - Total duration of the trajectory walk [s].
%   traj   - Trajectory array [Nsteps x 3 x Nparticles] or path to .traj
%            file (see dime.histosim.loadtraj).  Units: m.
%   gwf    - Gradient waveform [Nsteps x 3 x Nmeasurements] or path to .mat
%            file containing variable GWF.  Units: T/m.
%   gamma  - Gyromagnetic ratio [rad/T/s].  Default: 267.522187e6.
%
%   Outputs
%   -------
%   mri_sig - Noise-free signal magnitude [Nmeasurements x 1].

if nargin < 5
    gamma = 267.522187e6;
end

% Load gradient waveforms
if isnumeric(gwf)
    GWF = gwf;
else
    tmp = load(gwf);
    GWF = tmp.GWF;
end

n_gstep = size(GWF, 1);
n_meas  = size(GWF, 3);

if n_gstep > Nsteps
    error('gwf cannot be longer than simulation interval!');
end

% Load trajectories
if isnumeric(traj)
    pos = traj;
else
    pos = dime.histosim.loadtraj(Nsteps, traj);
    pos = pos - pos(1,:,:);
end

dt = Tdur/Nsteps; % time step [s]

% Crop trajectory if needed
if n_gstep < Nsteps
    pos = pos(1:n_gstep, :, :);
end

mri_sig = zeros(n_meas, 1);

for i = 1:n_meas
    phi       = gamma * sum(pos .* GWF(:,:,i), [1 2]) * dt;
    fin_state = exp( 1i * phi(:) );
    fin_state(isnan(fin_state)) = [];
    mri_sig(i) = abs(mean(fin_state));
end
