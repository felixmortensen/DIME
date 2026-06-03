function xyz = loadtraj(Nsteps, filename)
% function xyz = dime.histosim.loadtraj(Nsteps, filename)
%
% Load a binary Histo-uSim trajectory file into a [Nsteps x 3 x Nparticles]
% array.  Positions are converted from mm to m.
%
%   Inputs
%   ------
%   Nsteps   - Number of time steps in the file.
%   filename - Path to binary .traj file (float32, stored as [3 x Nsteps x N]).
%
%   Outputs
%   -------
%   xyz - Particle positions [Nsteps x 3 x Nparticles] in meters.

fid  = fopen(filename, 'rb');
data = fread(fid, 'float32');
fclose(fid);

xyz = reshape(data/1e3, 3, Nsteps, []); % mm -> m
xyz = permute(xyz, [2, 1, 3]);
