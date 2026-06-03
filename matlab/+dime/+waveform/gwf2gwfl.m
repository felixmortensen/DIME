function [GWF, RF, DT, xps] = gwf2gwfl(gwf, rf, dt, bval, nrot, rot_file, onam)
% function [GWF, RF, DT, xps] = dime.waveform.gwf2gwfl(gwf, rf, dt, bval, nrot, rot_file, onam)
%
% Build a gradient waveform list (GWFL) suitable for signal simulation.
%
% Takes an [n x 3 x n_wf] waveform array (e.g. STE in slice 1, LTE in slice 2),
% applies a set of GFO rotation matrices, scales each waveform to the requested
% b-values, and compiles all combinations into a single GWFL array together with
% an XPS struct (b-value, b_delta, wf_ind, etc.).
%
% For 2D simulations (do_2d = true), the z-component is zeroed and the waveform
% is re-scaled to preserve the original b-value.
%
%   Inputs
%   ------
%   gwf      - [n x 3 x n_wf] gradient waveform(s) [T/m].
%              Typically n_wf = 2: (:,:,1) = STE, (:,:,2) = LTE.
%   rf       - [n x 1] refocusing sign vector (+1/-1)
%   dt       - raster time [s]
%   bval     - [1 x n_b] target b-values [ms/um^2] (e.g. [0 0.1 0.5 1 2])
%   nrot     - number of rotations (must match size of rot_file, or use 'evenly'
%              for 2D evenly spaced rotations about z)
%   rot_file - path to a .mat file containing variable rotMats [3 x 3 x nrot],
%              or the string 'evenly' to use uniformly spaced 2D rotations
%   onam     - output filename (without extension) to save the GWFL .mat file.
%              Pass '' or [] to skip saving.
%
%   Outputs
%   -------
%   GWF - [n x 3 x (n_wf * n_b * nrot)] full waveform list [T/m]
%   RF  - [n x (n_wf * n_b * nrot)] refocusing vectors
%   DT  - [1 x (n_wf * n_b * nrot)] raster times [s]
%   xps - struct with fields b, b_delta, m, m_delta, wf_ind (from fwf.gwf.toXps)

n_wf = size(gwf, 3);
n    = size(gwf, 1);

% Load or generate rotation matrices
if ischar(rot_file) && strcmp(rot_file, 'evenly')
    for i = 1:nrot
        R(:,:,i) = fwf.util.rotMat.zRot(360 / nrot * (i - 1));
    end
else
    rot = load(rot_file);
    R   = rot.rotMats;
    assert(size(R, 3) >= nrot, ...
        'rot_file contains %d rotations but nrot = %d was requested.', size(R,3), nrot);
    R = R(:,:,1:nrot);
end

% Compile waveform list
n_total = n_wf * numel(bval) * nrot;
GWF     = zeros(n, 3, n_total);
RF      = zeros(n, n_total);
DT      = zeros(1, n_total);
wf_ind  = zeros(1, n_total);

c = 1;
for i = 1:n_wf
    for j = 1:numel(bval)
        for k = 1:nrot
            tmp = gwf(:,:,i) * R(:,:,k);
            GWF(:,:,c) = fwf.gwf.force.bval(tmp, rf, dt, bval(j) * 1e9, 'amp');
            RF(:,c)    = rf;
            DT(c)      = dt;
            wf_ind(c)  = i;
            c = c + 1;
        end
    end
end

xps        = fwf.gwf.toXps(GWF, RF, DT);
xps.wf_ind = wf_ind;

% Save
if ~isempty(onam)
    save(onam, 'GWF', 'RF', 'DT', 'xps', 'wf_ind');
end
