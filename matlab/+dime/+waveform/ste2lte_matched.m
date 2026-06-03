function gwf_lte = ste2lte_matched(gwf, rf, dt, gmax)
% function gwf_lte = dime.waveform.ste2lte_matched(gwf, rf, dt, gmax)
%
% Construct a matched LTE from any STE waveform by eigendecomposition of the
% m-tensor (SPAS framework).  Works for STE waveforms with arbitrary M,
% including non-isotropic designs such as NOW.
%
% The function finds the principal axis system (PAS) of M, then selects the
% sign combination of the three PAS projections that maximises the LTE
% encoding efficiency.  The resulting LTE has the same m/b as the STE.
%
% For STE waveforms with an already isotropic M (e.g. DIME), the direct
% projection dime.waveform.ste2lte is used instead (Eq. 15).
%
%   Inputs
%   ------
%   gwf  - [n x 3] gradient waveform [T/m]
%   rf   - [n x 1] refocusing sign vector (+1/-1)
%   dt   - raster time [s]
%   gmax - maximum gradient amplitude [T/m] (used for efficiency calculation)
%
%   Outputs
%   -------
%   gwf_lte - [n x 3] matched LTE waveform [T/m]

Nt = size(gwf, 1);
M  = (gwf.' * gwf) * dt;   % relative m-tensor (without gamma^2)
m  = trace(M) / 3;

tol    = 1e-6;
is_iso = norm(M - m * eye(3), 'fro') <= tol;

if ~is_iso
    % Eigendecompose M to find principal axis system (SPAS)
    [~, ~, R] = eig(M);
    gpas = gwf * R;

    % Try all four sign combinations of the PAS axes and pick the most efficient
    permsigns = [ 1  1  1;
                 -1  1  1;
                  1 -1  1;
                  1  1 -1];

    bestEff = -Inf;
    best_u  = [1 1 1]';

    for k = 1:size(permsigns, 1)
        tmp = gpas .* permsigns(k, :);
        s   = tmp * [1 1 1]';
        eff = fwf.gwf.toEfficiency([s, zeros(Nt, 2)], rf, dt, gmax);

        if eff > bestEff
            bestEff = eff;
            best_u  = permsigns(k, :)';
        end
    end

    gpas    = gpas .* best_u.';
    gwf_lte = (gpas * [1 1 1]') * [1 0 0];

else
    % M is already isotropic (e.g. DIME): use direct projection (Eq. 15)
    gwf_lte = dime.waveform.ste2lte(gwf);
end
