function S = fitSignals(data_dir, wf_names, n_rot, do_2d, out_dir)
% function S = dime.analysis.fitSignals(data_dir, wf_names, n_rot, do_2d, out_dir)
%
% Fit dMRI cumulant models to noise-free signals from Histo-uSim substrates.
%
% Each .mat file in data_dir is expected to have fields:
%   noise_free_sig  - [N x 1] signal vector
%   fn_gwf          - path to the gradient waveform .mat file
%
% Each waveform .mat file is expected to have:
%   xps (or xps_2d) - struct with fields b [N x 1] and (optionally) b_delta
%
% Files in data_dir must be ordered cyclically by waveform family (one
% complete set of substrates per family, repeated for each family).
%
%   Inputs
%   ------
%   data_dir  - Path to folder containing substrate signal .mat files.
%   wf_names  - Cell array of waveform family names in cyclic file order,
%               e.g. {'DIME','NOW-OE','NOW-RM','NOW-ET'}.
%   n_rot     - Number of rotations per encoding direction.
%   do_2d     - 0 to use xps, 1 to use xps_2d (default: 0).
%   out_dir   - (Optional) path to save results.  If omitted, not saved.
%
%   Outputs
%   -------
%   S - Struct with fields S.<WavefamName>.<STE|LTE>.coeff.<S0|D|V|S> and
%       S.<WavefamName>.<STE|LTE>.rot containing per-rotation D values.

if nargin < 4, do_2d   = 0;   end
if nargin < 5, out_dir = '';  end

n_wf = numel(wf_names);
encodingNames = {'STE','LTE'};

% Signal model: S = S0*exp(-b*D + 0.5*b^2*V - (1/6)*b^3*Sk)
ft   = fittype('S0*exp(-x*D + 1/2 * x.^2 * V - 1/6 * x.^3 * S)', ...
    'dependent',{'y'}, 'independent',{'x'}, ...
    'coefficients',{'S0','D','V','S'});
opts = fitoptions(ft);
opts.StartPoint   = [1, 0.5, 0, 0];
opts.Lower        = [0, 0, -Inf, -Inf];
opts.Robust       = 'LAR';
opts.MaxFunEvals  = 5e4;
opts.MaxIter      = 5e3;

fnl = this_find_files(data_dir, '*.mat');
n_files = numel(fnl);
assert(mod(n_files, n_wf)==0, 'Files (%d) not a multiple of n_wf (%d).', n_files, n_wf);
n_sub = n_files / n_wf;

m = nan(4,     n_sub, n_wf, 2); % [coeff x substrate x waveform x encoding]
M = nan(n_rot, n_sub, n_wf, 2); % [rot   x substrate x waveform x encoding]

for i = 1:n_files
    fprintf('Substrate %d / %d\n', i, n_files);
    si = floor((i-1)/n_wf) + 1;
    wi = mod(i-1, n_wf) + 1;

    s   = load(fnl{i});
    nfs = s.noise_free_sig(:);
    wf  = load(s.fn_gwf);

    switch do_2d
        case 0, xps = wf.xps;
        case 1, xps = wf.xps_2d;
        otherwise, error('do_2d must be 0 or 1.');
    end

    [indSTE, indLTE, encPresent] = this_encoding_masks(xps, wf, nfs, n_rot);
    encMasks = {indSTE, indLTE};

    for ei = 1:2
        if ~encPresent(ei), continue; end
        encMask = encMasks{ei};

        x_all = xps.b(encMask) * 1e-9;
        y_all = nfs(encMask);

        % Pooled fit
        if numel(x_all) >= 4
            try
                f_pool  = fit(x_all, y_all, ft, opts);
                cv_pool = coeffvalues(f_pool).';
            catch ME
                warning('Pooled fit failed (sub %d, wf %s, enc %s): %s', ...
                    si, wf_names{wi}, encodingNames{ei}, ME.message);
                cv_pool = [NaN; NaN; NaN; NaN];
            end
        else
            cv_pool = [NaN; NaN; NaN; NaN];
        end
        m(:, si, wi, ei) = cv_pool;

        % Per-rotation fits
        nc = numel(x_all) / n_rot;
        XX = reshape(x_all, n_rot, nc);
        YY = reshape(y_all, n_rot, nc);
        Drot = nan(n_rot, 1);
        for k = 1:n_rot
            if numel(XX(k,:)) >= 4
                try
                    tmp     = fit(XX(k,:)', YY(k,:)', ft, opts);
                    cv      = coeffvalues(tmp);
                    Drot(k) = cv(2);
                catch
                    Drot(k) = NaN;
                end
            end
        end
        M(:, si, wi, ei) = Drot;
    end
end

% Pack into output struct
coeffNames = {'S0','D','V','S'};
S = struct();
for wi = 1:n_wf
    wfName = matlab.lang.makeValidName(wf_names{wi});
    for ei = 1:2
        enc = encodingNames{ei};
        for ci = 1:4
            S.(wfName).(enc).coeff.(coeffNames{ci}) = squeeze(m(ci, :, wi, ei));
        end
        S.(wfName).(enc).rot = squeeze(M(:, :, wi, ei));
    end
end

% Optionally save
if ~isempty(out_dir)
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    stamp  = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    outpth = fullfile(out_dir, sprintf('fits_%s_%s.mat', ...
        ternary(do_2d,'2D','3D'), stamp));
    save(outpth, 'S', '-v7.3');
    fprintf('Saved to %s\n', outpth);
end


%% ----- Helpers -----

function fnl = this_find_files(folder, pattern)
d   = dir(fullfile(folder, pattern));
fnl = sort(fullfile(folder, {d.name}));

function [indSTE, indLTE, encPresent] = this_encoding_masks(xps, wf, nfs, n_rot)
N = numel(nfs);
indSTE = false(N,1); indLTE = false(N,1);
encPresent = false(1,2);

if isfield(xps,'b_delta')
    tol = 1e-3;
    indSTE = (abs(xps.b_delta)     < tol);
    indLTE = (abs(xps.b_delta - 1) < tol);
    encPresent = [any(indSTE), any(indLTE)];
    if any(encPresent), return; end
end

nb = isfield(wf,'bval') * numel(wf.bval) + ~isfield(wf,'bval') * 6;
if N == n_rot*nb
    indLTE = true(N,1);
    encPresent = [false, true];
elseif N == 2*n_rot*nb
    block = n_rot*nb;
    indSTE(1:block)     = true;
    indLTE(block+1:end) = true;
    encPresent = [true, true];
else
    indLTE = true(N,1);
    encPresent = [false, true];
    warning('Unable to infer encoding split; assuming LTE-only.');
end

function r = ternary(cond, a, b)
if cond, r=a; else, r=b; end
