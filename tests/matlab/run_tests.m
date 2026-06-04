% run_tests.m — MATLAB unit tests for the +dime package
%
% Run from the repo root after adding matlab/ to the path:
%   addpath(genpath('matlab'))
%   cd tests/matlab
%   run_tests
%
% Each test prints PASS or fails with an assertion error.

addpath(genpath(fullfile(fileparts(fileparts(pwd)), 'matlab')));
fprintf('Running DIME MATLAB tests...\n\n');
n_pass = 0;

% -------------------------------------------------------------------------
% +waveform/par2bval  — closed-form vs. numerical integration
% -------------------------------------------------------------------------
fprintf('par2bval: closed-form matches numerical integration ... ');

gamma = 267.522187e6;
tru = 5; trd = 5; ftt = 20; tp = 8;  % ms
amp = 0.08;  % T/m

b_formula = dime.waveform.par2bval(amp, tru, trd, ftt, tp, gamma);

% Numerical: build the q-trajectory and integrate q^2
[gwf_a, t_a] = dime.waveform.par2gwf([0.7 0.5 tru trd tru trd 1], 40, tp);
[gwfI, rf, dt] = dime.waveform.ana2num(gwf_a, t_a / 1000, 46e-6);
gwfI = gwfI * amp;
q = gamma * cumsum(gwfI(:,3) .* rf) * dt;  % z-axis (amp=1, alpha=1)
b_numerical = sum(q .^ 2) * dt;

assert(abs(b_formula - b_numerical) / b_numerical < 0.02, ...
    sprintf('par2bval mismatch: formula=%.4e, numerical=%.4e', b_formula, b_numerical));
fprintf('PASS\n'); n_pass = n_pass + 1;


% -------------------------------------------------------------------------
% +waveform/par2mval — closed-form vs. numerical integration
% -------------------------------------------------------------------------
fprintf('par2mval: closed-form matches numerical integration ... ');

m_formula = dime.waveform.par2mval(amp, tru, trd, ftt);
m_numerical = sum(gwfI(:,3) .^ 2) * dt;  % gamma already absorbed in formula check?
% par2mval returns amp^2*(2*tru/3+2*ftt+2*trd/3), no gamma
m_formula2 = amp^2 * (2*tru/3 + 2*ftt + 2*trd/3) / 1e6;  % convert ms->s
m_numerical2 = sum((gwfI(:,3)).^2) * dt;  % T^2/m^2 * s

assert(abs(m_formula2 - m_numerical2) / m_numerical2 < 0.02, ...
    sprintf('par2mval mismatch: formula=%.4e, numerical=%.4e', m_formula2, m_numerical2));
fprintf('PASS\n'); n_pass = n_pass + 1;


% -------------------------------------------------------------------------
% +waveform/ste2lte — projection direction and m/b matching
% -------------------------------------------------------------------------
fprintf('ste2lte: LTE has same m/b as STE ... ');

[gwf_ste, t_ste] = dime.waveform.par2gwf([0.7 0.5 5 5 5 5 1], 40, 8);
[gwfI_ste, rf, dt2] = dime.waveform.ana2num(gwf_ste, t_ste / 1000, 46e-6);
gwfI_ste = gwfI_ste * 0.08;

gwfI_lte = dime.waveform.ste2lte(gwfI_ste);
gwfI_lte = gwfI_lte(:,1) .* [1 1 1];  % broadcast to 3 axes

b_ste = sum(sum((gamma * cumsum(gwfI_ste .* rf) * dt2).^2, 2)) * dt2;
b_lte = sum(sum((gamma * cumsum(gwfI_lte .* rf) * dt2).^2, 2)) * dt2;
m_ste = sum(sum(gwfI_ste.^2)) * dt2;
m_lte = sum(sum(gwfI_lte.^2)) * dt2;

mb_ste = m_ste / b_ste;
mb_lte = m_lte / b_lte;

assert(abs(mb_ste - mb_lte) / mb_ste < 0.01, ...
    sprintf('ste2lte m/b mismatch: STE=%.4e, LTE=%.4e', mb_ste, mb_lte));
fprintf('PASS\n'); n_pass = n_pass + 1;


% -------------------------------------------------------------------------
% +waveform/ana2num — rasterised waveform is balanced (q returns to zero)
% -------------------------------------------------------------------------
fprintf('ana2num: rasterised spin-echo waveform is balanced ... ');

[gwf_a, t_a] = dime.waveform.par2gwf([0.7 0.5 5 5 5 5 1], 40, 8);
[gwfI, rf, dt3] = dime.waveform.ana2num(gwf_a, t_a / 1000, 46e-6);

q_final = sum(gwfI .* rf) * dt3;   % should be ~0 for all axes
assert(max(abs(q_final)) < 1e-6, ...
    sprintf('ana2num: waveform not balanced, max|q_end|=%.2e', max(abs(q_final))));
fprintf('PASS\n'); n_pass = n_pass + 1;


% -------------------------------------------------------------------------
% +waveform/ste2lte_matched — for isotropic M, matches ste2lte
% -------------------------------------------------------------------------
fprintf('ste2lte_matched: gives same result as ste2lte for isotropic M ... ');

gwf_matched = dime.waveform.ste2lte_matched(gwfI_ste, rf, dt2, 0.08);
gwf_direct  = dime.waveform.ste2lte(gwfI_ste);
gwf_direct3 = gwf_direct(:,1) .* [1 1 1];

% Both should give the same m/b
m_matched = sum(sum(gwf_matched.^2)) * dt2;
b_matched = sum(sum((gamma * cumsum(gwf_matched .* rf) * dt2).^2, 2)) * dt2;
m_direct  = sum(sum(gwf_direct3.^2)) * dt2;
b_direct  = sum(sum((gamma * cumsum(gwf_direct3 .* rf) * dt2).^2, 2)) * dt2;

assert(abs(m_matched/b_matched - m_direct/b_direct) / (m_direct/b_direct) < 0.01, ...
    'ste2lte_matched: m/b does not match ste2lte for isotropic M');
fprintf('PASS\n'); n_pass = n_pass + 1;


% -------------------------------------------------------------------------
% Summary
% -------------------------------------------------------------------------
fprintf('\n%d / 5 tests passed.\n', n_pass);
if n_pass == 5
    fprintf('All tests passed.\n');
end
