function demo(n)
% function dime.optimize.demo
%
% Demonstrates the DIME optimizer for two scanner scenarios:
%   1 - 80 mT/m system  (gmax = 80 mT/m, smax = 200 T/m/s)
%   2 - 200 mT/m system (gmax = 200 mT/m, smax = 200 T/m/s)
%
% Scanner-specific SAFE hardware models (hw) are not distributed with this
% toolbox. Replace safe_example_hw_peripheral below with the hw struct for
% your scanner (see safe_pns_prediction documentation). Set mode = 0 to run
% without stimulation constraints.

if nargin < 1
    n = [1 2];
end

mode = 7;
dt   = 0.1e-3; % s

for i = 1:numel(n)

    clear hw

    switch n(i)
        case 1
            dur  = 40;   % ms
            tp   = 8;
            gmax = 0.08; % T/m
            smax = 200;  % T/m/s

            % Replace with your scanner hw struct, e.g. safe_hw_<yourscanner>
            hw = safe_example_hw_peripheral;

        case 2
            dur  = 28;  % ms
            tp   = 4;
            gmax = 0.2; % T/m
            smax = 200; % T/m/s

            % Replace with your scanner hw struct(s)
            hw = safe_example_hw_peripheral;

    end

    opt           = dime.optimize.options(gmax, smax, dur);

    [gwf, t]      = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);
    [gwf, rf, dt] = dime.waveform.ana2num(gwf, t/1000, dt);

    figure(i)
    clf
    dime.plot.gwfSetAndStim(gwf, rf, dt, hw)

end
