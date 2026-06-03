function [gwf, t, x] = optimize(dur, tp, gmax, smax, mode, hw, opt)
% function [gwf, t, x] = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt)
% By Filip Szczepankiewicz, Lund University
%
%   Designs a 3-axis diffusion-encoding gradient waveform (GWF) that maximizes
%   diffusion weighting (b-value) for a fixed timing budget, while enforcing
%   gradient hardware limits, double isotropic encoding, and (optionally)
%   peripheral nerve stimulation (PNS) limits using SAFE PNS prediction.
%
%   The optimization is performed with FMINCON, optionally wrapped in
%   GlobalSearch or MultiStart (see opt.optMode).
%
%   Inputs
%   ------
%   dur  - Duration [ms] of available encoding time on each side of the
%          refocusing pulse.
%   tp   - Pause [ms] between the end of the first and the start of the second
%          lobe (e.g., crusher/spacing + refocusing pulse).
%   gmax - Maximum gradient amplitude [T/m].
%   smax - Maximum slew rate [T/m/s].
%   mode - Cost/constraint mode for stimulation:
%          0 disables PNS constraints; 1-7 enable different PNS metrics
%          (see gwf2pnsCost for definitions).
%   hw   - SAFE hardware model struct array used for PNS and CNS prediction
%          (see safe_* functions; e.g. safe_example_hw_peripheral).
%          Requires https://github.com/filip-szczepankiewicz/safe_pns_prediction
%   opt  - Options struct for the optimizer and constraint tolerances.
%          Default is created by dime.optimize.options(gmax, smax, dur).
%          opt.optMode = 0, 1, or 2 allows selection between:
%          0: Single optimization with fmincon
%          1: GlobalSearch with fmincon
%          2: MultiStart with fmincon (default)
%
%   Outputs
%   -------
%   gwf  - Optimized gradient waveform [T/m], size [n x 3]. Note that this
%          waveform only includes the optimizer control points. To get a
%          rasterized waveform call dime.waveform.ana2num(gwf, t/1000, dt).
%   t    - Time points for gwf [ms]. These are unique per axis!
%   x    - Optimized parameter vector:
%          x(1) alpha      Fractional amplitude of x/y vs z
%          x(2) beta       Fraction of encoding time dedicated to x/y
%          x(3) rampUpXY   Ramp-up time for x/y gradients
%          x(4) rampDownXY Ramp-down time for x/y gradients
%          x(5) rampUpZ    Ramp-up time for z gradient
%          x(6) rampDownZ  Ramp-down time for z gradient
%          x(7) gampGlo    Global amplitude scale
%
%   Constraints enforced
%   --------------------
%   - b-tensor and m-tensor anisotropy tolerances (opt.tol_bAniso, opt.tol_mAniso)
%   - Optional stimulation constraints: max(PNS) / L2-norm / LTE worst-case variants
%     depending on MODE, with tolerance opt.tol_stim (via SAFE prediction).

if nargin < 1
    dur  = 40; % ms
    tp   = 6;  % ms
    mode = 7;  % stim mode

    gmax = 0.08; % T/m
    smax = 200;  % T/m/s

    opt = dime.optimize.options(gmax, smax, dur);
    hw  = safe_example_hw_peripheral;

    [gwf, t] = dime.optimize.optimize(dur, tp, gmax, smax, mode, hw, opt);

    dt = 0.1e-3; % Raster time of interpolated waveform
    [gwf, rf, dt] = dime.waveform.ana2num(gwf, t/1000, dt);

    clf
    dime.plot.gwfSetAndStim(gwf, rf, dt, hw);
end

if nargin < 7
    opt = dime.optimize.options(gmax, smax, dur);
end

% Check that problem is reasonable
assert(sum(opt.lb([3 3 4 4 5 6]))<dur, 'Shortest ramps do not fit in duration! Dur must be longer!')
assert(sum(opt.ub([3 3 4 4 5 6]))<dur, 'Longest ramps do not fit in duration! Consider extending dur or shortening the max ramp times.')


problem = createOptimProblem('fmincon',...
            'objective', @this_cost,...
            'x0', opt.x0,...
            'lb', opt.lb,...
            'ub', opt.ub,...
            'nonlcon', @this_nlcon,...
            'options', opt.fmc);


switch opt.optMode
    case 0 % Single solve, works in simple setups
        x = fmincon(problem);

    case 1 % Global solver
        x = run(GlobalSearch('NumTrialPoints', opt.numTrialPts, 'NumStageOnePoints', opt.numInitialPts), problem);

    case 2 % Multistart solver
        x = run(MultiStart('UseParallel', true), problem, opt.numStartPts);
end


[gwf, t]    = dime.waveform.par2gwf(x, dur, tp);

% Apply global scale
gwf = gwf*gmax;

    function cost = this_cost(v)
        tmpA  = v(1);
        tmpB  = v(2);
        tmpRU = v(3);
        tmpRD = v(4);
        tmpRZ = v(5);
        tmpRZn= v(6);
        gGlo  = v(7);

        fttt = dur - (2*(tmpRU+tmpRD) + (tmpRZ+tmpRZn));
        fttx = fttt * tmpB/2;
        fttz = fttt * (1 - tmpB);
        bx   = dime.waveform.par2bval(tmpA, tmpRU, tmpRD, fttx,   0, 1);
        by   = bx;
        bz   = dime.waveform.par2bval(   1, tmpRZ, tmpRZn, fttz, tp, 1);
        cost = -(bx + by + bz)*gGlo^2;
    end

    function [C, Ceq] = this_nlcon(v)
        a_tmp   = v(1);
        b_tmp   = v(2);
        ru_tmp  = v(3);
        rd_tmp  = v(4);
        rz_tmp  = v(5);
        rdz_tmp = v(6);

        fttt = dur - (2*(ru_tmp+rd_tmp) + (rz_tmp+rdz_tmp));
        fttx = fttt * b_tmp/2;
        fttz = fttt * (1 - b_tmp);

        bx   = dime.waveform.par2bval(a_tmp, ru_tmp, rd_tmp,  fttx,  0, 1);
        bz   = dime.waveform.par2bval(    1, rz_tmp, rdz_tmp, fttz, tp, 1);
        mx   = dime.waveform.par2mval(a_tmp, ru_tmp, rd_tmp,  fttx);
        mz   = dime.waveform.par2mval(    1, rz_tmp, rdz_tmp, fttz);

        C    = [abs(bx - bz)/mean([bx, bz]) - opt.tol_bAniso;
                abs(mx - mz)/mean([mx, mz]) - opt.tol_mAniso];

        Ceq  = [];

        if mode
            [gwfTmp, tTmp]   = dime.waveform.par2gwf(v, dur, tp);
            [gwfI, rfi, dti] = dime.waveform.ana2num(gwfTmp, tTmp, 0.01);
            gwf_ste          = gwfI*gmax;

            c_pns = [];

            for i = 1:numel(hw)
                tmp = this_gwf2pnsCost(gwf_ste, rfi, dti, hw(i), mode);
                c_pns = [c_pns tmp];
            end

            C = [C; (c_pns'- opt.tol_stim)];
        end

    end
end


% ----- Support functions -----

function c_pns = this_gwf2pnsCost(gwf, rfi, dti, hw, mode)

if mode
    pnsVals_ste = safe_gwf_to_pns(gwf, rfi, dti/1e3, hw, 0);
end

switch mode
    case 0
        c_pns = [];

    case 1
        c_pns = max(pnsVals_ste);

    case 2
        c_pns = max(pnsVals_ste(:));

    case 3
        c_pns = max(vecnorm(pnsVals_ste,2,2));

    case 4
        c_pns = [max(pnsVals_ste) max(vecnorm(pnsVals_ste,2,2))];

    case 5
        lte = dime.waveform.ste2lte(gwf); lte = lte(:,1) * [0 1 0];
        pnsVals_lte = safe_gwf_to_pns(lte, rfi, dti/1e3, hw, 0);
        c_pns = max(pnsVals_lte(:,2));

    case 6
        lte = dime.waveform.ste2lte(gwf); lte = lte(:,1) * [1 1 1];
        pnsVals_lte = safe_gwf_to_pns(lte, rfi, dti/1e3, hw, 0);
        c_pns = [max(pnsVals_ste(:)) max(pnsVals_lte(:))];

    case 7
        lte = dime.waveform.ste2lte(gwf);
        pnsVals_lte = safe_gwf_to_pns(lte, rfi, dti/1e3, hw, 0);
        c_pns = [max(pnsVals_ste(:)) max(pnsVals_lte(:)) max(vecnorm(pnsVals_ste,2,2))];

    otherwise
        error('Stimulation mode not recognized!')
end

end
