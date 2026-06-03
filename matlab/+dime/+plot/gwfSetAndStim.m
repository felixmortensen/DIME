function gwfSetAndStim(gwf, rf, dt, hw)
% function dime.plot.gwfSetAndStim(gwf, rf, dt, hw)
%
% Plot STE/LTE waveforms and predicted nerve stimulation (PNS/CNS).
% Requires the fwf toolbox and safe_pns_prediction library.
%
%   Inputs
%   ------
%   gwf - Gradient waveform [n x 3] T/m (STE)
%   rf  - Refocusing vector [n x 1] (+1/-1)
%   dt  - Raster time [s]
%   hw  - SAFE hardware model struct array

lte = dime.waveform.ste2lte(gwf); lte = lte(:,1) .* [1 1 1];
xps = fwf.gwf.toXps(gwf, rf, dt);

h = 4;
w = numel(hw);

subplot(h, 1, 1)
fwf.plot.wf2d(gwf, rf, dt)
title(['STE (gmax = ' num2str(max(abs(gwf(:)*1e3)),'%.1f') ' mT/m) with ' ...
    'b = '          num2str(xps.b/1e9,  '%.2f' ) ' ms/um^2, '...
    'b_{\Delta} = ' num2str(xps.b_delta, '%.3f') ', and ' ...
    'm_{\Delta} = ' num2str(xps.m_delta, '%.3f')]);

subplot(h, 1, 2)
fwf.plot.wf2d(lte, rf, dt)
title(['LTE (gmax = ' num2str(max(abs(gwf(:)*1e3)),'%.1f') ' mT/m)'])

for j = 1:w
    ns_ste = safe_gwf_to_pns(gwf, rf, dt, hw(j), 0);
    ns_lte = safe_gwf_to_pns(lte, rf, dt, hw(j), 0);

    subplot(h, w, 2*w+1 + (j-1))
    safe_plot(ns_ste, dt)
    title([hw(j).model ' stimulation for STE (' num2str(max(vecnorm(ns_ste, 2, 2)),'%.1f') '%)'])
    legend off

    subplot(h, w, 2*w+1 + (j-1)+w)
    safe_plot(ns_lte, dt)
    title([hw(j).model ' stimulation for LTE (' num2str(max(ns_ste(:)),'%.1f') '%)'])
    legend off
end
