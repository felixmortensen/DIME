function lte = ste2lte(gwf)
% function lte = dime.waveform.ste2lte(gwf)
%
% Project STE gradient waveform onto the LTE direction u = (-1,+1,+1)/sqrt(3).
% The returned LTE waveform is a single-axis signal scaled to the norm of that
% projection; see Eq. 15 in the DIME paper.

lte = sum(gwf .* [-1 1 1], 2) * [1 0 0];
