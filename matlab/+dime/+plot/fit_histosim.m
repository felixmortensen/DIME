function fig = fit_histosim(fits_200, fits_80, wf_names)
% function fig = dime.plot.fit_histosim(fits_200, fits_80, wf_names)
%
% Scatter plot of MD_STE vs MD_LTE across histology-derived substrates, with
% histograms of DeltaMD = MD_LTE - MD_STE per waveform.  Figure 9 in paper.
%
%   Inputs
%   ------
%   fits_200  : output struct from dime.analysis.fitSignals for the 200 mT/m system
%   fits_80   : output struct from dime.analysis.fitSignals for the  80 mT/m system
%   wf_names  : {1 x 4} cell of waveform field names as they appear in the
%               fits structs, e.g. {'DIME','NOW_OE','NOW_RM','NOW_ET'}
%               (use matlab.lang.makeValidName conventions)
%
%   Output
%   ------
%   fig : figure handle

if nargin < 3
    wf_names = fieldnames(fits_200);
end
assert(numel(wf_names)==4, 'wf_names must have exactly 4 entries.');

colors = {[0 0 0], [0.66 0.66 0.66], [0.29 0.57 0.89], [0.56 0.09 0.09]};
labels_disp = {'DIME','NOW-OE','NOW-RM','NOW-ET'};

% Collect D values: [substrates_200; substrates_80]
STE = cell(4,1);  LTE = cell(4,1);
for i = 1:4
    n = wf_names{i};
    STE{i} = [fits_200.(n).STE.coeff.D; fits_80.(n).STE.coeff.D];
    LTE{i} = [fits_200.(n).LTE.coeff.D; fits_80.(n).LTE.coeff.D];
end

res = cellfun(@(l,s) l - s, LTE, STE, 'UniformOutput', false);
[mu, se] = cellfun(@(r) deal(mean(r(:)), std(r(:))), res, 'UniformOutput', false);

all_res = vertcat(res{:});
L = 1.05 * max(abs(all_res));
Nb = 17;
centers = linspace(-L, L, Nb);
bw = centers(2) - centers(1);
edges = [centers - bw/2, centers(end) + bw/2];

fig = figure('Color','w');
tl = tiledlayout(4, 2, 'TileSpacing','compact','Padding','compact');

% Left: scatter MD_STE vs MD_LTE
nexttile(tl,[4 1]);
hold on;
hLine = gobjects(4,1);
for i = 1:4
    hLine(i) = scatter(STE{i}(:), LTE{i}(:), 36, ...
        'MarkerFaceColor', colors{i}, ...
        'MarkerEdgeColor', 'black', ...
        'DisplayName', labels_disp{i});
end
uistack(hLine(1),'top');
lims = [min([STE{:}; LTE{:}]) max([STE{:}; LTE{:}])];
plot(lims, lims, 'k--', 'LineWidth', 1);
xlabel('MD_{STE} [\mum^2/ms]','Interpreter','tex');
ylabel('MD_{LTE} [\mum^2/ms]','Interpreter','tex');
legend(hLine, labels_disp, 'Location','NW','Box','off');
set(gca,'FontSize',15,'FontName','Times New Roman');
box off;

% Right: histograms of DeltaMD
ax = gobjects(4,1);
for i = 1:4
    ax(i) = nexttile(tl);
    histogram(res{i}, 'BinEdges',edges, 'FaceColor',colors{i}, 'EdgeColor','k','FaceAlpha',0.75);
    hold on;
    xline(0,'k--','LineWidth',2);
    ylabel('Counts');
    if i==4, xlabel('\DeltaMD [\mum^2/ms]','Interpreter','tex'); end
    text(0.53, 0.85, sprintf('\\DeltaMD = %.2f \\pm %.2f \\mum^2/ms', mu{i}, se{i}), ...
        'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top', ...
        'FontName','Times New Roman','FontSize',13,'Interpreter','tex');
    set(ax(i),'FontSize',15,'FontName','Times New Roman','YColor','none');
    if i < 4, set(ax(i),'XTickLabel',[]); end
    box off;
end
linkaxes(ax,'x');
