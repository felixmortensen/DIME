function spiderplot(data, varargin)
% function dime.plot.spiderplot(data, Name, Value, ...)
%
% Radar/spider chart with optional centre divider and radial tick labels.
%
%   Inputs
%   ------
%   data  : [n x m] numeric matrix
%             n rows = spokes (variables), m cols = data series
%
%   Name-Value pairs (all optional)
%   --------------------------------
%   'Labels'            {n x 1} cell   spoke labels
%   'SeriesNames'       {m x 1} cell   legend entries
%   'Limits'            [n x 2]        [min max] per spoke
%   'Colors'            [m x 3] RGB    fill colours per series
%   'FillAlpha'         scalar         fill transparency (default 0.15)
%   'LineWidth'         scalar         outline width (default 1.5)
%   'GridLevels'        integer        concentric rings (default 5)
%   'Title'             string         figure title
%   'ShowDivider'       logical        vertical dashed divider (default true)
%   'DividerLabels'     {1 x 2} cell   e.g. {'PNS','CNS'}
%   'RadialTickLabels'  numeric vector e.g. [25 50 75 100]

p = inputParser;
addRequired(p,  'data',             @(x) isnumeric(x) && ismatrix(x));
addParameter(p, 'Labels',           {},         @iscell);
addParameter(p, 'SeriesNames',      {},         @iscell);
addParameter(p, 'Limits',           [],         @isnumeric);
addParameter(p, 'Colors',           [],         @isnumeric);
addParameter(p, 'FillAlpha',        0.15,       @(x) isscalar(x) && x>=0 && x<=1);
addParameter(p, 'LineWidth',        1.5,        @isscalar);
addParameter(p, 'GridLevels',       5,          @(x) isscalar(x) && x>=1);
addParameter(p, 'Title',            '',         @(x) ischar(x) || isstring(x));
addParameter(p, 'do_turn',          1,          @isscalar);
addParameter(p, 'markerSize',       0,          @isscalar);
addParameter(p, 'ShowDivider',      true,       @isscalar);
addParameter(p, 'DividerLabels',    {},         @(x) iscell(x) || isstring(x));
addParameter(p, 'RadialTickLabels', [10 50 100],@(x) isnumeric(x) && isvector(x));
parse(p, data, varargin{:});
opt = p.Results;

[n, m] = size(data);

if isempty(opt.Labels),      opt.Labels      = arrayfun(@(k) sprintf('Var %d',    k), 1:n, 'UniformOutput', false); end
if isempty(opt.SeriesNames), opt.SeriesNames = arrayfun(@(k) sprintf('Series %d', k), 1:m, 'UniformOutput', false); end
if isstring(opt.Labels),       opt.Labels       = cellstr(opt.Labels);      end
if isstring(opt.SeriesNames),  opt.SeriesNames  = cellstr(opt.SeriesNames); end
if isstring(opt.DividerLabels),opt.DividerLabels= cellstr(opt.DividerLabels); end

% Limits / normalisation
if isempty(opt.Limits)
    lo = 0;  hi = max(data(:));
else
    assert(isequal(size(opt.Limits), [n 2]), 'Limits must be [%d x 2].', n);
    lo = opt.Limits(:,1);  hi = opt.Limits(:,2);
end
span = hi - lo;  span(span==0) = 1;
norm_data = (data - lo) ./ span;

if isempty(opt.Colors), opt.Colors = lines(m); end

% Spoke angles (top = pi/2, clockwise)
angles = linspace(pi/2, pi/2 - 2*pi, n+1);
if opt.do_turn, angles = angles + (angles(2)-angles(1))/2; end
angles = angles(1:end-1);

ax = axes('Units','normalized','Position',[0.08 0.08 0.84 0.84]);
hold(ax,'on');
axis(ax,'equal','off');
gridColor = [0.82 0.82 0.82];

for g = 1:opt.GridLevels
    r_g = g / opt.GridLevels;
    th  = [angles, angles(1)];
    plot(ax, r_g*cos(th), r_g*sin(th), 'Color', gridColor, 'LineWidth', 1);
end
for k = 1:n
    plot(ax, [0, cos(angles(k))], [0, sin(angles(k))], 'Color', gridColor, 'LineWidth', 1);
end

if opt.ShowDivider
    plot(ax, [0 0], [-1 1], '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 2);
end

% Radial tick labels
hRadial = gobjects(0);
tickVals = opt.RadialTickLabels(:);
lo_tick = lo(1);  hi_tick = hi(1);
tickR = (tickVals - lo_tick) ./ (hi_tick - lo_tick);
keep  = isfinite(tickR) & tickR>=0 & tickR<=1;
tickVals = tickVals(keep);  tickR = tickR(keep);
if opt.do_turn, dividerScale = cos(pi/n); else, dividerScale = 1; end

for i = 1:numel(tickR)
    y_here = tickR(i) * dividerScale;
    if   abs(tickR(i)-1) < 1e-12, va = 'bottom';
    elseif abs(tickR(i)) < 1e-12, va = 'top';
    else,                          va = 'middle';
    end
    hRadial(end+1,1) = text(ax, 0.02, y_here, sprintf('%g%%', tickVals(i)), ...
        'HorizontalAlignment','left','VerticalAlignment',va, ...
        'FontSize',15,'Color',[0.35 0.35 0.35],'BackgroundColor','w','Margin',0.5);
end

hDividerText = gobjects(0);
if opt.ShowDivider && ~isempty(opt.DividerLabels)
    assert(numel(opt.DividerLabels)==2, 'DividerLabels must have exactly 2 entries.');
    hDividerText(1,1) = text(ax,-0.25,1.18,opt.DividerLabels{1}, ...
        'HorizontalAlignment','right','VerticalAlignment','bottom','FontWeight','bold','FontSize',17);
    hDividerText(2,1) = text(ax, 0.25,1.18,opt.DividerLabels{2}, ...
        'HorizontalAlignment','left', 'VerticalAlignment','bottom','FontWeight','bold','FontSize',17);
end

hLabels = gobjects(n,1);
for k = 1:n
    cosA = cos(angles(k));  sinA = sin(angles(k));
    if     abs(cosA) < 0.13, ha = 'center';
    elseif cosA > 0,         ha = 'left';
    else,                    ha = 'right';
    end
    if     abs(sinA) < 0.13, va = 'middle';
    elseif sinA > 0,         va = 'bottom';
    else,                    va = 'top';
    end
    hLabels(k) = text(ax, 1.02*cosA, 1.02*sinA, opt.Labels{k}, ...
        'HorizontalAlignment',ha,'VerticalAlignment',va,'FontSize',15);
end

XV = zeros(n+1, m);  YV = zeros(n+1, m);
for s = 1:m
    r  = norm_data(:,s)';
    th = [angles, angles(1)];  rv = [r, r(1)];
    XV(:,s) = (rv .* cos(th))';  YV(:,s) = (rv .* sin(th))';
end

for s = 1:m
    patch(ax, XV(:,s)', YV(:,s)', opt.Colors(s,:), ...
        'FaceAlpha',opt.FillAlpha,'EdgeColor','none');
end

hLine = gobjects(m,1);
for s = 1:m
    hLine(s) = plot(ax, XV(:,s)', YV(:,s)', 'Color',opt.Colors(s,:),'LineWidth',opt.LineWidth);
    if opt.markerSize
        plot(ax, XV(1:end-1,s)', YV(1:end-1,s)', 'o', ...
            'MarkerFaceColor',opt.Colors(s,:),'MarkerEdgeColor','w','MarkerSize',opt.markerSize);
    end
end

legend(ax, hLine, opt.SeriesNames, ...
    'Location','southoutside','Orientation','horizontal','Box','off','FontSize',15);
if ~isempty(opt.Title)
    title(ax, opt.Title, 'FontSize',15,'FontWeight','bold');
end

drawnow;
xlo=-1.05; xhi=1.05; ylo=-1.05; yhi=1.10;
for k = 1:numel([hLabels(:); hRadial(:); hDividerText(:)])
    h = [hLabels; hRadial; hDividerText];
    ext = get(h(k),'Extent');
    xlo=min(xlo,ext(1)); ylo=min(ylo,ext(2));
    xhi=max(xhi,ext(1)+ext(3)); yhi=max(yhi,ext(2)+ext(4));
end
margin = 0.06 * max(xhi-xlo, yhi-ylo);
axis(ax, [xlo-margin, xhi+margin, ylo-margin, yhi+margin]);

set(gca,'FontName','Times New Roman','FontSize',15);
hold(ax,'off');
