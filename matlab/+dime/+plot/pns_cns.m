function ax = pns_cns(CNS, PNS, varargin)
% function ax = dime.plot.pns_cns(CNS, PNS, Name, Value, ...)
%
% Mirror spider/radar plot for peripheral (PNS) and cardiac (CNS) nerve
% stimulation.  PNS is shown on the left half, CNS on the right.
%
%   Inputs
%   ------
%   CNS, PNS : [1 x 4] vectors — [STE_x, STE_y, STE_z, LTE] as % of limit
%
%   Name-Value pairs (all optional)
%   --------------------------------
%   'RMax'        : radial maximum (default: 1.05 * max([CNS PNS]))
%   'Title'       : string
%   'LeftName'    : left-half label (default 'PNS')
%   'RightName'   : right-half label (default 'CNS')
%   'FontName'    : default 'Times New Roman'
%   'FontSize'    : default 15
%   'LineWidth'   : default 2
%   'FillAlpha'   : default 0.30
%   'AngleRange'  : [top bottom] degrees from top, clockwise (default [15 165])

CNS = CNS(:).';  PNS = PNS(:).';
assert(numel(CNS)==4 && numel(PNS)==4, 'CNS and PNS must be 1x4: [STE_x STE_y STE_z LTE].');

p = inputParser;
addParameter(p,'RMax',      []);
addParameter(p,'Title',     '');
addParameter(p,'LeftName',  'PNS');
addParameter(p,'RightName', 'CNS');
addParameter(p,'FontName',  'Times New Roman');
addParameter(p,'FontSize',  15);
addParameter(p,'LineWidth', 2);
addParameter(p,'FillAlpha', 0.30);
addParameter(p,'FillGray',  0.75);
addParameter(p,'AngleRange',[15 165]);
parse(p, varargin{:});
opt = p.Results;

labels = {'STE x','STE y','STE z','LTE'};

Rmax = opt.RMax;
if isempty(Rmax)
    Rmax = 1.05 * max([CNS PNS]);
    if Rmax <= 0, Rmax = 1; end
end

a      = linspace(opt.AngleRange(1), opt.AngleRange(2), 4);
thetaR = deg2rad(90 - a);
thetaL = pi - thetaR;
polxy  = @(r,th) deal(r.*cos(th), r.*sin(th));

ax = gca; cla(ax); hold(ax,'on');
axis(ax,'equal'); ax.Visible = 'off';
axis(ax, [-1.30 1.30 -1.30 1.30]*Rmax);

t = linspace(0, 2*pi, 600);
plot(ax, Rmax*cos(t), Rmax*sin(t), 'k-', 'LineWidth', 2);
plot(ax, [0 0], [-1.15 1.15]*Rmax, 'k:', 'LineWidth', 2);

for k = 1:4
    [xr,yr] = polxy([0 Rmax], [thetaR(k) thetaR(k)]); plot(ax,xr,yr,'k:','LineWidth',0.9);
    [xl,yl] = polxy([0 Rmax], [thetaL(k) thetaL(k)]); plot(ax,xl,yl,'k:','LineWidth',0.9);
end

fillColor = opt.FillGray * [1 1 1];
rC = [CNS CNS(1)];  rP = [PNS PNS(1)];
[xC,yC] = polxy(rC, [thetaR thetaR(1)]);
[xP,yP] = polxy(rP, [thetaL thetaL(1)]);
patch(ax,xC,yC,fillColor,'FaceAlpha',opt.FillAlpha,'EdgeColor','none');
patch(ax,xP,yP,fillColor,'FaceAlpha',opt.FillAlpha,'EdgeColor','none');
plot(ax,xC,yC,'k-','LineWidth',opt.LineWidth);
plot(ax,xP,yP,'k-','LineWidth',opt.LineWidth);

rtxt = 1.18*Rmax;
for k = 1:4
    [xr,yr] = polxy(rtxt, thetaR(k));
    text(ax,xr,yr,labels{k},'FontName',opt.FontName,'FontSize',opt.FontSize, ...
        'HorizontalAlignment','left','VerticalAlignment','middle');
    [xl,yl] = polxy(rtxt, thetaL(k));
    text(ax,xl,yl,labels{k},'FontName',opt.FontName,'FontSize',opt.FontSize, ...
        'HorizontalAlignment','right','VerticalAlignment','middle');
end

text(ax, 0.55*Rmax, 1.30*Rmax, opt.RightName, 'FontWeight','bold', ...
    'HorizontalAlignment','center','FontName',opt.FontName,'FontSize',opt.FontSize+1);
text(ax,-0.55*Rmax, 1.30*Rmax, opt.LeftName,  'FontWeight','bold', ...
    'HorizontalAlignment','center','FontName',opt.FontName,'FontSize',opt.FontSize+1);

if ~isempty(opt.Title)
    text(ax, 0, -1.32*Rmax, opt.Title, ...
        'HorizontalAlignment','center','FontName',opt.FontName,'FontSize',opt.FontSize+2);
end

hold(ax,'off');
