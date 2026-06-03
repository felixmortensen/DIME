function fig = con_and_q(files, titles, varargin)
% function fig = dime.plot.con_and_q(files, titles, Name, Value, ...)
%
% Four-panel figure of concomitant gradients g_c(t) and dephasing vector q(t)
% for four waveforms side by side.
%
%   Inputs
%   ------
%   files  : {1 x 4} cell of .mat file paths (each with gwf, rf, dt)
%   titles : {1 x 4} cell of subplot titles
%
%   Name-Value pairs (optional)
%   ---------------------------
%   'ConYLim'  : y-limits for concomitant gradient axis (default [0 1.5] mT/m)
%   'QYLim'    : y-limits for dephasing vector axis     (default [1 3500] rad/m)
%   'WaveXLim' : x-axis limits [ms]                     (default [15 90])
%   'FontSize' : default 15

p = inputParser;
addParameter(p,'ConYLim', [0 1.5], @(x) isnumeric(x) && numel(x)==2);
addParameter(p,'QYLim',   [1 3500],@(x) isnumeric(x) && numel(x)==2);
addParameter(p,'WaveXLim',[15 90], @(x) isnumeric(x) && numel(x)==2);
addParameter(p,'FontSize',15,      @isscalar);
parse(p, varargin{:});
opt = p.Results;

assert(iscell(files)  && numel(files) ==4, 'files must be {1x4}.');
assert(iscell(titles) && numel(titles)==4, 'titles must be {1x4}.');

ncols = 4;
enc = struct('ste',[],'rf',[],'dt',[]);
for c = 1:ncols
    S = load(files{c});
    enc(c).ste = S.gwf(:,:,1);
    enc(c).rf  = S.rf(:);
    enc(c).dt  = S.dt;
end

% Precompute concomitant gradients + q(t)
gwfCon = cell(1,ncols);
t_ms   = cell(1,ncols);
q_t    = cell(1,ncols);
for c = 1:ncols
    rf_lte = enc(c).rf;
    rf_lte(ceil(numel(rf_lte)/2):end) = 1;
    rf_ste = enc(c).rf;
    rf_ste(ceil(numel(rf_ste)/2):end) = -1;
    [gwfCon{c}, ~] = this_gwfCon_and_q(enc(c).ste, rf_lte, enc(c).dt);
    t_ms{c} = fwf.gwf.toTime(gwfCon{c}, rf_lte, enc(c).dt) * 1000;
    q_t{c}  = vecnorm(fwf.gwf.toQt(gwfCon{c}, rf_ste, enc(c).dt), 2, 2);
end

col_xyz = {[0.65 0.65 0.65], [0.3 0.3 0.3], [0.75 0.15 0.15]};
opt_ste = gwf_opt([]);
opt_ste.gwf.col           = col_xyz;
opt_ste.gwf.edge_col      = col_xyz;
opt_ste.gwf.gwf_linestyle = {'-','-','-'};

fig = figure('Color','w');
set(groot,'defaultAxesFontName','Times New Roman', ...
          'defaultAxesFontSize',opt.FontSize, ...
          'defaultTextFontName','Times New Roman', ...
          'defaultTextFontSize',opt.FontSize);

tlo = tiledlayout(1,4,'TileSpacing','compact','Padding','compact');
ax  = gobjects(1,4);

for c = 1:ncols
    ax(c) = nexttile(tlo, c);
    box(ax(c),'on');

    yyaxis(ax(c),'left');
    hcon = gwf_plot(gwfCon{c}, enc(c).rf, enc(c).dt, opt_ste);
    hold(ax(c),'on');
    title(ax(c), titles{c}, 'Interpreter','none','FontWeight','normal');

    if c==1
        yyaxis(ax(c),'left');  ylabel(ax(c),'g_c(t) [mT/m]');
    else
        yyaxis(ax(c),'left');  set(ax(c),'YTickLabel',[]);
    end
    ylim(ax(c), opt.ConYLim);

    yyaxis(ax(c),'right');
    plot(ax(c), t_ms{c}, q_t{c}, 'LineWidth',1.5,'HandleVisibility','off','Color','w');
    q = plot(ax(c), t_ms{c}, q_t{c}, 'k--','LineWidth',1.5,'DisplayName','Dephasing vector');
    ylim(ax(c), opt.QYLim);

    if c==ncols
        yyaxis(ax(c),'right');  ylabel(ax(c),'q(t) [rad/m]');
        yticks(ax(c),[0 1000 2000 3000]);
    else
        yyaxis(ax(c),'right');  set(ax(c),'YTickLabel',[]);
    end

    yyaxis(ax(c),'left');  ax(c).YColor = [0 0 0];
    yyaxis(ax(c),'right'); ax(c).YColor = 'k';
    xlabel(ax(c),'Time [ms]');

    if c==ncols
        legend(ax(c),[hcon(:); q], ...
            {'g_{c,x}','g_{c,y}','g_{c,z}','q(t)'}, ...
            'Location','northwest','Interpreter','tex','Box','off');
    end

    xlim(ax(c), opt.WaveXLim);
    ax(c).TickLength = [0.01 0.01];
    ax(c).LineWidth  = 0.8;
    set(ax(c),'FontSize',opt.FontSize);
end
end


function [gwf_con, q] = this_gwfCon_and_q(gwf, rf, dt)
    gwf_actual = fwf.gwf.toActual(gwf, rf, dt, [1 1 1]*0.1, 3, 1/2, 0);
    gwf_con    = gwf_actual - gwf;
    q          = vecnorm(cumsum(gwf_con.*rf), 2, 2);
end
