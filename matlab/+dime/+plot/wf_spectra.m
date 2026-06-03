function fig = wf_spectra(files, titles, gmax_mTm, varargin)
% function fig = dime.plot.wf_spectra(files, titles, gmax_mTm, Name, Value, ...)
%
% Three-row by four-column figure: STE waveforms, LTE waveforms, and
% encoding power spectra.  Corresponds to Figures 4 and SA in the paper.
%
%   Inputs
%   ------
%   files     : {1 x 4} cell of .mat file paths, each containing gwf [n x 3 x 2]
%               (slice 1 = STE, slice 2 = LTE), rf, dt
%   titles    : {1 x 4} cell of subplot titles, e.g. {'DIME','NOW-OE','NOW-RM','NOW-ET'}
%   gmax_mTm  : maximum gradient amplitude [mT/m], used for y-axis limits
%
%   Name-Value pairs (optional)
%   ---------------------------
%   'WaveXLim'    : x-axis limits for waveform rows (default [15 90] ms)
%   'SpectraXLim' : x-axis limits for spectra row  (default [0 100] Hz)
%   'FontSize'    : default 15
%   'ShowLegends' : default true

p = inputParser;
addParameter(p,'WaveXLim',   [15 90],  @(x) isnumeric(x) && numel(x)==2);
addParameter(p,'SpectraXLim',[0 100],  @(x) isnumeric(x) && numel(x)==2);
addParameter(p,'FontSize',   15,       @isscalar);
addParameter(p,'ShowLegends',true,     @isscalar);
parse(p, varargin{:});
opt = p.Results;

assert(iscell(files)  && numel(files) ==4, 'files must be a {1x4} cell array.');
assert(iscell(titles) && numel(titles)==4, 'titles must be a {1x4} cell array.');

% Load waveforms
ncols = 4;
enc = struct('ste',[],'lte',[],'rf',[],'dt',[]);
for c = 1:ncols
    S = load(files{c});
    enc(c).ste = S.gwf(:,:,1);
    enc(c).lte = S.gwf(:,:,2);
    enc(c).rf  = S.rf(:);
    enc(c).dt  = S.dt;
end

% Waveform plot styles
opt_ste = gwf_opt([]);
opt_ste.gwf.col           = {[0.65 0.65 0.65], [0.3 0.3 0.3], [0.75 0.15 0.15]};
opt_ste.gwf.edge_col      = opt_ste.gwf.col;
opt_ste.gwf.gwf_linestyle = {'-','-','-'};

opt_lte = gwf_opt([]);
opt_lte.gwf.col           = {'#6D94C5','#6D94C5','#6D94C5'};
opt_lte.gwf.edge_col      = {[0 0 0],[0 0 0],[0 0 0]};
opt_lte.gwf.gwf_linestyle = {'--','--','--'};

fig = figure('Color','w');
set(groot,'defaultAxesFontName','Times New Roman', ...
          'defaultAxesFontSize',opt.FontSize, ...
          'defaultTextFontName','Times New Roman', ...
          'defaultTextFontSize',opt.FontSize);

tlo = tiledlayout(3,4,'TileSpacing','compact','Padding','compact');
ax  = gobjects(3,4);

%% Row 1: STE waveforms
for c = 1:ncols
    ax(1,c) = nexttile(tlo, c);
    hold(ax(1,c),'on');  box(ax(1,c),'on');
    title(ax(1,c), titles{c}, 'Interpreter','none','FontWeight','normal');
    yline(ax(1,c), 0, 'k-', 'LineWidth', 0.5);
    gwf_plot(enc(c).ste, enc(c).rf, enc(c).dt, opt_ste);
    xlim(ax(1,c), opt.WaveXLim);
    ylim(ax(1,c), [-gmax_mTm gmax_mTm]);
    set(ax(1,c),'YTick',[-gmax_mTm 0 gmax_mTm]);
    if c==1, ylabel(ax(1,c),'g_{STE}(t) [mT/m]','Interpreter','tex'); end
end

%% Row 2: LTE waveforms
for c = 1:ncols
    ax(2,c) = nexttile(tlo, 4+c);
    hold(ax(2,c),'on');  box(ax(2,c),'on');
    yline(ax(2,c), 0, 'k-', 'LineWidth', 0.5);
    gwf_plot(enc(c).lte, enc(c).rf, enc(c).dt, opt_lte);
    xlim(ax(2,c), opt.WaveXLim);
    ylim(ax(2,c), [-gmax_mTm gmax_mTm]);
    set(ax(2,c),'YTick',[-gmax_mTm 0 gmax_mTm]);
    if c==1, ylabel(ax(2,c),'g_{LTE}(t) [mT/m]','Interpreter','tex'); end
end

%% Row 3: encoding power spectra
for c = 1:ncols
    ax(3,c) = nexttile(tlo, 8+c);
    hold(ax(3,c),'on');  box(ax(3,c),'on');
    axes(ax(3,c));

    chBefore = allchild(ax(3,c));
    gwf_plot_spectra(enc(c).ste, enc(c).rf, enc(c).dt, opt_ste);
    hSTE = flipud(setdiff(allchild(ax(3,c)), chBefore, 'stable'));

    opt2 = opt_lte;  opt2.gwf.ps_scale = 0.65;
    chBefore = allchild(ax(3,c));
    gwf_plot_spectra(enc(c).lte, enc(c).rf, enc(c).dt, opt2);
    hLTE = flipud(setdiff(allchild(ax(3,c)), chBefore, 'stable'));

    xlim(ax(3,c), opt.SpectraXLim);
    xlabel(ax(3,c),'f [Hz]');
    if c==1, ylabel(ax(3,c),'Enc. power [a.u.]'); end

    if opt.ShowLegends && c==ncols
        hLeg = []; labLeg = {};
        tags = {'STE x','STE y','STE z','LTE'};
        for k = 1:min(3,numel(hSTE)), hLeg(end+1)=hSTE(k); labLeg{end+1}=tags{k}; end
        if ~isempty(hLTE),             hLeg(end+1)=hLTE(1); labLeg{end+1}='LTE'; end
        if ~isempty(hLeg)
            legend(ax(3,c), hLeg, labLeg, 'Location','best','Box','off','FontSize',opt.FontSize);
        end
    end
end

%% Cleanup
for c = 2:4
    for r = 1:3
        set(ax(r,c),'YTickLabel',[]); ylabel(ax(r,c),'');
    end
end
for c = 1:4
    set(ax(1,c),'XTickLabel',[]); xlabel(ax(1,c),'');
    set(ax(2,c),'XTickLabel',[]); xlabel(ax(2,c),'');
end
for r = 1:3
    for c = 1:4
        if ~isgraphics(ax(r,c)), continue; end
        ax(r,c).TickLength = [0.01 0.01];
        ax(r,c).LineWidth  = 0.8;
        set(ax(r,c),'FontSize',opt.FontSize);
    end
end
