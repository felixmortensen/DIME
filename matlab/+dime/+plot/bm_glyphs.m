function h = bm_glyphs(xps, idx, normalizeEachTensor)
% function h = dime.plot.bm_glyphs(xps, idx, normalizeEachTensor)
%
% Plot b-tensor and m-tensor glyph ellipsoids from an XPS structure (Figure 4).
%
%   Inputs
%   ------
%   xps                  : XPS struct with fields xps.bt and xps.mt ([N x 6])
%                          Column order: [xx yy zz xy xz yz]
%   idx                  : tensor indices to plot (default: 1:min(500,N))
%   normalizeEachTensor  : true = shape/orientation only (default)
%                          false = preserve relative magnitude

if nargin < 2 || isempty(idx),                  idx = 1:min(500, size(xps.bt,1)); end
if nargin < 3 || isempty(normalizeEachTensor),  normalizeEachTensor = true;       end

idx = idx(:)';
assert(isstruct(xps) && isfield(xps,'bt') && isfield(xps,'mt'), ...
    'xps must have fields bt and mt.');
assert(size(xps.bt,2)==6 && size(xps.mt,2)==6, 'xps.bt and xps.mt must be [N x 6].');

B = xps.bt(idx,:);
M = xps.mt(idx,:);

theme.FaceColor  = [0.90 0.18 0.12];
theme.FaceAlpha  = 0.12;
theme.EdgeColor  = 'none';
theme.PointColor = [0 0 0];
theme.PointSize  = 4;

h.fig = figure('Color','w');
set(h.fig,'Position',[100 100 1100 520]);

h.axB = axes(h.fig,'Units','normalized','Position',[0.02 0.03 0.47 0.86]);
h.B   = this_plot_glyphs(B, normalizeEachTensor, theme);

h.axM = axes(h.fig,'Units','normalized','Position',[0.51 0.03 0.47 0.86]);
h.M   = this_plot_glyphs(M, normalizeEachTensor, theme);

annotation(h.fig,'textbox',[0.02 0.90 0.47 0.07],'String','Diffusion sensitivity', ...
    'EdgeColor','none','HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',15);
annotation(h.fig,'textbox',[0.51 0.90 0.47 0.07],'String','Restriction sensitivity', ...
    'EdgeColor','none','HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontWeight','bold','FontSize',15);

drawnow;
h.insetB = this_orientation_inset(h.fig,[0.045 0.73 0.15 0.15], h.axB.View);

set(findall(h.fig,'-property','FontName'),'FontName','Times New Roman');
end


function hSurf = this_plot_glyphs(T6, normalizeEachTensor, theme)
    nT = size(T6,1);
    [Xs,Ys,Zs] = sphere(18);
    P = [Xs(:), Ys(:), Zs(:)];
    hold on;

    globalScale = 0;
    for i = 1:nT
        T  = this_voigt2tensor(T6(i,:));
        ev = eig(0.5*(T+T'));
        globalScale = max(globalScale, max(abs(ev)));
    end
    if globalScale == 0, globalScale = 1; end

    hSurf = gobjects(nT,1);
    for i = 1:nT
        T = this_voigt2tensor(T6(i,:));
        T = 0.5*(T+T');
        [V,D] = eig(T);
        lambda = diag(D);
        if all(abs(lambda) < 1e-12), continue; end
        lambda = max(lambda, 0);
        if normalizeEachTensor
            nz = lambda(lambda>0);
            if isempty(nz), continue; end
            lambda = lambda ./ mean(nz);
        else
            lambda = lambda ./ globalScale;
        end
        radii = sqrt(lambda);
        G = P * diag(radii) * V';
        hSurf(i) = surf(reshape(G(:,1),size(Xs)), reshape(G(:,2),size(Ys)), reshape(G(:,3),size(Zs)), ...
            'FaceColor',theme.FaceColor,'FaceAlpha',theme.FaceAlpha,'EdgeColor',theme.EdgeColor, ...
            'Marker','.','MarkerEdgeColor',theme.PointColor,'MarkerSize',theme.PointSize);
    end
    axis equal tight off; view(3); camzoom(1.25); camlight headlight; lighting gouraud;
end


function T = this_voigt2tensor(t)
    T = [t(1) t(4) t(5); t(4) t(2) t(6); t(5) t(6) t(3)];
end


function hInset = this_orientation_inset(fig, pos, viewAngles)
    hInset = axes(fig,'Units','normalized','Position',pos,'Color','none');
    hold(hInset,'on');
    col = [0 0 0];
    quiver3(hInset,0,0,0,1,0,0,0,'LineWidth',2.4,'MaxHeadSize',0.8,'Color',col);
    quiver3(hInset,0,0,0,0,1,0,0,'LineWidth',2.4,'MaxHeadSize',0.8,'Color',col);
    quiver3(hInset,0,0,0,0,0,1,0,'LineWidth',2.4,'MaxHeadSize',0.8,'Color',col);
    text(hInset,1.23,0,0,'x','FontSize',15,'FontWeight','bold','HorizontalAlignment','center','Color',col);
    text(hInset,0,1.23,0,'y','FontSize',15,'FontWeight','bold','HorizontalAlignment','center','Color',col);
    text(hInset,0,0,1.23,'z','FontSize',15,'FontWeight','bold','HorizontalAlignment','center','Color',col);
    axis(hInset,'equal off');
    xlim(hInset,[-0.2 1.45]); ylim(hInset,[-0.2 1.45]); zlim(hInset,[-0.2 1.45]);
    view(hInset, viewAngles);
end
