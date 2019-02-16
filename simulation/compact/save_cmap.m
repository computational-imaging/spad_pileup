function [] = save_cmap( cmap_filename, clims, nticks, type )
    %Save colormap in file cmap_filename
    
    cfig=figure;
    left=100; bottom=100 ; width=150 ; height=800;
    pos=[left bottom width height];
    axis off
    if strcmp(type, 'gray')
        colorMap = gray(256);
    elseif  strcmp(type, 'hot')
        colorMap = hot(256);
    elseif  strcmp(type, 'hsv')
        colorMap = hsv(256);
    elseif  strcmp(type, 'prism')
        colorMap = prism(256);
    elseif  strcmp(type, 'colorcube')
        colorMap = colorcube(256);
    elseif  strcmp(jet, 'jet')
        colorMap = jet(256);
    elseif  strcmp(type, 'lines')
        colorMap = lines(256);
    else
        error('CMAP not supported\n');
    end
    
    colormap(colorMap);
    c = colorbar('Position',[0.1 0.1 0.4 0.8],'YAxisLocation','right', 'FontSize', 12);
    caxis(clims)
    %c.Limits = [0 0.5];
    c.Ticks = linspace(clims(1), clims(2), nticks);

    %Set ticklabels
    cticks = cell(0);
    for i = 1:length(c.Ticks)
        cticks{i} = sprintf('%1.2f', c.Ticks(i));
    end
    c.TickLabels = cticks;
    set(cfig,'OuterPosition',pos)

    %Export
    addpath('./exportfig');
    export_fig cmap_filename -png -r800
    close(cfig)

end

