%Synthetic data test for "Primal dual cross-channel deconvolution"
function test_psf_fit_data()

    clear;
    close all;
    
    %Experimentfolder 
    exp_folder = '../../data/experiment_17aug2017/wall_scan';

    %Smooth
    smooth = false;
    
    %Load nd-filter files
    nd_filters = [0.001, 1.0];
    transmittance = 1./nd_filters;
    base_time = 200; %ms, also can be 10 for the other data
    acq_time = base_time * transmittance;

    %Load files
    fname = [];
    for ii = 1:length(nd_filters)
        fname{ii} = sprintf('%s/red_wall_scan_sec_%.01f.mat', exp_folder, acq_time(ii)/1000);
    end

    %Load smalles ND-Filter
    load(fname{1})
    
    %Data for low counts
    low_indices = 1:7;
    data_low = [];
    for ii = low_indices
        if isempty(data_low)
            data_low.countsbuffer = double(data{1,ii}(:));
        else
            data_low.countsbuffer = cat(2, data_low.countsbuffer, double(data{1,ii}(:)));
        end
    end
    data_mean_low = zeros(size(data_low.countsbuffer(:,1)));
    for ii = 1:size(data_low.countsbuffer,2)
        data_mean_low = data_mean_low + data_low.countsbuffer(:,ii)/max( data_low.countsbuffer(:,ii) );
    end
    psf = data_mean_low/size(data_low.countsbuffer,2);

    %Correct
    %T = length(psf);
    %N_trials = Tacq/1000 * 5 * 10^6;
    %psf = coates(T, N_trials,  psf);
    psf = psf(:);
    
    
    [rmatched_val, rmatched] = max( log(psf) );
    psf = psf(rmatched-500:rmatched+500);
    psf = psf - min(psf(:));
    
    %Fitting PSF
    if smooth
        
        %Smooth
        psf_smooth = sgolayfilt(psf, 3, 11);

        %Offset
        psf_smooth_offset = min(psf_smooth(:));
        psf_smooth = psf_smooth - psf_smooth_offset;
        psf = psf - psf_smooth_offset;

        % normalize
        psf = squeeze( psf / sum(psf(:)) );
        psf_smooth = squeeze( psf_smooth / sum(psf_smooth(:)) );

        %Plot psf
        figure(),
        hold on;
        plot(psf);
        plot(psf_smooth);
        hold off;
        legend('PSF', 'Smooth PSF')
        title('Temporal PSF of the SPAD')
    
        psf = psf_smooth;

        %Plot psf
        figure(),
        hold on;
        plot(psf);
        hold off;
        title('Fitting PSF')
    else
        
        % normalize
        psf = squeeze( psf / sum(psf(:)) );
        
        %Plot psf
        figure(),
        hold on;
        plot(psf);
        hold off;
        title('Temporal PSF of the SPAD')
    end

    %Mixure model
    xtimes = (0:length(psf(:))-1)';
    mixtures = 'gauss7';
    fitopt = fitoptions(mixtures, 'TolX', 1e-19, 'TolFun', 1e-19, 'MaxIter', 5000, 'MaxFunEvals', 10000);
    gm = fit(xtimes,psf,mixtures, fitopt);
    gm
    
    %Transform
    gma = argnames(gm);
    gmc = [];
    gmc.a = [];
    gmc.b = [];
    gmc.c = [];
    for argi = 1:length(gma)
        arg = gma(argi);
        if strcmp(arg, 'x')
            continue;
        end
        
        %Get coefficient
        ct =  arg{1}(1);
        [idx] = strread(arg{1}(2:end), '%d');
        eval(sprintf('cc = gm.%s;',arg{1}));
        
        %Save result
        if strcmp(ct,'a')
            gmc.a(idx) = cc;
        elseif strcmp(ct,'b')
            gmc.b(idx) = cc;
        elseif strcmp(ct,'c')
            gmc.c(idx) = cc;
        end
        
    end
    
    %Normalize
    %ffitr = gmix(linspace(-100000, 100000, 200000)', gmc );
    %gmc.a = gmc.a/sum(ffitr(:));
    
    %Plot and find peak
    xtimesint = linspace(xtimes(1), xtimes(end), 10000)';
    ffit = gmix(xtimesint, gmc, 0 );
    
    %Peak
    [mval, midx] = max( ffit(:) );
    maxtime = xtimesint(midx);

    figure(),
    hold on;
    plot(xtimes, psf(:), 'ob');
    plot(xtimesint, ffit, '-r');
    plot(maxtime, mval, '*m');
    hold off;
    title('Finalized fit');
    
    figure(),
    hold on;
    plot(xtimes, psf(:), '-b');
    plot(xtimesint, ffit, '-r');
    plot(maxtime, mval, '*m');
    hold off;
    title('Finalized fit compare');
   
    %Offset gaussian model and plot
    gmc.b = gmc.b - maxtime;
    
    %Save gaussian mixture model
    smooth_ext = '';
    if smooth
        smooth_ext = '_smooth';
    end
    save(sprintf('%s/psf_model%s_avg_nomin.mat',exp_folder, smooth_ext), 'gmc');
    xtimesint_mean = linspace(-300, 300, 1000)';
    ffit_mean = gmix(xtimesint_mean, gmc, 0 );
    
    figure(),
    hold on;
    plot(xtimesint_mean, ffit_mean, '-r');
    hold off;
    grid on;
    title('Centered PSF');
    
    %Fit PSF to the current measurements
    
    xt = linspace(0, 2000, 2000)';
    m = 165.3;
    y = gmix(xt, gmc, m ) + 0.01 * rand(size(xt));
    z_true = m;
    
    figure(),
    hold on;
    plot(xt, y, '-g');
    hold off;
    grid on;
    title('Observation PSF');

    %Try to find the time
    psf = ffit_mean/sum(ffit_mean(:));
    lambda = 1000;
    gamma = 0;
    v = 0;
    z_est = fit_psf(xt, y, psf, gmc, lambda, gamma, v)
    
    %Plot fit
    gmix = @(z) gmix(xt,gmc,z);
            
    figure(),
    hold on;
    plot(xt, gmix(z_est), '-r');
    hold off;
    grid on;
    title('Fitted PSF');

end

function res = gmix(x,gc,z)
    %Result function
    res = zeros(size(x));
    for i = 1:size(gc.a,2)
        res = res + gc.a(i)*exp(-((x-gc.b(i) -z)./gc.c(i)).^2);
    end
end

function z_val = fit_psf(xt, y, psf, gc, lambda, gamma, v)

    %Vectorized objective
    objGrad_f = @(zn) obj_grad_func(zn, xt, y, gc, lambda, gamma, v);

    %Minfunc
    addpath('./minFunc_2012/minFunc')
    addpath('./minFunc_2012/minFunc/compiled')
    addpath('./minFunc_2012/autoDif')

    % Hessian-Free Newton
    options = [];
    options.Method = 'newton';
    options.display = 'iter';
    options.DerivativeCheck = true;
    %options.numDiff = 1;
    options.MaxIter = 100;
    options.MaxFunEvals = 100;

    %Matched filter
    [rmatched_val, rmatched] = max( log(imfilter(y, psf, 'corr', 0)) );
    z0 = xt(rmatched)
    
    z_val = minFunc(objGrad_f, z0, options)
               
    %Function
    %gfit = @(z) lambda/2 * norm( gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
    %            gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
    %            gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) - y, 2 )^2 + gamma/2 * norm( z - v, 2 )^2;
    %zl = fminbnd(gfit,0,xt(end))

end

%Computes fixed objective
function [f_x, g_x, H_x] = obj_grad_func( z, xt, y, gc, lambda, gamma, v )
    
    %Model  
    gmixz = zeros(size(xt));
    for i = 1:size(gc.a,2)
        gmixz = gmixz + gc.a(i)*exp(-((xt-gc.b(i) -z)./gc.c(i)).^2);
    end
               
    %Function
    f_x = lambda/2 * norm( gmixz - y, 2 )^2 + gamma/2 * norm( z - v, 2 )^2;

    %Gradient
    g_x = [];
    if nargout > 1
       
        %Gradient of model
        gmix_dz = zeros(size(xt));
        for i = 1:size(gc.a,2)
            gmix_dz = gmix_dz + gc.a(i)*exp(-((xt-gc.b(i) - z)./gc.c(i)).^2) .* (2*(xt-gc.b(i) - z)/gc.c(i)^2);
        end

        %Gradient
        g_x = lambda * sum( (gmixz - y) .* gmix_dz, 1) + gamma * ( z - v);
        
    end
    
    H_x = [];
    if nargout > 2
        
        %Hessian of model
        gmix_hz = zeros(size(xt));
        for i = 1:size(gc.a,2)
            gmix_hz = gmix_hz + gc.a(i)*exp(-((xt-gc.b(i) - z)./gc.c(i)).^2) .* ( (2*(xt-gc.b(i) - z)/gc.c(i)^2).^2 - 2/gc.c(i)^2 );
        end
        
        %Hessian
        H_x = lambda * sum( gmix_dz .* gmix_dz + (gmixz - y) .* gmix_hz, 1) + gamma;
    end
    
end

%Coates method
function [ r ] = coates(T, N, h)
    r = zeros(T,1);
    r(1) = -log(1-h(1)/N);
    for k = 2:T
        tmp = N - sum(h(1:k - 1));
        % TODO: Unclear what to do here.
        if tmp == 0 || h(k)/tmp == 1
            %error('Coates: tmp == 0 || h(k)/tmp == 1')
            r(k) = 0;
            %r(k) = -log(1-h(k)/N);
        else
            if ~isreal(-log(1-h(k)/tmp))
                error('Coates failed')
            end
            r(k) = -log(1-h(k)/tmp);
        end
    end
end

