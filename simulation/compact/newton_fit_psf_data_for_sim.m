%Synthetic data test for "Primal dual cross-channel deconvolution"
function test_psf_fit_data()

    clear;
    close all;
    
    %Experimentfolder 
    %experiment_folder = 'far_red_power_1.75';
    experiment_folder = 'far_red_power_1.76';
    %experiment_folder = 'far_blue_power_3.8';
    
    %Smooth
    smooth = true;
    
    %Load nd-filter files
    nd_filters = [0.01];
    transmittance = 1./nd_filters;
    base_time = 200; %ms, also can be 10 for the other data
    acq_time = base_time * transmittance;

    %Load files
    fname = [];
    for ii = 1:length(nd_filters)
        fname{ii} = sprintf('../../data/experiment_9aug2017/pileup/%s/red_diff_%.01f_sec_%.01f.mat',experiment_folder,nd_filters(ii)*100, acq_time(ii)/1000);
    end

    %Load smalles ND-Filter
    load(fname{1});
    psf = double(countsbuffer);
    psf = psf(:);
    [rmatched_val, rmatched] = max( log(psf) );
    psf = psf(rmatched-200:rmatched+400);
    psf = psf - min(psf(:));
    
    %Fitting PSF
    if smooth
        
        %Smooth
        psf_smooth = sgolayfilt(psf, 17, 21);

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
    mixtures = 'gauss3';
    fitopt = fitoptions(mixtures, 'TolX', 1e-19, 'TolFun', 1e-19, 'MaxIter', 5000, 'MaxFunEvals', 10000);
    gm = fit(xtimes,psf,mixtures, fitopt);
    gm
    
    %Normalize
    gmix = @(x,a1,b1,c1,a2,b2,c2,a3,b3,c3) a1*exp(-((x-b1)./c1).^2) + a2*exp(-((x-b2)./c2).^2) + a3*exp(-((x-b3)./c3).^2);
    ffitr = gmix(linspace(-100000, 100000, 200000)', gm.a1, gm.b1, gm.c1, gm.a2, gm.b2, gm.c2, gm.a3, gm.b3, gm.c3 );
    
    %Transform
    gm.a1 = gm.a1/sum(ffitr(:));
    gm.a2 = gm.a2/sum(ffitr(:));
    gm.a3 = gm.a3/sum(ffitr(:));
    
    %Plot and find peak
    xtimesint = linspace(xtimes(1), xtimes(end), 10000)';
    ffit = gmix(xtimesint, gm.a1, gm.b1, gm.c1, gm.a2, gm.b2, gm.c2, gm.a3, gm.b3, gm.c3 );
    
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
    gm_fit = gm;
    
    %Offset gaussian model and plot
    gm.b1 = gm.b1 - maxtime;
    gm.b2 = gm.b2 - maxtime;
    gm.b3 = gm.b3 - maxtime;
    
    %Scale and offset
    sc = 5*2; %4.5;
    gm.b1 = gm.b1/sc;
    gm.c1 = gm.c1/sc;
    gm.b2 = gm.b2/sc;
    gm.c2 = gm.c2/sc;
    gm.b3 = gm.b3/sc;
    gm.c3 = gm.c3/sc;

    asc = 7;
    gm.a1 = gm.a1*asc;
    gm.a2 = gm.a2*asc;
    gm.a3 = gm.a3*asc;

    %Save gaussian mixture model
    gm_zeromax = gm;

    %Save gaussian mixture model
    save(sprintf('./psf_model_%s_20ps.mat', experiment_folder), 'gm_zeromax');
       xtimesint_mean = linspace(-30, 30, 1000)';
    ffit_mean = gmix(xtimesint_mean, gm.a1, gm.b1, gm.c1, gm.a2, gm.b2, gm.c2, gm.a3, gm.b3, gm.c3 );
    
    figure(),
    hold on;
    plot(xtimesint_mean, ffit_mean, '-r');
    hold off;
    grid on;
    title('Centered PSF');
    
    %Fit PSF to the current measurements
    
    xt = linspace(0, 500, 500)';
    m = 165.3;
    y = gmix(xt, gm.a1, gm.b1 + m, gm.c1, gm.a2, gm.b2 + m, gm.c2, gm.a3, gm.b3 + m, gm.c3 ) + 0.1 * rand(size(xt));
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
    z_est = fit_psf(xt, y, psf, gm, lambda, gamma, v)
    
    %Plot fit
    gmix = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
                gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
                gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2);
            
    figure(),
    hold on;
    plot(xt, gmix(z_est), '-r');
    hold off;
    grid on;
    title('Fitted PSF');

end

function z_val = fit_psf(xt, y, psf, gm, lambda, gamma, v)

    %Vectorized objective
    objGrad_f = @(zn) obj_grad_func(zn, xt, y, gm, lambda, gamma, v);

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
    
    z_val_ = newton_opt(objGrad_f, z0, 1e-5, 1e-4, 100, 1)
               
    %Function
    %gfit = @(z) lambda/2 * norm( gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
    %            gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
    %            gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) - y, 2 )^2 + gamma/2 * norm( z - v, 2 )^2;
    %zl = fminbnd(gfit,0,xt(end))

end

function [x] = newton_opt(func, x0, optTol, stepTol, maxIter, verbose)

    x = x0;
    for it = 1:maxIter
      [f,g,H] = func(x);
      if verbose
        fprintf('Iteration [%3d] Func --> %g \n', it, f);
      end
      if norm(g,2) < optTol
        if verbose
            fprintf('Optimality achieved\n');
        end
        break;
      end
      
      %Descent
      t = 1;
      while t > stepTol && func(x - t * g / H) > f
        t = t * 0.5;
      end
      if t < stepTol            
          if verbose
            fprintf('Step below tol\n');
          end
          break;
      end
            
      x = x - t * g / H;      
    end

end

%Computes fixed objective
function [f_x, g_x, H_x] = obj_grad_func( z, xt, y, gm, lambda, gamma, v )
    
    %Model
    gmix = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
                gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
                gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2);
               
    %Function
    f_x = lambda/2 * norm( gmix(z) - y, 2 )^2 + gamma/2 * norm( z - v, 2 )^2;

    %Gradient
    g_x = [];
    if nargout > 1
       
        %Gradient of model
        gmix_d = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* (2*(xt-gm.b1 - z)/gm.c1^2) + ...
                      gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* (2*(xt-gm.b2 - z)/gm.c2^2) + ...
                      gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* (2*(xt-gm.b3 - z)/gm.c3^2);
               
        %Gradient
        g_x = lambda * sum( (gmix(z) - y) .* gmix_d(z), 1) + gamma * ( z - v);
        
    end
    
    H_x = [];
    if nargout > 2
        
        %Hessian of model
        gmix_h = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* ( (2*(xt-gm.b1 - z)/gm.c1^2).^2 - 2/gm.c1^2 ) + ...
                      gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* ( (2*(xt-gm.b2 - z)/gm.c2^2).^2 - 2/gm.c2^2 ) + ...
                      gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* ( (2*(xt-gm.b3 - z)/gm.c3^2).^2 - 2/gm.c3^2 );
        
        %Hessian
        H_x = lambda * sum( gmix_d(z) .* gmix_d(z) + (gmix(z) - y) .* gmix_h(z), 1) + gamma;
    end
    
end

