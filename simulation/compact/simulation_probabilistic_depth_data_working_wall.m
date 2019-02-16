function [] = simulation_coates()

% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.

clear;
close all;

%Dead time
dead_time = 74.7 * 10^(-9);

%Dataset
nd_filters = [0.001, 1.0];
transmittance = 1./nd_filters;
base_time = 200; %ms, also can be 10 for the other data
acq_time = base_time * transmittance;

%Data
laser_type = 'red';
data_folder = '../../data/experiment_17aug2017/wall_scan';
amb_setting = 0.15;

%Errors
errors = [];
    
%Select apropriate dataset
for data_idx = 34:1:49 %49
    
    %Close all
    close all;
    clearvars -except errors data_folder data_idx laser_type ambient data_folder laser_types dead_time nd_filters transmittance base_time acq_time amb_setting
    
    %Print
    fprintf('\n#### Processing %s ####\n', data_folder)
    
    for ii = 1:length(nd_filters)
        fname{ii} = sprintf('%s/red_wall_scan_sec_%.01f.mat', data_folder, acq_time(ii)/1000);
    end
    %Plotting
    plotpad = [128,256];
    
    %Load smalles ND-Filter
    load(fname{1})

    %Data for low counts
    low_indices = data_idx + (-1:1);
    data_low = [];
    for ii = low_indices
        if isempty(data_low)
            data_low.countsbuffer = double(data{1,ii}(:));
        else
            data_low.countsbuffer = cat(2, data_low.countsbuffer, double(data{1,ii}(:)));
        end
    end
    data_low.Resolution = 4;
    data_mean_low = zeros(size(data_low.countsbuffer(:,1)));
    for ii = 1:size(data_low.countsbuffer,2)
        data_mean_low = data_mean_low + data_low.countsbuffer(:,ii)/max( data_low.countsbuffer(:,ii) );
    end
    data_mean_low = data_mean_low/size(data_low.countsbuffer,2);
    
    %Display
    figure();
    hold on;
    [~,zm] = max(data_low.countsbuffer(:,1));
    xlim([zm - plotpad(1), zm + plotpad(2)])
    for ii = 1:size(data_low.countsbuffer,2)
     %   plot( data_low.countsbuffer(:,ii)/max( data_low.countsbuffer(:,ii) ) )
    end
    plot(data_mean_low , 'm')
    hold off;
    ylabel('Relative count')
    xlabel('Bin #')
    title('Low counts');

    %Potentially use the mean
    data_low.countsbuffer = data_mean_low * max(data_low.countsbuffer(:,2));

    %Crop and shift
    shiftval = 0; %17780; %17780;
    lengthval = 5e4;
    data_low.countsbuffer = circshift( data_low.countsbuffer(1:lengthval), [shiftval,0]);

    %Shift around max
    [rmatched_val, rmatched] = max( log(max(data_low.countsbuffer,0)) );
    crop_indices = rmatched + (-200:300);
    %crop_indices = [];

    %Crop
    if ~isempty(crop_indices)
        data_low.countsbuffer = data_low.countsbuffer(crop_indices);
    end

    %Plot countrate correction
    %{
    cr_meas = linspace(1, 13 * 10^6, 100)/1;
    cr_corr = 1./(1 - cr_meas * dead_time);
    figure();
    plot(cr_meas, cr_corr);
    title('CR correction');
    %}

    % Constants.
    if strcmp(laser_type, 'red')
        QE = 0.34; %Quantum efficiency for red
    elseif strcmp(laser_type, 'blue')
        QE = 0.32; %Quantum efficiency for blue
    end
    DC = 0.00; %1e-5;  % dark counts per exposure
    amb = amb_setting; %1e-5; % ambient light
    bucket_size = double(data_low.Resolution) * 1e-12 ;
    T = length(data_low.countsbuffer);
    times = (0:T-1)';
    xt = times;

    %PSF
    xtpsf = (0:T)';

    % Make a pulse centered at d.
    c = 3e8; % speed of light

    %Laser FWHM assumptions
    FWHM_laser = 50e-12; % high noise
    sigma_laser = FWHM_laser/(2*sqrt(2*log(2)));
    FWHM_spad = 50e-12; % for high noise
    sigma_spad = FWHM_spad/(2* sqrt(2*log(2)));
    sigma_true = sqrt(sigma_spad^2 + sigma_laser^2);

    %Get ground truth
    [rmatched_val, rmatched] = max( log(max(data_low.countsbuffer,0)) );
    d = c*xt(rmatched)*bucket_size/2; % distance to target in meters
    z = (2*d/c/bucket_size); %Center bucket

    %Show
    figure();
    hold on;
    plot( xt, data_low.countsbuffer )
    plot( z, data_low.countsbuffer(round(z + 1)), '*')
    ylabel('Relative count')
    xlabel('Bin #')
    xlim([z - plotpad(1), z + plotpad(2)])
    hold off;
    legend('PSF', 'Gaussian fit PSF')
    title('Laser pulse')

    %Load PSF
    psf_model = load(sprintf('%s/psf_model_avg_nomin.mat',data_folder));
    %psf_model = load(sprintf('%s/psf_model_0.01_nomin.mat',data_folder));
    %psf_model = load(sprintf('%s/psf_model_smooth_testcoates_nomin.mat',data_folder));
    %psf_model = load(sprintf('%s/psf_model_nomin.mat',data_folder));
    gmc = psf_model.gmc;

    %PSF
    a_psf = 1.0;
    cz = xtpsf(floor(length(xtpsf)/2) + 1); %Center bucket
    psf = get_parametric_pulse(a_psf, cz, xtpsf, gmc );

    %Gaussian PSF for fitting
    gmc_gauss = gmc;
    gmc_gauss.a = [1/sqrt(2*pi*(sigma_true/bucket_size)^2)];
    gmc_gauss.b = [0];
    gmc_gauss.c = [sqrt(2)*(sigma_true/bucket_size)];
    psf_gauss = get_parametric_pulse(a_psf, cz, xtpsf, gmc_gauss );

    %Pulse
    figure();
    hold on;
    plot( psf )
    plot( psf_gauss )
    ylabel('Relative count')
    xlabel('Bin #')
    xlim([cz - plotpad(1), cz + plotpad(2)])
    hold off;
    legend('PSF', 'Gaussian fit PSF')
    title('Laser pulse')

    %Data from target light level
    target_ll = length(nd_filters);
    I = 20.0; %120;
    fprintf('##### Running ND filter with density %2.2f #####', nd_filters(target_ll) )
    data_target = load(fname{target_ll}); 
    data_target.countsbuffer = data_target.data{1,data_idx};
    data_target.countsbuffer = double(data_target.countsbuffer(:));
    
    %Offset
    scale_fact = max(double(data_target.data{1,data_idx}))/max(double(data_target.data{1,1}));
    amb = amb * scale_fact;

    %Shift and crop
    data_target.countsbuffer = circshift( data_target.countsbuffer(1:lengthval), [shiftval,0]);
    if ~isempty(crop_indices)
        data_target.countsbuffer = data_target.countsbuffer(crop_indices);
    end

    %Intensity
    verbose = true;
    N_trials = round(acq_time(target_ll)/1000 * 5 * 10^6);
    fprintf('\n##### Running for %d trials #####\n', N_trials )

    %Load data
    buckets = data_target.countsbuffer(:);

    %Clean ?
    %buckets = sgolayfilt(buckets, 3, 11);

    %Measurement
    figure()
    hold on;
    plot( xt, buckets/sum(buckets(:)) );
    %set(gca,'Yscale','log');
    ylabel('Relative count')
    xlabel('Bin #')
    title('Measurement')

    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(buckets, bucket_size, psf, T, times);
    %plot( log_mean/bucket_size,log_max, '*')

    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(buckets, bucket_size, sigma_true, T, times);
    gaussmean_nv = gaussmean;
    %plot( gaussmean/bucket_size,gaussint, '*')
    xlim([log_mean/bucket_size - plotpad(1), log_mean/bucket_size + plotpad(2)])
%fprintf('[I = %2.1f], Naive: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    fprintf('[I = %2.1f], Naive LM: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*log_mean/2*1000 - d*1000), abs(log_mean/bucket_size - z))
    errors = cat(2,errors, zeros(6,1));
    errors(1,end) = abs(c*log_mean/2*1000 - d*1000);
    errors(2,end) = abs(log_mean/bucket_size - z);
    %}

    %Apply on coates result
    rc = coates(T, N_trials,  buckets);
    plot(rc/sum(rc(:)), '-m');
    llabel=sprintf('Pile-up correction for I = %2.1f', I);

    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(rc, bucket_size, psf, T, times);
    %plot( log_mean/bucket_size,log_max, '*')

    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(rc, bucket_size, sigma_true, T, times);
    %plot( gaussmean/bucket_size,gaussint, '*')
%fprintf('[I = %2.1f], Coates: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    fprintf('[I = %2.1f], Coates LM: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*log_mean/2*1000 - d*1000), abs(log_mean/bucket_size - z))
    errors(3,end) = abs(c*log_mean/2*1000 - d*1000);
    errors(4,end) = abs(log_mean/bucket_size - z);

    %Compute a and z
    [zv, av] = fit_pileup(buckets, N_trials, xt, gmc, psf, QE, I, amb, DC);
    fprintf('[I = %2.1f], Probabilistic: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv*bucket_size/2*1000 - d*1000), abs(zv - z))
    errors(5,end) = abs(c*zv*bucket_size/2*1000 - d*1000);
    errors(6,end) = abs(zv - z);
    %plot( zv,av, '*m')

    %plot( zv,max(rc(:)/sum(rc(:))), '*m')

    %Aligned
    z_ref = log_mean/bucket_size;
    psf_align = get_parametric_pulse(av, zv, xt, gmc );
    psf_align = psf_align * max(rc(:)/sum(rc(:)))/max(psf_align(:)); %  * 0.98;
    %plot(xt, psf_align, '-m')  

    meas_aligned = data_low.countsbuffer/max(data_low.countsbuffer(:)) * max(rc(:)/sum(rc(:)));
    plot( xt, meas_aligned, '-r' )
    legend('Measurements 1.0',  'Coates-corrected 1.0', 'Measurement 0.01')
    %plot( z,max(rc(:)/sum(rc(:))), '*b') %Target
    hold off;

    %Buckets
    %psf_raw = get_parametric_pulse(av* 0.98, z, xt, gmc );
    %buckets_raw = get_buckets( N_trials, I, psf_raw, QE, amb, DC, T, verbose);

    %{
    buckets_raw = get_buckets( N_trials, 2.1, data_low.countsbuffer, QE, amb, DC, T, verbose);

    figure();
    hold on;
    plot( xt, buckets/sum(buckets(:)), 'b' );
    xlim([log_mean/bucket_size - plotpad(1), log_mean/bucket_size + plotpad(2)])
    plot(xt, psf_align, 'm')  
    plot( xt, buckets_raw/sum(buckets_raw(:)), 'c' )
    ylabel('Relative count')
    xlabel('Bin #')
    title('Measurement')
    hold off;
    %}

    %[zv, av] = fit_pileup_newton(buckets, N_trials, xt, gmc, psf, QE, I, amb, DC);
    %fprintf('[I = %2.1f], Probabilistic Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv*bucket_size/2*1000 - d*1000), abs(zv - z))

end
errors

mean(errors,2)

end

%Parametric model
function [ pulse ] = get_parametric_pulse(a, z, x, gc )
    pulse = zeros(size(x));
    for i = 1:size(gc.a,2)
        pulse = pulse + gc.a(i)*exp(-((x-gc.b(i) -z)./gc.c(i)).^2);
    end
    pulse = a * pulse;
end

%Trials
function [ min_so_far ] = run_trial(counts, T)
    
    min_so_far = inf; % min of T means not detected.
    for t = 1:size(counts,1)
        nt = counts(t);
        if nt > 0
            hits = t;
            if hits < min_so_far
                if 0 < hits && hits <= T
                    min_so_far = hits;
                % Project firing times onto [1,T].
                elseif hits < 1
                    min_so_far = 1;
                else
                    min_so_far = T;
                end
            end
        end
    end
end

%Histogram   
function [ buckets ] = get_buckets(N, I, r, QE, amb, DC, T, verbose)
    lamb = QE*(I*r + amb) + DC;
    hist = zeros(N, 1);
    for trial = 1:N
        counts = poissrnd( lamb(:) );
        min_so_far = run_trial(counts(:), T);
        hist(trial) = min_so_far;
    end
    
    % Get histogram:
    buckets = zeros(T,1);
    for t = 1:T
        buckets(t) = sum(double(hist == t));
    end
    total_misses = N - sum(buckets);
    
    % Verbose
    if verbose
        fprintf('Misses for I=%2.1f are %3.1f%% \n', I, total_misses/N * 100)
    end
end

%Computes model for shift z and amplitude a
function [f_x, g_x, HK_x] = model_func( z, a, xt, gc )
    
    %Model
    psfz = zeros(size(xt));
    for i = 1:size(gc.a,2)
        psfz = psfz + gc.a(i)*exp(-((xt-gc.b(i) -z)./gc.c(i)).^2);
    end  
    f_x = a * psfz;

    %Gradient
    g_x = [];
    HK_x = [];
    if nargout > 1
       
        %Gradient of model
        gmix_dz = zeros(size(xt));
        for i = 1:size(gc.a,2)
            gmix_dz = gmix_dz + gc.a(i)*exp(-((xt-gc.b(i) - z)./gc.c(i)).^2) .* (2*(xt-gc.b(i) - z)/gc.c(i)^2);
        end   
        g_x = cat(2, a * gmix_dz, psfz);
        
        %HK_x
        if nargout > 2
            gmix_ddz = zeros(size(xt));
            for i = 1:size(gc.a,2)
                gmix_ddz = gmix_ddz + gc.a(i)*exp(-((xt-gc.b(i) - z)./gc.c(i)).^2) .* ( (2*(xt-gc.b(i) - z)/gc.c(i)^2).^2 - 2/gc.c(i)^2 );
            end
            HK_x = cat(2, a * gmix_ddz, gmix_dz, gmix_dz, zeros(size(gmix_dz)) );
        end
    
    end
    
end

%Objective function
function [f_x, g_x, H_x] = obj_grad_func( z, a, h, N, xt, gm, QE, I, amb, DC )

    %Get model and gradient
    if nargout > 2
        [lambda_model, lambda_grad, lambda_HK] = model_func( z, a, xt, gm );
    else
        [lambda_model, lambda_grad] = model_func( z, a, xt, gm );
    end
    
    %Scaling and offset
    lambda_model = QE*(I*lambda_model + amb) + DC;
    lambda_grad = QE*I*lambda_grad;
    if nargout > 2
        lambda_HK = QE*I*lambda_HK;
    end
    
    %Numerical issues
    ep = eps();
    lambda_model( abs(lambda_model) < ep) = ep;
    
    %Function value
    lcumsum = cumsum( lambda_model(1:end-1) );
    lcumsum = [0;lcumsum];
    f_x = sum(lambda_model) * (N - sum(h)) + sum(h .* lcumsum) - sum( h.* log( 1-exp(-lambda_model) ) ); 
    
    %Gradient
    g_x = [];
    if nargout > 1
        
        %Gradient w.r.t. lambda
        hcumsum = cumsum( h(end:-1:2) );
        hcumsum = [hcumsum(end:-1:1);0];
        g_lambda = (N - sum(h)) + hcumsum - h.*exp(-lambda_model)./(1 - exp(-lambda_model)); 
    
        %Gradient w.r.t. model params
        g_x = lambda_grad' * g_lambda;
    end
    
    %Hessian
    H_x = [];
    if nargout > 2
        H_lambda_diag = h.*exp(lambda_model)./((exp(lambda_model) - 1).^2); 
        H_x = lambda_grad' * (repmat(H_lambda_diag,[1,2]) .* lambda_grad) + reshape( g_lambda' * lambda_HK, [2,2]);
    end
    
end

function [zv, av] = fit_pileup(h, N, xt, gm, psf, QE, I, amb, DC)

    %Return random sample if 0
    if sum(h) == 0
        zv = rand(1)*(xt(end)+1);
        av = 1;
        return;
    end

    %Vectorized objective
    objGrad_f = @(xn) obj_grad_func( xn(1), xn(2), h, N, xt, gm, QE, I, amb, DC );

    %Minfunc
    addpath('./minFunc_2012/minFunc')
    addpath('./minFunc_2012/minFunc/compiled')
    addpath('./minFunc_2012/autoDif')

    % Hessian-Free Newton
    options = [];
    options.Method = 'lbfgs';
    options.display = 'iter';
    options.DerivativeCheck = false;
    %options.numDiff = 1;
    options.MaxIter = 1000;
    options.MaxFunEvals = 1000;

    %Matched filter init
    if false
        [rmatched_val, rmatched] = max( log(imfilter(h/N, psf, 'corr', 0)) );
        z0 = xt(rmatched);

        %Estimate a
        [z0_model, ~] = model_func( z0, 1.0, xt, gm );
        a0 = (z0_model'*(h/N - DC - QE*amb)/(QE)) /(z0_model'*z0_model);
    else
        T = xt(end)+1;
        rc = coates(T, N,  h);
        [rmatched_val, rmatched] = max( log(imfilter(rc, psf, 'corr', 0)) );
        z0 = xt(rmatched) + 4;
        a0 = rc(rmatched)/max(psf)/I;
    end
    
    %Initial estimate
    x0 = [z0;a0];
    %x_val = minFunc(objGrad_f, x0, options);
    
    options = optimoptions('fmincon','Display','none');
    [x_val,~,~,~,~] = fmincon(objGrad_f,x0,[],[],[],[],[z0-30;0.01],[z0+30;100],[],options);
    
    
    zv = x_val(1);
    av = x_val(2);
    
end


function [zv, av] = fit_pileup_newton(h, N, xt, gm, psf, QE, I, amb, DC)

    %Return random sample if 0
    if sum(h) == 0
        zv = rand(1)*(xt(end)+1);
        av = 1;
        return;
    end

    %Vectorized objective
    objGrad_f = @(xn) obj_grad_func( xn(1), xn(2), h, N, xt, gm, QE, I, amb, DC );

    %Minfunc
    addpath('./minFunc_2012/minFunc')
    addpath('./minFunc_2012/minFunc/compiled')
    addpath('./minFunc_2012/autoDif')

    % Hessian-Free Newton
    options = [];
    options.Method = 'lbfgs';
    options.display = 'final';
    options.DerivativeCheck = false;
    %options.HessianModify = 2;
    %options.cgSolve = 1;
    %options.numDiff = 1;
    %options.progTol = 1e-5;
    options.MaxIter = 50;
    options.MaxFunEvals = 100;

    %Matched filter after coates init
    T = xt(end)+1;
    rc = coates(T, N,  h);    
    [rmatched_val, rmatched] = max( log(imfilter(rc, psf, 'corr', 0)') );
    z0 = xt(rmatched);
    a0 = rc(rmatched)/max(psf)/I;
    
    %Default offset
    z0 = z0 + 3;
    a0 = 0.5;

    %Initial estimate
    x0 = [z0;a0];
    %{
    tic;
    x_val_ = minFunc(objGrad_f, x0, options)
    toc()
    tic;
    %}
    x_val = newton_opt(objGrad_f, x0, 1e-4, 1e-7, 1e-9, 200, 'none');
    %x_val = qnewton_opt(objGrad_f, x0, 1e-4, 1e-8, 1e-9, 200, 'final')
    %toc()
    
    zv = x_val(1);
    av = x_val(2);
    
end

function [x] = newton_opt(func, x0, optTol, stepTol, progTol, maxIter, verbose)

    x = x0;
    t = 0;
    maxStepIters = ceil( log2(1/stepTol) );
    for it = 1:maxIter
        
        [f,g,H] = func(x);
        if strcmp(verbose, 'iter')
            fprintf('Iteration [%3d] Step [%g] Func --> %g \n', it, t, f);
        end
        if norm(g,2) < optTol
            if strcmp(verbose, 'iter') || strcmp(verbose, 'final')
                fprintf('Optimality achieved\n');
            end
            break;
        end

        % Take Newton step if Hessian is pd,
        % otherwise take a step with negative curvature
        [R,posDef] = chol(H);
        if posDef == 0
            d = -R\(R'\g);
        else
            [V,D] = eig((H+H')/2);
            D = diag(D);
            D = max(abs(D),max(max(abs(D)),1)*1e-12);
            d = -V*((V'*g)./D);
        end
        
        % Directional Derivative
        gtd = g'*d;

        % Check that progress can be made along direction
        if gtd > -progTol
            if strcmp(verbose, 'iter') || strcmp(verbose, 'final')
                fprintf('Directional Derivative below progTol\n');
            end
            break;
        end
        
        %Backtrack
        t = 1;
        tr = 0.5;
        for i = 1:maxStepIters
            x_new = x + t * d;
            [fn] = func(x_new);
            if any(~isreal(x_new)) || any(x_new) < eps() || ~isreal(fn) || fn > f
                t = t * tr;
            else
                break;
            end            
        end  
         
        %Do step
        if t < stepTol
          if strcmp(verbose, 'iter') 
            fprintf('Step below tol\n');
          end
          break;
        end
        x = x + t * d;      
        
    end
    if strcmp(verbose, 'final')
        fprintf('Final [%3d] Step [%g] Func --> %g \n', it, t, f);
    end

end

function [x] = qnewton_opt(func, x0, optTol, stepTol, progTol, maxIter, verbose)

    x = x0;
    t = 0;
    maxStepIters = ceil( log2(1/stepTol) );
    for it = 1:maxIter
        
        [f,g] = func(x);
        if strcmp(verbose, 'iter')
            fprintf('Iteration [%3d] Step [%g] Func --> %g \n', it, t, f);
        end
        if norm(g,2) < optTol
            if strcmp(verbose, 'iter') || strcmp(verbose, 'final')
                fprintf('Optimality achieved\n');
            end
            break;
        end

        % Take Newton step if Hessian is pd,
        if it == 1
            d = -g;
        else
            % Compute difference vectors
            y = g-g_old;
            s = t*d;
            
            if it == 2
                % Use Cholesky of Hessian approximation
                R = sqrt((y'*y)/(y'*s))*eye(length(g));
            end
                
            Bs = R'*(R*s);
            if y'*s > 1e-10
                R = cholupdate(cholupdate(R,y/sqrt(y'*s)),Bs/sqrt(s'*Bs),'-');
            else
                if strcmp(verbose,'iter')
                    fprintf('Skipping Update\n');
                end
            end
            d = -R\(R'\g);
        end
        g_old = g;
        
        % Directional Derivative
        gtd = g'*d;

        % Check that progress can be made along direction
        if gtd > -progTol
            if strcmp(verbose, 'iter') || strcmp(verbose, 'final')
                fprintf('Directional Derivative below progTol\n');
            end
            break;
        end
        
        %Initial stepsize
        if it == 1
            t = min(1,1/sum(abs(g)));
        else
            t = 1;
        end

        %Backtrack
        for i = 1:maxStepIters
            x_new = x + t * d;
            [fn,~,~] = func(x_new);
            if any(~isreal(x_new)) || any(x_new) < eps() || ~isreal(fn) || fn > f
                t = t * 0.5;
            else
                break;
            end            
        end   
         
        %Do step
        if t < stepTol
          if strcmp(verbose, 'iter')
            fprintf('Step below tol\n');
          end
          break;
        end
        x = x + t * d;      
        
    end
    if strcmp(verbose, 'final')
        fprintf('Final [%3d] Step [%g] Func --> %g \n', it, t, f);
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
            error('Coates: tmp == 0 || h(k)/tmp == 1')
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

%Gaussian fit
function [gaus_mean, gauss_int] = fit_gaussian(hist, bucket_size, sigma_true, T, times)
    
    %Return random sample if 0
    if sum(hist) == 0
        gaus_mean = rand(1)*T*bucket_size;
        gauss_int = 0;
        return;
    end
    
    %Normalize
    hist = hist/sum(hist(:));
    SCALE = 1/bucket_size;
    x = (times*bucket_size)*SCALE;
    
    %%% NL FIT
    %{
    % equation for Gaussian distribution
    norm_func = @(p,x) p(1) .* exp(-((x - p(2))/(sqrt(2)*p(3)) ).^2);

    initGuess(1) = max(hist(:));  % amplitude
    initGuess(2) = sum(x.*hist);    % mean centred on 0 degrees
    initGuess(3) = sigma_true*SCALE;      % SD in degrees

    % use nlinfit to fit Gaussian using Least Squares
    [bestfit,resid]=nlinfit(x, hist, norm_func, initGuess);
    gauss_int = bestfit(1);
    gaus_mean = bestfit(2)/SCALE;
    gaus_mean = max(0, min(gaus_mean, (T-1)*bucket_size));
    %}
    
    %%% GAUSSIAN FIT
    f = fit(x,hist,'gauss1');
    gauss_int = f.a1;
    gaus_mean = f.b1/SCALE;
    gaus_mean = max(0, min(gaus_mean, (T-1)*bucket_size));
    
    %{
    figure();
    hold on;
    plot(f, x.',hist.', '.');
    hold off;
    %}
end

%Gaussian fit
function [log_mean, log_max] = log_matched_filter(hist, bucket_size, psf, T, times)
    
    %Return random sample if 0
    if sum(hist) == 0
        log_mean = rand(1)*T*bucket_size;
        log_max = 0;
        return;
    end
    
    %Normalize
    hist = hist/sum(hist(:));
    psf = psf/sum(psf(:));
    
    %Filter
    [rmatched_val, rmatched] = max( log(imfilter(hist, psf, 'corr', 0)) );
    rmatched_time = (rmatched - 1) * bucket_size;
    log_mean = max(0, min(rmatched_time, (T-1)*bucket_size));
    log_max = hist(rmatched);
end

function errors = get_errors(NUM_TRIALS, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c)

    errors = zeros(size(photons,1),1);
    for idx = 1:size(photons,1)
        photon_count = photons(idx);
        
        avg_error = 0;
 
        n_reps = 1;
        verbose = false;
        for rep = 1:n_reps
            I = photon_count;
            buckets = get_buckets( NUM_TRIALS, I, true_r, QE, amb, DC, T, verbose);
            if correct
                hist = coates(T, NUM_TRIALS, buckets);
            else
                hist = buckets;
            end
            
            %Fitting method
            if strcmp(fit_method, 'gaussian-fit')
                [estimate,~] = fit_gaussian(hist, bucket_size, sigma_true, T, times);
            elseif strcmp(fit_method, 'log-matched')
                [estimate,~] = log_matched_filter(hist, bucket_size, psf, T, times);
            elseif strcmp(fit_method, 'probabilistic')
                %[estimate, ~] = fit_pileup(buckets, NUM_TRIALS, times, gm, psf, QE, I, amb, DC
                [estimate, ~] = fit_pileup_newton(buckets, NUM_TRIALS, times, gm, psf, QE, I, amb, DC);
                estimate = estimate * bucket_size;
            end

            d_est = c*estimate/2*1000;
            avg_error = avg_error + abs(d_est - d_true*1000);
        end
        errors(idx) = avg_error/n_reps;
    end
    
end

function err = parallel_error_run(x, N, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c)
    rng(x); %Random seed
    err =  get_errors(N, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c);
end