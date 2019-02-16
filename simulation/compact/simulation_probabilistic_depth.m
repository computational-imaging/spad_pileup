function [] = simulation_coates()

% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.
% Multiple gaussian components

clear;
close all;

% Constants.
I = 1; % photons/laser pulse
QE = 1;%.30; %Quantum efficiency
DC = 0.0; %1e-5;  % dark counts per exposure
amb = 0; %1e-5; % ambient light
bucket_size = 2e-12;
T = 100 * 3;
times = (0:T-1)';
xt = times;

%PSF
xtpsf = (0:T)';

% Make a pulse centered at d.
c = 3e8; % speed of light
d = 0.015; % distance to target in meters

mu_laser = 2*d/c;
wavelength = 455e-9; % nanometers
FWHM_laser = 50e-12; % high noise
% FWHM_laser = 128e-12 % 128 ps, low noise
sigma_laser = FWHM_laser/(2*sqrt(2*log(2)));
% FWHM = 2 sqrt(2ln(2))sigma
mu_spad = 0;
FWHM_spad = 50e-12; % for high noise
% FWHM_spad = 70e-12; % 70 ps, low noise
sigma_spad = FWHM_spad/(2* sqrt(2*log(2)));
sigma_true = sqrt(sigma_spad^2 + sigma_laser^2);

%Load PSF
psf_model = load('psf_model.mat')
gm = psf_model.gm_zeromax;

%Gaussian psf
%{
gm.a1 = 1/sqrt(2*pi*(sigma_true/bucket_size)^2);
gm.b1 = 0;
gm.c1 = sqrt(2)*(sigma_true/bucket_size);
gm.a2 = 0;
gm.b2 = 0;
gm.c2 = 1;
gm.a3 = 0;
gm.b3 = 0;
gm.c3 = 1;
%}
%sigma_true = gm.c1/sqrt(2)*bucket_size;

%Get pulse
a = 1.0;
z = (2*d/c/bucket_size); %Center bucket
r = get_parametric_pulse(a, z, xt, gm );
true_r = r;

%PSF
cz = xtpsf(floor(length(xtpsf)/2) + 1); %Center bucket
mu_laser_center = cz * bucket_size;
psf = get_parametric_pulse(a, cz, xtpsf, gm );

%Pulse
figure();
hold on;
plot(r)
ylabel('Relative count')
xlabel('Bin #')
title('Laser pulse')

%True signal
figure();
hold on;
legendlabels = cell(0);
plot(xt, true_r/sum(true_r(:)));
llabel = 'True signal';
legendlabels{end + 1} = llabel;
for I = [0.1,1.0,10,80,100,160] % [80, 160] 
    
    lamb = QE*(I*true_r + amb) + DC;
    llabel = sprintf('I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    verbose = true;
    N_trials = 1e5;
    buckets = get_buckets( N_trials, I, true_r, QE, amb, DC, T, verbose);
    plot( xt, buckets/sum(buckets(:)) );
    
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(buckets, bucket_size, psf, T, times);
    plot( log_mean/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(buckets, bucket_size, sigma_true, T, times);
    plot( gaussmean/bucket_size,gaussint, '*')
    fprintf('[I = %2.1f], Naive: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    %}
    
    %Apply on coates result
    rc = coates(T, N_trials,  buckets);
    plot(rc/sum(rc(:)), '--');
    llabel=sprintf('Pile-up correction for I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(rc, bucket_size, psf, T, times);
    plot( log_mean/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(rc, bucket_size, sigma_true, T, times);
    plot( gaussmean/bucket_size,gaussint, '*')
    fprintf('[I = %2.1f], Coates: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    
    %Compute a and z
    [zv, av] = fit_pileup(buckets, N_trials, xt, gm, psf, QE, I, amb, DC);
    fprintf('[I = %2.1f], Probabilistic: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv*bucket_size/2*1000 - d*1000), abs(zv - z))

    [zv, av] = fit_pileup_newton(buckets, N_trials, xt, gm, psf, QE, I, amb, DC);
    fprintf('[I = %2.1f], Probabilistic Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv*bucket_size/2*1000 - d*1000), abs(zv - z))
    
end

%Legend
legend(legendlabels);
ylabel('Relative count');
xlabel('Bin #');
title('Raw histograms');
hold off;

%%%%%%%%%%%%%%
%%%% Plot error, SNR, more trials so smoother
%%%% combine gaussian fitting + pile-up correction
%%%%%%%%%%%%%%

REPS = 32; %128; %128;
fit_method = 'gaussian-fit'; %'log-matched';

raw_list = [];
pc_list = [];
prob_list = [];
photons = logspace(-4, 1, 20)';
num_trial_list = logspace(4, 6, 5)';

photons = [photons;photons(end) * 2];
photons = [photons;photons(end) * 2];
photons = [photons;photons(end) * 2];
photons = [photons;photons(end) * 2];
photons = [photons;photons(end) * 2];
%photons = [photons;photons(end) * 2];
%photons = [photons;photons(end) * 2];
%photons = [photons;photons(end) * 2];

num_trial_list = num_trial_list(1);

for idx = 1:size(num_trial_list,1)
    N = double( int64( num_trial_list(idx) ) );
    
    %Compute true values
    correct = true;
    vals = zeros(REPS,size(photons,1));
    parfor i = 1:REPS  
        fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        w = warning ('off','all');
        vals(i,:) = parallel_error_run(i, N, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c);    
        warning(w);
    end
    pc_errors = sum(vals,1)/REPS;

    %Compute error values
    correct = false;
    parfor i = 1:REPS  
        fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        w = warning ('off','all');
        vals(i,:) = parallel_error_run(i, N, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c);    
        warning(w);
    end
    raw_errors = sum(vals,1)/REPS;
    
    %Compute true values
    correct = false;
    vals = zeros(REPS,size(photons,1));
    parfor i = 1:REPS  
        fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        w = warning ('off','all');
        vals(i,:) = parallel_error_run(i, N, photons, correct, 'probabilistic', bucket_size, d, sigma_true, true_r, psf, gm, T, times, QE, amb, DC, c);    
        warning(w);
    end
    prob_errors = sum(vals,1)/REPS;

    %Concatenate and plot
    raw_list = [raw_list, raw_errors'];
    pc_list = [raw_list, pc_errors'];
    prob_list = [prob_list, prob_errors'];
    
    %Show expected detections
    figure();
    loglog(photons, cat(1,raw_errors,pc_errors, prob_errors));
    xlabel('Expected Photon Detections');
    ylabel('Avg. Error (mm)');
    legend('Raw', 'Pile-Up Correction', 'Probabilistic', 'Location', 'SouthWest');
    title(sprintf('%d trials', N))
    hold off;
    pause(5.0); 
end
    
end

%Parametric model
function [ pulse ] = get_parametric_pulse(a, z, x, gm )
    pulse = a * ( gm.a1*exp(-((x-gm.b1-z)./gm.c1).^2) + gm.a2*exp(-((x-gm.b2-z)./gm.c2).^2) + gm.a3*exp(-((x-gm.b3-z)./gm.c3).^2) );
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
function [f_x, g_x, HK_x] = model_func( z, a, xt, gm )
    
    %Model
    gmix = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
                gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
                gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2);
    psfz = gmix(z);
    f_x = a * psfz;

    %Gradient
    g_x = [];
    HK_x = [];
    if nargout > 1
       
        %Gradient of model
        gmix_d = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* (2*(xt-gm.b1 - z)/gm.c1^2) + ...
                      gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* (2*(xt-gm.b2 - z)/gm.c2^2) + ...
                      gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* (2*(xt-gm.b3 - z)/gm.c3^2);
        gmix_d_z = gmix_d(z);
        g_x = cat(2, a * gmix_d_z, psfz);
        
        %HK_x
        if nargout > 2
            gmix_dd = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* ( (2*(xt-gm.b1 - z)/gm.c1^2).^2 - 2/gm.c1^2 ) + ...
                           gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* ( (2*(xt-gm.b2 - z)/gm.c2^2).^2 - 2/gm.c2^2 ) + ...
                           gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* ( (2*(xt-gm.b3 - z)/gm.c3^2).^2 - 2/gm.c3^2 );
            HK_x = cat(2, a * gmix_dd(z), gmix_d_z, gmix_d_z, zeros(size(gmix_d_z)) );
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
    options.display = 'none';
    options.DerivativeCheck = false;
    %options.numDiff = 1;
    options.MaxIter = 100;
    options.MaxFunEvals = 100;

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
        z0 = xt(rmatched);
        a0 = rc(rmatched)/max(psf)/I;
    end
    
    %Initial estimate
    x0 = [z0;a0];
    x_val = minFunc(objGrad_f, x0, options);
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
            %error('Coates: tmp == 0 || h(k)/tmp == 1')
            r(k) = 0;
        else
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
    
    %{
    figure();
    hold on;
    plot(x.',hist.', '.');
    plot(x.', norm_func(bestfit, x) );
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