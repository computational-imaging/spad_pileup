function [] = simulation_coates()

% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.

clear;
close all;

% Constants.
I = 1; % photons/laser pulse
QE = 1;%.30; %Quantum efficiency
DC = 0; %1e-5;  % dark counts per exposure
amb = 0; %1e-5; % ambient light
bucket_size = 2e-12;
T = 100;
times = (0:T-1)';

% Make a pulse centered at d.
c = 3e8; % speed of light
d = 0.015; % distance to target in meters
center_bucket = (2*d/c/bucket_size);
fprintf( 'Center_bucket %2.2f\n', center_bucket);

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

%Get pulse
r = get_pulse(times, bucket_size, sigma_true, mu_laser, mu_spad);
true_r = r;

%Pulse
figure();
hold on;
plot(r)
ylabel('Relative count')
xlabel('Bin #')
title('Laser pulse')
hold off;

%True signal
figure();
hold on;
legendlabels = cell(0);
plot(true_r/sum(true_r(:)));
llabel = 'True signal';
legendlabels{end + 1} = llabel;
for I = [0.1,1,10]
    
    lamb = QE*(I*true_r + amb) + DC;
    llabel = sprintf('I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    verbose = true;
    buckets = get_buckets( 1e5, I, true_r, QE, amb, DC, T, verbose);
    plot( buckets/sum(buckets(:)) );
    
    %{
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(buckets, bucket_size, r, T, times);
    plot( (log_mean + bucket_size/2)/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(buckets, bucket_size, sigma_true, T, times);
    plot( (gaussmean + bucket_size/2)/bucket_size,gaussint, '*')
    %}
    
    %{
    %Apply on coates result
    rc = coates(T, 1e5,  buckets);
    plot(rc/sum(rc(:))); %, '--');
    llabel=sprintf('Pile-up correction for I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(rc, bucket_size, r, T, times);
    plot( (log_mean + bucket_size/2)/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(rc, bucket_size, sigma_true, T, times);
    plot( (gaussmean + bucket_size/2)/bucket_size,gaussint, '*')
    %}
end

%Legend
legend(legendlabels);
ylabel('Relative count');
xlabel('Bin #');
title('Raw histograms');
hold off;

% Make plot of error vs # photons
photons = logspace(-3, 1, 20)';

%Bucket size vs Coates improvement
bucket_size_list = logspace(log10(2), log10(50), 10)' * 1e-12;
ratio_list = [];
multsigma = sqrt(2);
%{
for idxb = 1:length(bucket_size_list(:))
    
    %Bucket list index
    bucket_size2 = bucket_size_list(idxb);
    fprintf('\n########## Bucket [%2d/%2d] ############\n\n', idxb, length(bucket_size_list(:)) )
    
    %True distance
    true_d = T/2*bucket_size2*c/2;
    pulse = get_pulse(times, bucket_size2, sigma_true*sqrt(2), 2*true_d/c, mu_spad);
    
    %Fitting method
    fit_method = 'gaussian-fit';
    %fit_method = 'log-matched';
    
    %Ratios
    REPS = 500;
    ratios = zeros(REPS,1);
    
    parfor i = 1:REPS  
        w = warning ('off','all');
        if mod(i-1,10) == 0
            fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        end
        ratios(i) = parallel_error_ratio_run(i, 1e4, photons, fit_method, bucket_size2, true_d, sigma_true*multsigma, pulse, T, times, QE, amb, DC, c);
        warning(w);
    end
  
    ratio = median(ratios(:));
    fprintf('\n%g ### %2.2f\n\n', bucket_size2, ratio )
    ratio_list = [ratio_list, ratio];
end
%}

figure();
hold on;
ratio_list = [4.1901    4.9215    4.7103    5.2816    4.6869    4.8465    4.1791    4.4048    4.1687    4.6080];
xlabel('Bin size (ps)')
ylabel('Median of Coates/raw accuracy')
ylim([0,10]);
semilogx(bucket_size_list*1e12, ratio_list)
hold off;
title('Ratio list');

%%%%%%%%%%%%%%
%%% SNR plot
%%%%%%%%%%%%%%
%{
%Error or reps vs #photons
REPS = 25;
fit_method = 'gaussian-fit'; %'log-matched';
trials = 1e5;

%Compute true values
correct = true;
vals = zeros(REPS,size(photons,1));
parfor i = 1:REPS  
    fprintf('Repetitions [%2d/%2d]\n', i, REPS)
    w = warning ('off','all');
    vals(i,:) = parallel_error_run(i, trials, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, T, times, QE, amb, DC, c);    
    warning(w);
end
pc_errors = sum(vals,1)/REPS;

%Compute error values
correct = false;
vals = zeros(REPS,size(photons,1));
parfor i = 1:REPS  
    fprintf('Repetitions [%2d/%2d]\n', i, REPS)
    w = warning ('off','all');
    vals(i,:) = parallel_error_run(i, trials, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, T, times, QE, amb, DC, c);    
    warning(w);
end
raw_errors = sum(vals,1)/REPS;

figure()
semilogx(photons, cat(2, 10*log10(750./raw_errors'), 10*log10(750./pc_errors')) );
xlabel('Photon count');
ylabel('SNR (dB)');
legend('Raw', 'Pile-Up Correction', 'Location', 'NorthWest');
title('Coates correction');
%}

%%%%%%%%%%%%%%
%%%% Plot error, SNR, more trials so smoother
%%%% combine gaussian fitting + pile-up correction
%%%%%%%%%%%%%%

REPS = 128; %128;
fit_method = 'gaussian-fit'; %'log-matched';

raw_list = [];
pc_list = [];
photons = logspace(-4, 1, 20)';
num_trial_list = logspace(4, 6, 5)';

%num_trial_list = num_trial_list(1:2);

for idx = 1:size(num_trial_list,1)
    N = double( int64( num_trial_list(idx) ) );
    
    %Compute true values
    correct = true;
    vals = zeros(REPS,size(photons,1));
    parfor i = 1:REPS  
        fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        w = warning ('off','all');
        vals(i,:) = parallel_error_run(i, N, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, T, times, QE, amb, DC, c);    
        warning(w);
    end
    pc_errors = sum(vals,1)/REPS;

    %Compute error values
    correct = false;
    parfor i = 1:REPS  
        fprintf('Repetitions [%2d/%2d]\n', i, REPS)
        w = warning ('off','all');
        vals(i,:) = parallel_error_run(i, N, photons, correct, fit_method, bucket_size, d, sigma_true, true_r, T, times, QE, amb, DC, c);    
        warning(w);
    end
    raw_errors = sum(vals,1)/REPS;


    %Concatenate and plot
    raw_list = [raw_list, raw_errors'];
    pc_list = [raw_list, pc_errors'];
    
    %Show expected detections
    figure();
    loglog(photons, cat(1,raw_errors,pc_errors));
    xlabel('Expected Photon Detections');
    ylabel('Avg. Error (mm)');
    legend('Raw', 'Pile-Up Correction', 'Location', 'SouthWest');
    title(sprintf('%d trials', N))
    hold off;
    pause(2.0);
    
end
    
end

%Pulse
function [ pulse ] = get_pulse(times, bucket_size, sigma_true, mu_laser, mu_spad)

    tmp = normcdf((times+1)*bucket_size, mu_laser+mu_spad , sigma_true);
    pulse = tmp - normcdf(times*bucket_size, mu_laser+mu_spad , sigma_true);
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
    x = (times*bucket_size + bucket_size/2)*SCALE;
    
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
function [log_mean, log_max] = log_matched_filter(hist, bucket_size, r, T, times)
    
    %Return random sample if 0
    if sum(hist) == 0
        log_mean = rand(1)*T*bucket_size;
        log_max = 0;
        return;
    end
    
    %Normalize
    hist = hist/sum(hist(:));
    r = r/sum(r(:));
    
    %Filter
    [rmatched_val, rmatched] = max( log(imfilter(hist, r, 'corr', 0)) );
    rmatched_time = (rmatched - 1) * bucket_size + bucket_size/2;
    log_mean = max(0, min(rmatched_time, (T-1)*bucket_size));
    log_max = hist(rmatched);
end

function errors = get_errors(NUM_TRIALS, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c)

    errors = zeros(size(photons,1),1);
    for idx = 1:size(photons,1)
        photon_count = photons(idx);
        
        avg_error = 0;
 
        n_reps = 1;
        verbose = false;
        for rep = 1:n_reps
            buckets = get_buckets( NUM_TRIALS, photon_count, true_r, QE, amb, DC, T, verbose);
            if correct
                hist = coates(T, NUM_TRIALS, buckets);
            else
                hist = buckets;
            end
            
            %Fitting method
            if strcmp(fit_method, 'gaussian-fit')
                [estimate,~] = fit_gaussian(hist, bucket_size, sigma_true, T, times);
            elseif strcmp(fit_method, 'log-matched')
                [estimate,~] = log_matched_filter(hist, bucket_size, true_r, T, times);
            end

            d_est = c*estimate/2*1000;
            avg_error = avg_error + abs(d_est - d_true*1000);
        end
        errors(idx) = avg_error/n_reps;
    end
    
end
 
function error_ratio = get_ratio(NUM_TRIALS, photons, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c)

    errors = zeros(2, size(photons,1));
    for idx = 1:size(photons,1)
        photon_count = photons(idx);
        
        verbose = false;
        buckets = get_buckets( NUM_TRIALS, photon_count, true_r, QE, amb, DC, T, verbose);
        correct_state = logical([false, true]);
        for j = 1:length(correct_state)
            correct = correct_state(j);
            if correct
                hist = coates(T, NUM_TRIALS, buckets);
            else
                hist = buckets;
            end
            
            %Fitting method
            if strcmp(fit_method, 'gaussian-fit')
                [estimate,~] = fit_gaussian(hist, bucket_size, sigma_true, T, times);
            elseif strcmp(fit_method, 'log-matched')
                [estimate,~] = log_matched_filter(hist, bucket_size, true_r, T, times);
            end

            d_est = c*estimate/2*1000;
            errors(j,idx) = abs(d_est - d_true*1000);
        end
    end
    
    error_ratio = min( reshape(errors(1,:),[],1) )/min( reshape(errors(2,:),[],1) );
end

function err = parallel_error_run(x, N, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c)
    rng(x); %Random seed
    err =  get_errors(N, photons, correct, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c);
end

function err_ratios = parallel_error_ratio_run(x, N, photons, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c)
    rng(x); %Random seed
    err_ratios = get_ratio(N, photons, fit_method, bucket_size, d_true, sigma_true, true_r, T, times, QE, amb, DC, c);
end