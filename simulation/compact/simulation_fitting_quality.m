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
hold off;

%True signal
figure();
hold on;
legendlabels = cell(0);
plot(xt, true_r/sum(true_r(:)));
llabel = 'True signal';
legendlabels{end + 1} = llabel;
for I = [0.1,1,10]
    
    lamb = QE*(I*true_r + amb) + DC;
    llabel = sprintf('I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    verbose = true;
    buckets = get_buckets( 1e5, I, true_r, QE, amb, DC, T, verbose);
    plot( xt, buckets/sum(buckets(:)) );
    
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(buckets, bucket_size, psf, T, times);
    plot( log_mean/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(buckets, bucket_size, sigma_true, T, times);
    plot( gaussmean/bucket_size,gaussint, '*')
    fprintf('[I = %2.1f], Naive: Error in depth is %3.2f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    %}
    
    %Apply on coates result
    rc = coates(T, 1e5,  buckets);
    plot(rc/sum(rc(:)), '--');
    llabel=sprintf('Pile-up correction for I = %2.1f', I);
    legendlabels{end + 1} = llabel;
    
    %Logmatched filter
    [log_mean, log_max] = log_matched_filter(rc, bucket_size, psf, T, times);
    plot( log_mean/bucket_size,log_max, '*')
    
    %Fit gaussian to distorted histograms
    [gaussmean, gaussint] = fit_gaussian(rc, bucket_size, sigma_true, T, times);
    plot( gaussmean/bucket_size,gaussint, '*')
    fprintf('[I = %2.1f], Coates: Error in depth is %3.2f mm, %2.2f bins\n', I, abs(c*gaussmean/2*1000 - d*1000), abs(gaussmean/bucket_size - z))
    
    %Compute a and z
    %[zv, av] = fit_pileup(z, a, h, N, xt, gm, rc, psf);

end

%Legend
legend(legendlabels);
ylabel('Relative count');
xlabel('Bin #');
title('Raw histograms');
hold off;

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
