function [] = simulate_depth_2D()

% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.
clear;
close all;

%Dataset
dsets = cell(0);
dsets{1} = 'Motorcycle';
dsets{2} = 'Adirondack';
dsets{3} = 'Jadeplant';
dsets{4} = 'Piano';
dsets{5} = 'Pipes';
dsets{6} = 'Playroom';
dsets{7} = 'Playtable';
dsets{8} = 'Recycle';
dsets{9} = 'Shelves';
dsets{10} = 'Vintage';
dsets{11} = 'Bicycle1';

%Processing
%plist = [2,3]; %snake-2
%plist = [4,5]; %snake-4
%plist = [6,7]; %snake-5
%plist = [8,9]; %kona-1
plist = [10,11]; %knna-2

%Select dataset
for didx = plist

    %Clear all old data
    clearvars -except dsets didx plist
    close all;

    %Dataset
    dataset = sprintf('%s_falloff', dsets{didx});
    dataset_fn = './middlefield_depth';
    
    %Print
    fprintf('\n###########################################################\n')
    fprintf('#################### Processing [%d/%d] ##################\n', didx, length(dsets))
    fprintf('###########################################################\n')

    % Constants.
    I = 40; %[0.1,1.0,10,80,100,160] % [80, 160] 
    QE = 1;%.30; %Quantum efficiency
    DC = 0.0; %1e-5;  % dark counts per exposure
    amb = 0; %1e-5; % ambient light
    bucket_size = 20e-12;
    pad = [30,10];
    T = 256 + sum(pad(:));
    times = (0:T-1)';
    xt = times;
    %PSF
    xtpsf = (0:100)';
    c = 3e8; % speed of light

    %falloff
    falloff = true;

    %Load
    data = load(sprintf('%s/%s.mat', dataset_fn, dataset));

    %Z scaling
    sc = 0.25;
    fprintf('Z scaling adjustment %2.3f\n', sc)
    data.Z = data.Z*sc;
    data.zlim = data.zlim*sc;
    data.Z = data.Z - data.zlim(1);
    data.zlim = data.zlim - data.zlim(1);

    %Falloff
    falloff = isfield(data, 'falloff') && falloff;
    if falloff
        fprintf('Falloff multiplier adjustment %2.2f\n', 1/sc^2)
        data.falloff = data.falloff * 1/sc^2;
        
        I_scale = 1 / max(data.falloff(:) .* data.A(:));
        I = I * I_scale;
        fprintf('Falloff multiplier adjustment yielding new I = %2.2f.\n', I)
    end

    %Resize
    %{
    figure(), imshow(data.A), title('Amplitude');
    if falloff
        figure(), imagesc(data.falloff), axis image, colorbar, colormap gray, title('Scaled Falloff');
        figure(), imagesc(data.falloff .* data.A .* I_scale), axis image, colorbar, colormap gray, title('Adjusted Falloff Amplitude');
        figure(), imagesc(data.falloff .* data.A), axis image, colorbar, colormap gray, title('Effective Amplitude');
    end
    %}

    %Store falloff in amplitude
    if falloff
        data.A = data.A .* data.falloff;
    end

    %Thresh
    fprintf('Hmin %g, Hmax %g, Zmin %g, Zmax %g\n', 0, bucket_size*times(end- sum(pad(:)) )*c/2*1000, data.zlim(1)*1000, data.zlim(2)*1000);
    data.Z = min(data.Z, bucket_size*times(end - sum(pad(:)) )*c/2);
    %data.A(data.mask) = 0;

    %Offset through pad_dist
    offset_pad = bucket_size*pad(1)*c/2;
    data.Z = data.Z + offset_pad;

    %Show data
    %figure(), imshow(data.A), title('Amplitude');
    %figure(), imagesc(data.Z), axis equal, colormap hsv, colorbar;

    % Make a pulse centered at d.
    d = reshape(data.Z,1,[]); % distance to target in meters

    FWHM_laser = 50e-12; % high noise
    % FWHM_laser = 128e-12 % 128 ps, low noise
    sigma_laser = FWHM_laser/(2*sqrt(2*log(2)));
    FWHM_spad = 50e-12; % for high noise
    % FWHM_spad = 70e-12; % 70 ps, low noise
    sigma_spad = FWHM_spad/(2* sqrt(2*log(2)));
    sigma_true = sqrt(sigma_spad^2 + sigma_laser^2);

    %Load PSF
    psf_model = load('../psf_model.mat');
    gm = psf_model.gm_zeromax;

    %{
    %Gaussian psf
    psf_name = 'GaussianPSF';
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
    psf_name = 'PicoquantBluePSF';
    sigma_true = gm.c1/sqrt(2)*bucket_size;

    %Get pulse
    z = 2*d/c/bucket_size;
    a = reshape(data.A,1,[]); %ones(size(z));
    r = get_parametric_pulse(a, z, xt, gm );
    true_r = r;

    %PSF
    cz = xtpsf(floor(length(xtpsf)/2) + 1); %Center bucket
    psf = get_parametric_pulse(1, cz, xtpsf, gm );

    %Buckets
    verbose = true;
    threads = 32;
    N_trials = 1e5;
    buckets = get_buckets( N_trials, I, true_r(:,:), QE, amb, DC, T, threads, verbose);

    %Save all buckets
    pf = '';
    if falloff
        pf = '_falloff';
    end
    save(sprintf('%s_%s_%3.1f_pad%dx%d%s.mat', dataset, psf_name, I, pad(1), pad(2), pf));

    %Buckets
    fprintf('Done computing %d pixels.\n', size(true_r,2))
    
end

end

%Histogram   
function [ buckets ] = get_buckets(N, I, r, QE, amb, DC, T, threads, verbose)

    lamb = QE*(I*r + amb) + DC;
    buckets = zeros(T,size(r,2));
    
    N_batch = 1000;
    hist = zeros(N_batch, size(r,2));
    
    pp = gcp('nocreate');
    if ~isempty(pp)
        delete(pp);
    end
    pp = parpool('local',threads);
    for to = 1:N_batch:N
        
        %Progress
        if verbose && mod(to - 1, 10000) == 0
            fprintf('Trial [%d/%d]\n', to, N)
        end
        
        N_curr_batch = min(N_batch,N-to+1);
        parfor trial = 1:N_curr_batch
            counts = poissrnd( lamb );
            [sel, min_so_far] = max( counts ~=0, [], 1 );
            hist(trial,:) = min_so_far;
        end

        % Get histogram:
        parfor t = 1:T
            buckets(t,:) = buckets(t,:) + sum(double(hist(1:N_curr_batch,:) == t),1);
        end
    end
    delete(pp);
    
    total_misses = N - sum(buckets,1);
    
    % Verbose
    if verbose
        fprintf('Misses for I=%2.1f are %3.4f%% \n', I, sum(total_misses(:))/(N*size(r,2)) * 100)
    end
end

%Parametric model
function [ pulse ] = get_parametric_pulse(a, z, x, gm )
    %Repmat
    if any(size(a) ~= size(z))
        error('Sizes not matching');
    end
    if size(z,2) > 1
        x = repmat(x, [1,size(z,2)]);
        a = repmat(a, [size(x,1),1]);
        z = repmat(z, [size(x,1),1]);
    end

    pulse = a .* ( gm.a1*exp(-((x-gm.b1-z)./gm.c1).^2) + gm.a2*exp(-((x-gm.b2-z)./gm.c2).^2) + gm.a3*exp(-((x-gm.b3-z)./gm.c3).^2) );
end
