function [] = simulation_depth_2D_prior()

% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.
clear;
close all;
g = gpuDevice(1);
reset(g)

%Dataset
%fn = 'motorcycle_small_GaussianPSF_40.0_pad';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad';
%fn = 'motorcycle_small_GaussianPSF_40.0_pad20x10';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad20x10';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad30x10';

%Falloff
%fn = 'motorcycle_small_falloff_GaussianPSF_11.5_pad30x10_falloff';
%fn = 'motorcycle_small_falloff_PicoquantBluePSF_11.5_pad30x10_falloff';

%Long range
%fn = 'motorcycle_small_falloff_GaussianPSF_184.0_pad30x10_falloff_long';
fn = 'motorcycle_small_falloff_PicoquantBluePSF_184.0_pad30x10_falloff_long';

%Large Dataset
%fn = 'motorcycle_PicoquantBluePSF_40.0_pad_30';

fdata = sprintf('./dataset/middlefield_depth_results/%s.mat',fn);
load(fdata, 'N_trials', 'd', 'QE', 'I', 'amb', 'DC','sigma_true', 'data', 'buckets', 'psf_name', 'gm',  'a', 'z', 'bucket_size', 'c', 'xt', 'xtpsf', 'T', 'times' );
sz = size(data.A);
mask = reshape(data.mask, 1, []);

%Normalize a to be between [0,1] with scale to I
asc = 1/max(max(a(:)),1);
I = 1/asc * I;
a = asc * a;

%Remove beginning and end
buckets(1:3,:) = 0;
buckets(end:-1:end-2,:) = 0;

%Potential subsampling
mask = mask(:)';

%Load PSF
psf_model = load('psf_model.mat');
gm = psf_model.gm_zeromax;

%Gaussian psf
if strcmp(psf_name, 'GaussianPSF')
    
    gm.a1 = 1/sqrt(2*pi*(sigma_true/bucket_size)^2);
    gm.b1 = 0;
    gm.c1 = sqrt(2)*(sigma_true/bucket_size);
    gm.a2 = 0;
    gm.b2 = 0;
    gm.c2 = 1;
    gm.a3 = 0;
    gm.b3 = 0;
    gm.c3 = 1;
    
elseif strcmp(psf_name, 'PicoquantBluePSF')
    sigma_true = gm.c1/sqrt(2)*bucket_size;
end

%Pulse
%{
figure();
imagesc(reshape(a,sz));
axis image, colormap gray, colorbar;
title('Ground-truth amplitude')

figure();
imagesc(reshape(z,sz)*c*bucket_size/2 );
axis image, colormap hsv, colorbar;
title('Ground-truth depth')
%}

%PSF
cz = xtpsf(floor(length(xtpsf)/2) + 1); %Center bucket
psf = get_parametric_pulse(1, cz, xtpsf, gm );
print_rec = 1:round(size(buckets,2)/10):size(buckets,2); %Plot subset of 100

%Display methods
display_methods = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Naive method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Logmatched filter
tt = tic();
[log_mean_naive, log_max_naive] = log_matched_filter(buckets, bucket_size, psf, T, times);

%Fit gaussian to distorted histograms
[gaussmean_naive, gaussint_naive] = fit_gaussian(buckets, bucket_size, sigma_true, T, times, mask);
toc(tt)

for j = print_rec
    fprintf('[I = %2.1f], Naive: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_naive(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_naive(1,j)/bucket_size - z(1,j)))
end

%Display
if display_methods
    figure();
    imagesc(reshape(min(gaussint_naive,1),sz));
    axis image, colormap gray, colorbar;
    title('Gauss-fit amplitude')

    figure();
    imagesc(reshape(c*gaussmean_naive/2,sz) );
    axis image, colormap hsv, colorbar;
    title('Gauss-fit depth')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Apply on coates result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tt = tic();
rc = coates(T, N_trials,  buckets);

%Logmatched filter
[log_mean_rc, log_max_rc] = log_matched_filter(rc, bucket_size, psf, T, times);

%Display
if display_methods
    figure();
    imagesc(reshape(min(log_max_rc,1),sz));
    axis image, colormap gray, colorbar;
    title('Log-matched Coates amplitude')

    figure();
    imagesc(reshape(c*log_mean_rc/2,sz) );
    axis image, colormap hsv, colorbar;
    title('Log-matched Coates depth')
end

%Fit gaussian to distorted histograms
[gaussmean_rc, gaussint_rc] = fit_gaussian(rc, bucket_size, sigma_true, T, times, mask);
toc(tt)

for j = print_rec
    fprintf('[I = %2.1f], Coates: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_rc(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_rc(1,j)/bucket_size - z(1,j)))
end

%Display
if display_methods
    %Display
    figure();
    imagesc(reshape(min(gaussint_rc,1),sz));
    axis image, colormap gray, colorbar;
    title('Gauss-fit Coates amplitude')

    figure();
    imagesc(reshape(c*gaussmean_rc/2,sz) );
    axis image, colormap hsv, colorbar;
    title('Gauss-fit Coates depth')
end
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Probabilistic model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Compute a and z
tt = tic();
gpu_implem =false;
[zv, av] = fit_pileup_newton(buckets, N_trials, xt, gm, psf, QE, I, amb, DC, mask, gpu_implem);
toc(tt);
zv(mask(:)) = 0;
av(mask(:)) = 0;

for j = print_rec
    fprintf('[I = %2.1f], Probabilistic Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv(1,j)*bucket_size/2*1000 - d(1,j)*1000), abs(zv(1,j) - z(1,j)))
end

%Display
if display_methods
    %Display
    figure();
    imagesc(reshape(min(av,1),sz));
    axis image, colormap gray, colorbar;
    title('Probabilistic Newton amplitude')

    figure();
    imagesc(reshape(zv,sz)*c*bucket_size/2 );
    axis image, colormap hsv, colorbar;
    title('Probabilistic Newton depth')
end
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Spatio-Temporal Reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initial iterate
[zvinit, avinit] = naive_coates(buckets, N_trials, xt, psf, mask);
x0 = cat(3, reshape(zvinit, sz), reshape(avinit, sz));

%Algorihtm params
sizeI = size(x0);
lambda_residual = 1000;
lambda_tv_depth = 1.0;
lambda_tv_albedo = 1.0;
verbose = 'all';
max_it = 150;
gpu_implem =false;
warmstart = 1;

%Scaling for z
scz = size(buckets,1) - 1;

%Compute a and z
tt = tic();
[ res ] = admm_linearized(buckets, mask, x0, scz, sizeI, ...
                                  lambda_residual, lambda_tv_depth, lambda_tv_albedo, warmstart, ...
                                  N_trials, xt, gm, QE, I, amb, DC, gpu_implem, ...
                                  max_it, verbose);                           
toc(tt)

%Z and A prior
zprior = reshape(res(:,:,1),1,[]);
aprior = reshape(res(:,:,2),1,[]);

for j = print_rec
    fprintf('[I = %2.1f], Probabilistic Newton TV: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zprior(1,j)*bucket_size/2*1000 - d(1,j)*1000), abs(zprior(1,j) - z(1,j)))
end

%Display
if display_methods
    %Display
    figure();
    imagesc(reshape(min(aprior,1),sz));
    axis image, colormap gray, colorbar;
    title('Probabilistic Newton TV amplitude')

    figure();
    imagesc(reshape(zprior*c*bucket_size/2,sz) );
    axis image, colormap hsv, colorbar;
    title('Probabilistic Newton TV depth')
end

%Save result
fres = sprintf('result_prior_%s.mat',fn);
save(fres, 'zprior', 'aprior', 'zv', 'av', 'log_mean_naive', 'log_max_naive', 'gaussmean_rc', 'gaussint_rc', 'gaussmean_naive', 'gaussint_naive', 'log_mean_rc', 'log_max_rc');


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

%Computes model for shift z and amplitude a
function [f_x, g_x, HK_x] = model_func( z, a, xt, gm )

    %Repmat
    if any(size(a) ~= size(z))
        error('Sizes not matching');
    end
    if size(z,2) > 1
        xt = repmat(xt, [1,size(z,2)]);
        a = repmat(a, [size(xt,1),1]);
        z = repmat(z, [size(xt,1),1]);
    end
    
    %Model
    gmix = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) + ...
                gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) + ...
                gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2);
    psfz = gmix(z);
    f_x = a .* psfz;

    %Gradient
    g_x = [];
    HK_x = [];
    if nargout > 1
       
        %Gradient of model
        gmix_d = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* (2*(xt-gm.b1 - z)/gm.c1^2) + ...
                      gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* (2*(xt-gm.b2 - z)/gm.c2^2) + ...
                      gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* (2*(xt-gm.b3 - z)/gm.c3^2);
        gmix_d_z = gmix_d(z);
        g_x = cat(3, a .* gmix_d_z, psfz);
        
        %HK_x
        if nargout > 2
            gmix_dd = @(z) gm.a1*exp(-((xt-gm.b1 - z)./gm.c1).^2) .* ( (2*(xt-gm.b1 - z)/gm.c1^2).^2 - 2/gm.c1^2 ) + ...
                           gm.a2*exp(-((xt-gm.b2 - z)./gm.c2).^2) .* ( (2*(xt-gm.b2 - z)/gm.c2^2).^2 - 2/gm.c2^2 ) + ...
                           gm.a3*exp(-((xt-gm.b3 - z)./gm.c3).^2) .* ( (2*(xt-gm.b3 - z)/gm.c3^2).^2 - 2/gm.c3^2 );
            HK_x = cat(3, a .* gmix_dd(z), gmix_d_z, gmix_d_z, zeros(size(gmix_d_z)) );
        end
    
    end
    
end

%Objective function
function [f_x, g_x, H_x] = obj_grad_func( z, a, h, N, xt, gm, QE, I, amb, DC )

    %Get model and gradient
    if nargout > 2
        [lambda_model, lambda_grad, lambda_HK] = model_func( z, a, xt, gm );
    elseif nargout > 1
        [lambda_model, lambda_grad] = model_func( z, a, xt, gm );
    else
        [lambda_model] = model_func( z, a, xt, gm );
    end
    
    %Scaling and offset
    lambda_model = QE*(I*lambda_model + amb) + DC;
    if nargout > 1
        lambda_grad = QE*I*lambda_grad;
    end
    if nargout > 2
        lambda_HK = QE*I*lambda_HK;
    end
    
    %Numerical issues
    ep = eps();
    lambda_model( abs(lambda_model) < ep) = ep;
    
    %Function value
    lcumsum = cumsum( lambda_model(1:end-1,:) ,1 );
    lcumsum = cat(1, zeros([1,size(h,2)]), lcumsum);
    f_x = sum(lambda_model,1) .* (N*ones([1,size(h,2)]) - sum(h,1)) + sum(h .* lcumsum,1) - sum( h.* log( 1-exp(-lambda_model) ), 1 ); 
    
    %Gradient
    g_x = [];
    if nargout > 1
        
        %Gradient w.r.t. lambda
        hcumsum = cumsum( h(end:-1:2,:), 1 );
        hcumsum = cat(1, hcumsum(end:-1:1,:), zeros([1,size(h,2)]) );
        g_lambda = repmat(N*ones([1,size(h,2)]) - sum(h,1), [size(h,1),1]) + hcumsum - h.*exp(-lambda_model)./(1 - exp(-lambda_model)); 
    
        %Gradient w.r.t. model params
        g_x = sum( lambda_grad .* repmat(g_lambda,[1,1,2]), 1);
        g_x = reshape( shiftdim(g_x,2), [2, size(h,2)]);
    end
    
    %Hessian
    H_x = [];
    if nargout > 2
        H_lambda_diag = h.*exp(lambda_model)./((exp(lambda_model) - 1).^2); 
        
        %H_x = lambda_grad' * (repmat(H_lambda_diag,[1,1,2]) .* lambda_grad) + reshape( g_lambda' * lambda_HK, [2,2]);
        H_x = zeros(1,size(h,2),2,2);
        if isa(h, 'gpuArray')
            H_x = gpuArray( H_x );
        end
        H_x(:,:,1,1) =  sum( lambda_grad(:,:,1) .* H_lambda_diag .* lambda_grad(:,:,1), 1)  +  sum( g_lambda .* lambda_HK(:,:,1), 1);
        H_x(:,:,2,1) =  sum( lambda_grad(:,:,2) .* H_lambda_diag .* lambda_grad(:,:,1), 1)  +  sum( g_lambda .* lambda_HK(:,:,2), 1);
        H_x(:,:,1,2) =  sum( lambda_grad(:,:,1) .* H_lambda_diag .* lambda_grad(:,:,2), 1)  +  sum( g_lambda .* lambda_HK(:,:,3), 1);
        H_x(:,:,2,2) =  sum( lambda_grad(:,:,2) .* H_lambda_diag .* lambda_grad(:,:,2), 1);
        H_x = reshape( shiftdim(H_x,2), [2,2, size(h,2)]);
    end
    
end

function [zv, av] = fit_pileup_newton(h, N, xt, gm, psf, QE, I, amb, DC, mask, gpu)

    %Return random sample if 0
    nz = size(h,2);
    hist_zeros = (sum(h,1) == 0) | mask;
    N_zeros = sum(hist_zeros);
    h = h(:,~hist_zeros);

    %Vectorized objective
    if ~gpu
        objGrad_f = @(xn) obj_grad_func( xn(1,:), xn(2,:), h, N, xt, gm, QE, I, amb, DC );
    else
        objGrad_f = @(xn,j) obj_grad_func( xn(1,:), xn(2,:), gpuArray(h), gpuArray(N), gpuArray(xt), gm, gpuArray(QE), gpuArray(I), gpuArray(amb), gpuArray(DC) );
    end
    
    %Shifted matched filter after coates init
    T = xt(end)+1;
    rc = coates(T, N,  h);    
    [rmatched_val, rmatched] = max( log(imfilter(rc, psf, 'corr', 0)),[],1);
    z0 = xt(rmatched)';
    %a0 = rc( sub2ind(size(rc),rmatched,1:size(rc,2)) )/max(psf(:))/I;
    z0 = min( z0 + 3, xt(end));
    a0 = 0.5 * ones(size(z0));

    %Initial estimate
    x0 = cat(1,z0,a0);
    x_val = newton_opt(objGrad_f, gpu, x0, 1e-4, 1e-7, 1e-9, 200, 'iter');
    zv = x_val(1,:);
    av = x_val(2,:);
    
    if N_zeros > 0
        %Copy over
        zv_pos = zv;
        av_pos = av;
        zv = zeros(1,nz);
        av = zeros(1,nz);
        zv(:,hist_zeros) = 0;
        av(:,hist_zeros) = 0;
        zv(:,~hist_zeros) = zv_pos;
        av(:,~hist_zeros) = av_pos;
    end
    
end

function [x,it] = newton_opt(func, gpu, x0, optTol, stepTol, progTol, maxIter, verbose)

    x = x0;
    t = 0;
    maxStepIters = ceil( log2(1/stepTol) );
    terminated = zeros(1,size(x0,2));
    for it = 1:maxIter
        
        %Sample function
        if ~gpu
            [f,g,H] = func(x);
        else
            [f,g,H] = func(gpuArray(x));
            g = gather(g);
            H = gather(H);
        end
        if strcmp(verbose, 'iter')
            fprintf('Iteration [%3d] Step [%g] Func --> %g \n', it, mean(t(:)), sum(f(:)) );
        end
        terminated = terminated | (sqrt(sum(g.^2,1)) < optTol);
        if all( terminated ) 
            break;
        end

        % Take Newton step if Hessian is pd,
        % otherwise take a step with negative curvature
        d = zeros(2,size(g,2));
        gtd = zeros(size(g,2),1);
        parfor j = 1:size(g,2)

            [R,posDef] = chol(H(:,:,j));
            if posDef == 0
                d(:,j) = -R\(R'\g(:,j));
            else
                [V,D] = eig((H(:,:,j)+H(:,:,j)')/2);
                D = diag(D);
                D = max(abs(D),max(max(abs(D)),1)*1e-12);
                d(:,j) = -V*((V'*g(:,j))./D);
            end

            % Directional Derivative
            gtd(j) = g(:,j)'*d(:,j);            
        end
        
        % Check that progress can be made along direction
        terminated = terminated | gtd' > -progTol;
        if all(terminated)
            break;
        end

        %Backtrack
        t = ones(1,size(g,2));
        t(terminated) = 0;
        tr = 0.5;
        for i = 1:maxStepIters
            x_new =  x + repmat(t,[2,1]) .* d;
            invalid_idx = ~isreal(x_new(1,:)) | x_new(1,:) < eps() | ~isreal(x_new(2,:)) | x_new(2,:) < eps() ;
            if any(invalid_idx(:))
                t(invalid_idx(:)) = t(invalid_idx(:)) * tr;
                continue;
            end  
            
            %Function query
            if ~gpu
                fn = func(x_new);
            else
                fn = func(gpuArray(x_new));
                fn = gather(fn);
            end
            invalid_idx = (imag(fn) ~=0 | fn > f) & ~terminated;
            if any(invalid_idx(:))
                t(invalid_idx(:)) = t(invalid_idx(:)) * tr;
            else
                break;
            end            
        end  
        
        %No progress if still invalid
        if i == maxStepIters
            t(invalid_idx(:)) = 0;
        end
        
        %Do step
        terminated = terminated | t < stepTol;
        if all(terminated)
            break;
        end
        
        x(:,~terminated) = x(:,~terminated) + repmat(t(1,~terminated),[2,1]) .* d(:,~terminated);    

    end
    if strcmp(verbose, 'final')
        fprintf('Final [%3d] Step [%g] Func --> %g \n', it, mean(t), sum(f));
    end

end

%Coates method
function [ r ] = coates(T, N, h)
    r = zeros(T,size(h,2));
    r(1,:) = -log(1-h(1,:)/N);
    for k = 2:T
        tmp = N - sum(h(1:k - 1,:),1);
        r(k,:) = -log(1-h(k,:)./tmp);
        r(k,tmp == 0 | h(k,:)./tmp == 1) = 0; %Set invalides to 0
    end
end

%Gaussian fit
function [gauss_mean, gauss_max] = fit_gaussian(hist, bucket_size, sigma_true, T, times, mask)
    
    %Return random sample if 0
    hist_zeros = (sum(hist,1) == 0) | mask;
    N_zeros = sum(hist_zeros);
    h_pos = hist(:,~hist_zeros);
    
    %Normalize
    hist = h_pos./repmat(sum(h_pos,1),[size(h_pos,1),1]);
    SCALE = 1/bucket_size;
    x = (times*bucket_size)*SCALE;
    
    % equation for Gaussian distribution
    norm_func = @(p,x) p(1) .* exp(-((x - p(2))/(sqrt(2)*p(3))).^2);
    gauss_mean = zeros(1,size(hist,2));
    gauss_max = zeros(1,size(hist,2));
    
    %Process in parallel
    process_prog = floor(size(hist,2)/10);
    parfor j = 1:size(hist,2)
        
        %Progress
        if mod(j-1,process_prog) == 0
            fprintf('Progress [%2d %%]\n', (j-1)/process_prog*10)
        end
        initGuess = zeros(3,1);
        initGuess(1) = max(hist(:,j));  % amplitude
        initGuess(2) = sum(x.*hist(:,j));    % mean centred on 0 degrees
        initGuess(3) = sigma_true*SCALE;      % SD in degrees

        % use nlinfit to fit Gaussian using Least Squares
        w = warning('off','all');
        [bestfit,resid]=nlinfit(x, hist(:,j), norm_func, initGuess);
        warning(w);
        gauss_max(1,j) = bestfit(1);
        gauss_mean(1,j) = max(0, min(bestfit(2)/SCALE, (T-1)*bucket_size));
    end
    
    if N_zeros > 0
        %Copy over
        gauss_mean_pos = gauss_mean;
        gauss_max_pos = gauss_max;
        gauss_mean = zeros(1,size(hist,2));
        gauss_max = zeros(1,size(hist,2));
        gauss_mean(:,hist_zeros) = 0;
        gauss_max(:,hist_zeros) = 0;
        gauss_mean(:,~hist_zeros) = gauss_mean_pos;
        gauss_max(:,~hist_zeros) = gauss_max_pos;
    end
end

%Gaussian fit
function [log_mean, log_max] = log_matched_filter(hist, bucket_size, psf, T, times)
    
    %Return random sample if 0
    hist_zeros = (sum(hist,1) == 0);
    N_zeros = sum(hist_zeros);
    h_pos = hist(:,~hist_zeros);
    
    %Normalize
    hist = h_pos./repmat(sum(h_pos,1),[size(h_pos,1),1]);
    psf = psf(:)/sum(psf(:));
    
    %Filter
    [rmatched_val, rmatched] = max( log(imfilter(hist, psf, 'corr', 0)),[],1);
    rmatched_time = (rmatched - 1) * bucket_size;
    log_mean = max(0, min(rmatched_time, (T-1)*bucket_size));
    log_max = hist( sub2ind(size(hist),rmatched,1:size(hist,2)) );
    
    if N_zeros > 0
        %Copy over
        log_mean_pos = log_mean;
        log_max_pos = log_max;
        log_mean = zeros(1,size(hist,2));
        log_max = zeros(1,size(hist,2));
        log_mean(:,hist_zeros) = 0;
        log_max(:,hist_zeros) = 0;
        log_mean(:,~hist_zeros) = log_mean_pos;
        log_max(:,~hist_zeros) = log_max_pos;
    end
    
end


function [zv, av] = naive_coates(h, N, xt, psf, mask)

    %Return random sample if 0
    nz = size(h,2);
    hist_zeros = (sum(h,1) == 0) | mask;
    N_zeros = sum(hist_zeros);
    h = h(:,~hist_zeros);

    %Shifted matched filter after coates init
    T = xt(end)+1;
    rc = coates(T, N,  h);    
    [rmatched_val, rmatched] = max( log(imfilter(rc, psf, 'corr', 0)),[],1);
    z0 = xt(rmatched)';
    %a0 = rc( sub2ind(size(rc),rmatched,1:size(rc,2)) )/max(psf(:))/I;
    z0 = min( z0 + 3, xt(end));
    a0 = 0.5 * ones(size(z0));

    %Initial estimate
    zv = z0;
    av = a0;
    
    if N_zeros > 0
        %Copy over
        zv_pos = zv;
        av_pos = av;
        zv = zeros(1,nz);
        av = zeros(1,nz);
        zv(:,hist_zeros) = 0;
        av(:,hist_zeros) = 0;
        zv(:,~hist_zeros) = zv_pos;
        av(:,~hist_zeros) = av_pos;
    end
    
end
