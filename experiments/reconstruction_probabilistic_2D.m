function [] = reconstruction_probabilistic_2D()

    % This code demonstrates the proposed method on two example dataset
    clear;
    %close all;
    
    %Script parameters
    display_methods = true; %Display all results ?
    display_log_matched = false; %Display log-matched ?
    quick_compute = false; %Run quick method ?
    display_pointclouds = true; %Display pointclouds ?
    
    %Scenes
    scenes = cell(0);
    scenes{1} = 'basrelief';
    scenes{2} = 'david';

    for scene_idx = 1:length(scenes) 
        
        %Display
        clearvars -except scene_idx scenes display_methods display_log_matched quick_compute display_pointclouds
        fprintf('\n\n################################################\n')
        fprintf('############ Runnig Scene %s #############\n', scenes{scene_idx})
        fprintf('################################################\n\n')
        
        %Load scene
        scene = scenes{scene_idx};
        if strcmp(scene, 'basrelief')
            %%%%%%%%%%%%
            %%% Basrelief
            %%%%%%%%%%%%
            laser_type = 'blue';
            data_folder = '../data';
            h = 150; %Dim H
            w = 150; %Dim W
            ebar = [0,5];
            dlimits_3D = [3.784,3.804]/2*1e4;
            
        elseif strcmp(scene, 'david')
            %%%%%%%%%%%%
            %%% David
            %%%%%%%%%%%%
            laser_type = 'blue';
            data_folder = '../data';
            h = 150; %Dim H
            w = 150; %Dim W
            ebar = [0,10];
            dlimits_3D = [1.8895,1.893]*10^4;
            
        else 
            error('Scene invalid');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Load data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %Load measurement
        load(sprintf('%s/%s.mat', data_folder, scene), 'buckets', 'sz', 'mask', 'bucket_size', 'psf', 'gm', ...
                     'd', 'z', 'I', 'xt', 'times', 'N_trials' , 'as', 'background', 'depth_offsets', 'crop_offset', 'depth_cropstart');

        % Constants.
        c = 3e8; % speed of light
        if strcmp(laser_type, 'red')
            QE = 0.34; %Quantum efficiency for red
        elseif strcmp(laser_type, 'blue')
            QE = 0.32; %Quantum efficiency for blue
        end
        DC = 0.00; % dark counts per exposure
        T = times(end)+1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Naive method
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %Logmatched filter
        [log_mean_naive, log_max_naive] = log_matched_filter(buckets, bucket_size, psf, T, times);
        log_mean_naive(logical(background(:))) = 0;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Probabilistic model
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %Compute a and z
        gpu_implem =true;
        [zv, av] = fit_pileup_newton(buckets, N_trials, xt, gm, psf, QE, I, as, DC, mask, gpu_implem);
        zv_prev = zv;
        zv(logical(background(:))) = 0;

        %Display
        if quick_compute && display_methods
            
            %Display Amplitude
            figure();
            imagesc(flipud(reshape(min(av*10,1),sz)'));
            axis image, colormap gray;
            title(sprintf('Scene [%s] amplitude (x10) for proposed probabilistic method (without priors)',scene));
            pause(0.5);
            
            %Display error
            if ~display_log_matched
                figure();
                imagesc( flipud(abs(reshape(zv*c*bucket_size/2*1000 - d*1000,sz))') );
                axis image, colormap jet, caxis([0,10]), colorbar;
                title(sprintf('Scene [%s] Depth Error [mm] for proposed method (without priors)',scene));
            else
                figure();
                imagesc(flipud(cat(2, abs(reshape( c*log_mean_naive/2*1000- d*1000,sz))',  ...
                                      abs(reshape(zv*c*bucket_size/2*1000 - d*1000,sz))' ) ));
                axis image, colormap jet, caxis(ebar), colorbar;
                title(sprintf('Scene [%s] Depth Error [mm]: Log-Matched Filter | Proposed (without priors)',scene)); 
            end
            pause(0.5);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Spatio-Temporal Reconstruction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Bypass for quick preview 
        if ~quick_compute

            %Initial iterate as warmstart
            x0 = cat(3, reshape(zv_prev, sz), reshape(av, sz));

            %Algorihtm params
            sizeI = size(x0);
            lambda_residual = 10.0;
            lambda_tv_depth = 1.0;
            lambda_tv_albedo = 1.0;
            verbose = 'brief';
            max_it = 20; %200;
            gpu_implem =true;
            warmstart = 1;

            %Compute a and z
            [ res ] = admm_linearized(buckets, mask, x0, size(buckets,1) - 1, sizeI, ...
                                            lambda_residual, lambda_tv_depth, lambda_tv_albedo, warmstart, ...
                                            N_trials, xt, gm, QE, I, as, DC, gpu_implem, ...
                                            max_it, verbose);                           
            %Z and A prior
            zprior_prev = reshape(res(:,:,1),1,[]);
            aprior = reshape(res(:,:,2),1,[]);
            zprior = zprior_prev;
            zprior(logical(background(:))) = 0;

            %Display
            if display_methods

                %Display Amplitude
                figure();
                imagesc(flipud(reshape(min(aprior*10,1),sz)'));
                axis image, colormap gray;
                title(sprintf('Scene [%s] amplitude (x10) for proposed probabilistic method',scene));
                pause(0.5);

                %Display Error
                if ~display_log_matched
                    figure();
                    imagesc( flipud(abs(reshape(zprior*c*bucket_size/2*1000 - d*1000,sz))') );
                    axis image, colormap jet, caxis([0,10]), colorbar;
                    title(sprintf('Scene [%s] Depth Error [mm] for proposed method',scene));
                    pause(0.5);
                else
                    %Display error maps
                    figure();
                    imagesc(flipud(cat(2, abs(reshape( c*log_mean_naive/2*1000- d*1000,sz))',  ...
                                          abs(reshape(zprior*c*bucket_size/2*1000 - d*1000,sz))' ) ));
                    axis image, colormap jet, caxis(ebar), colorbar; 
                    title(sprintf('Scene [%s] Depth Error [mm]: Log-Matched Filter | Proposed',scene)); 
                end
                pause(0.5);
            end

        else %Quick method
           %Plot per-pixel method
           zprior_prev = zv_prev;
           aprior = av;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Visualize Pointcloud
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if display_pointclouds
            %Target intensity
            intensity = repmat( flipud(reshape(aprior,sz)'), [1,1,3] ) * 10;

            %Assemble as point cloud
            pause(0.1);
            figure()
            Z = flipud(reshape(zprior_prev(:) + depth_offsets(:), sz)') *bucket_size*c/2 * 1000; %in mm
            [X,Y] =meshgrid(1:size(Z,1), 1:size(Z,2));
            Z_t = min( max(Z, dlimits_3D(1)), dlimits_3D(2)) - dlimits_3D(1);
            Z_t(Z_t==0) = -1;

            pcshow([X(:),Y(:),Z_t(:)], reshape(intensity,[],3), 'MarkerSize', 41), zlim([0, max(Z_t(:))-0.1]);
            set(gca,'Ydir','reverse')
            fig = gcf;
            fig.Color = 'white';
            grid off
            axis off
            view(0, 87);
            daspect([1.2 1 1])
            title(sprintf('Scene [%s] Pointcloud',scene))
            pause(0.5);
        end
        
    end %Next scene

return;

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
    
return;

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
    
return;

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
    z0 = min( z0, xt(end));
    a0 = 0.5 * ones(size(z0));

    %Initial estimate
    x0 = cat(1,z0,a0);
    %x_val = newton_opt(objGrad_f, gpu, x0, 1e-4, 1e-7, 1e-9, 1, 'iter'); %200, 'iter');
    x_val = newton_opt(objGrad_f, gpu, x0, 1e-4, 1e-7, 1e-9, 20, 'iter'); %200, 'iter');
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
    
return;

%Coates method
function [ r ] = coates(T, N, h)
    r = zeros(T,size(h,2));
    r(1,:) = -log(1-h(1,:)/N);
    for k = 2:T
        tmp = N - sum(h(1:k - 1,:),1);
        r(k,:) = -log(1-h(k,:)./tmp);
        r(k,tmp == 0 | h(k,:)./tmp == 1) = 0; %Set invalides to 0
    end
return;

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

return;

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
    
return;
