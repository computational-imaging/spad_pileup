function [ res ] = admm_linearized(h, mask, x0, scz, sizeI, ...
                                  lambda_residual, lambda_tv_depth, lambda_tv_albedo, warmstart, ...
                                  N, xt, gm, QE, I, amb, DC, gpu, ...
                                  max_it, verbose)    


    %Stack params
    lambdas_prior = [lambda_tv_depth, lambda_tv_albedo];
    
    %Scale z
    x0(:,:,1) = x0(:,:,1)/scz;

    %Reshape
    h = reshape( h, size(h,1), []);
    
    %Prepare mask
    if isempty(mask)
        mask = ones(size(h));
    else
        mask = reshape( mask, 1, []);
    end
    mask = (sum(h,1) == 0) | mask;
    mask = ~mask;
    
    %Display it.
    if strcmp(verbose, 'all')
        figure();
        xs = x0;
        %xs(:,:,1) = xs(:,:,1)/size(h,1);
        imshow(cat(2, reshape(xs,sizeI(1),[]), reshape(mask,sizeI(1),[])) ), title('Mask');
    end
    
    %Prox operators
    %Prox of F
    ProxF = @(v, vv, lambda_prox) solve_data_prox(lambda_residual, lambda_prox, v, mask, vv, warmstart, ...
                                         h, N, xt, gm, QE, I, amb, DC, scz, gpu );
    
    %Prox of G
    ProxG = @(v,lambda_prox) proxShrink(v, lambda_prox);
    
    %Penalty matrix A
    Amult = @(x)KMat(x, lambdas_prior, 1); 
    ATmult = @(x)KMat(x, lambdas_prior, -1);
    
    %Objective we try to solve
    objective = @(x) objectiveFunction( x, lambda_residual, mask, Amult, ...
                                        h, N, xt, gm, QE, I, amb, DC, scz, gpu );

    %Algorithm parameters
    if all(lambdas_prior == [1,1])
        L = sqrt(8);
    else
        L = compute_operator_norm(Amult, ATmult, sizeI);
    end
    
    lambda_algorithm = 1.0 * 4/5;
    mu_algorithm = .15 * (lambda_algorithm / L^2);
    %lambda_algorithm = 1.0 * 2/5;
    %mu_algorithm = .1 * (lambda_algorithm / L^2);
    
    %Overrelaxation
    alpha = 1.0; %Between 1.0 and 1.8
    
    %Default tolerances
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    
    %Set initial iterate
    x = reshape( x0, sizeI); %Simply start with backprojection solution
    z = zeros( size( Amult(x) ) );
    u = zeros( size( z ) );
    
    %Compute dimeinsinoality
    p_elem = length(z(:));
    n_elem = length(x(:));
    
    %Display it.
    if strcmp(verbose, 'all')
        iterate_fig = figure();
        clf;
        xs = x;
        %xs(:,:,1) = xs(:,:,1)/size(h,1);
        imshow(reshape(xs,sizeI(1),[])), title(sprintf('Lin-ADMM iterate %d',0));
    end
    
    %Debug
    if strcmp(verbose, 'all') || strcmp(verbose, 'brief')
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
          'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end
    
    %Do iterations
    for k = 1:max_it
        
        % x-update
        x = ProxF( x - (mu_algorithm / lambda_algorithm) * ATmult( Amult(x) - z + u ), x, mu_algorithm );

        % z-update with relaxation
        zold = z;
        Ax_hat = alpha* Amult(x) +(1-alpha)*zold;
        z = ProxG(Ax_hat + u, lambda_algorithm);
        
        % y-update
        u = u + Ax_hat - z;
        
        %Display it.
        if strcmp(verbose, 'all')
            figure(iterate_fig);
            clf;
            xs = x;
            %xs(:,:,1) = xs(:,:,1)/size(h,1);
            imshow(reshape(xs,sizeI(1),[])), title(sprintf('Lin-ADMM iterate %d',k));
        end

        % diagnostics, reporting, termination checks
        history.objval(k)  = objective(x);

        history.r_norm(k)  = norm( reshape( Amult(x) - z, [],1)); %Minus from virtual B matrix
        history.s_norm(k)  = norm( reshape( -(1/lambda_algorithm)* ATmult(z - zold), [], 1)); %Minus from virtual B matrix

        history.eps_pri(k) = sqrt(p_elem)*ABSTOL + RELTOL*max( norm( reshape( Amult(x), [],1 )), norm( reshape(-z, [], 1) ) );
        history.eps_dual(k)= sqrt(n_elem)*ABSTOL + RELTOL*norm( reshape( (1/lambda_algorithm) * ATmult(u), [], 1) );

        if strcmp(verbose, 'all') || strcmp(verbose, 'brief')
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%g\n', k, ...
                history.r_norm(k), history.eps_pri(k), ...
                history.s_norm(k), history.eps_dual(k), history.objval(k));
        end

        %Check for optimaliy
        if (history.r_norm(k) < history.eps_pri(k) && ...
           history.s_norm(k) < history.eps_dual(k))
             break;
        end
    end
    
    %Result
    res = x;
    
    %Scale z
    res(:,:,1) = res(:,:,1)*scz;
 
return;

function prox = proxShrink(v, lambda_prox)

    %Isotropic
    Amplitude = @(u)sqrt(sum(v.^2,4));
    prox = max( 0, 1 - lambda_prox./ repmat( Amplitude(v), [1,1,1,2] ) ) .* v;

    %Anisotropic
    %prox = max( 0, 1 - lambda_prox./ abs(v) ) .* v;
    
return;

function f_val = objectiveFunction( x, lambda_residual, mask, Amult, ...
                                    h, N, xt, gm, QE, I, amb, DC, scz, gpu )
    %Filter
    h = h(:,mask);
    xs = cat(1, reshape(x(:,:,1),1,[]), reshape(x(:,:,2),1,[]) );
    xs = xs(:,mask);
    
    %Equal to prev
    vz = xs(1,:);
    va = xs(2,:);

    %Vectorized objective
    if ~gpu
        objGrad_f = @(xn) obj_grad_func( xn(1,:), xn(2,:), scz, h, N, xt, gm, QE, I, amb, DC, lambda_residual, 1, vz, va );
        f_x = objGrad_f(xs);
    else
        objGrad_f = @(xn,j) obj_grad_func( xn(1,:), xn(2,:), scz, ...
                                           gpuArray(h), gpuArray(N), gpuArray(xt), gm, gpuArray(QE), gpuArray(I), gpuArray(amb), gpuArray(DC), ...
                                           lambda_residual, 1, gpuArray(vz), gpuArray(va));
        f_x = objGrad_f(gpuArray(xs));
        f_x = gather(f_x);
    end
    
    g_Ax = sum( abs( reshape( Amult(x), [], 1 ) ), 1 );
    
    %Function val
    f_val = sum(f_x(:)) + g_Ax;
    
return;

function [ result ] = KMat( x, lambdas_prior, flag )
    
    %Iterate over the channels
    if flag > 0       

        %Computes A * x
        %Computes deriv * x
        xx = x(:,[2:end end],:)-x;
        xy = x([2:end end],:,:)-x;

        %Stack result
        result = cat(4, xx, xy);
        
        %Weight
        result(:,:,1,:) = lambdas_prior(1) * result(:,:,1,:);
        result(:,:,2,:) = lambdas_prior(2) * result(:,:,2,:);

    elseif flag < 0 
        
        %Computes A' * x
        %Weight
        x(:,:,1,:) = lambdas_prior(1) * x(:,:,1,:);
        x(:,:,2,:) = lambdas_prior(2) * x(:,:,2,:);

        %Computes deriv' * x
        xx = x(:,:,:,1)-x(:,[1 1:end-1],:,1);
        xx(:,1,:)   = x(:,1,:,1);
        xx(:,end,:) = -x(:,end-1,:,1);

        xy = x(:,:,:,2)-x([1 1:end-1],:,:,2);
        xy(1,:,:)   = x(1,:,:,2);
        xy(end,:,:) = -x(end-1,:,:,2);

        %Result
        result = - (xy + xx);

    end
    
return;

function [rprox] = solve_data_prox(lambda_residual, tau, v, mask, vv, warmstart, ...
                                    h, N, xt, gm, QE, I, amb, DC, scz, gpu )

    %Reshape
    vz = reshape( v(:,:,1), 1, []);
    va = reshape( v(:,:,2), 1, []);
    
    %Reshape
    if warmstart == 1
        vvz = reshape( vv(:,:,1), 1, []);
        vva = reshape( vv(:,:,2), 1, []);
        vvz_sub = vvz(:,mask);
        vva_sub = vva(:,mask);  
    end
    
    %Return random sample if 0
    nz = size(h,2);
    N_zeros = sum(~mask);
    h = h(:,mask);
    vz_sub = vz(:,mask);
    va_sub = va(:,mask);
    
    %Vectorized objective
    if ~gpu
        objGrad_f = @(xn) obj_grad_func( xn(1,:), xn(2,:), scz, h, N, xt, gm, QE, I, amb, DC, lambda_residual, tau, vz_sub, va_sub );
    else
        objGrad_f = @(xn,j) obj_grad_func( xn(1,:), xn(2,:), scz, ...
                                           gpuArray(h), gpuArray(N), gpuArray(xt), gm, gpuArray(QE), gpuArray(I), gpuArray(amb), gpuArray(DC), ...
                                           lambda_residual, tau, gpuArray(vz_sub), gpuArray(va_sub));
    end
    
    %Initialize with previous iterate
    if warmstart == 0
        x0 = cat(1,vz_sub,va_sub);
    elseif warmstart == 1
        x0 = cat(1,vvz_sub,vva_sub);
    end
    
    %x_val = newton_opt(objGrad_f, gpu, x0, 1e-4, 1e-7, 1e-9, 50, 'iter');
    x_val = newton_opt(objGrad_f, gpu, x0, 1e-4, 1e-7, 1e-5, 20, 'iter');
    zopt = x_val(1,:);
    aopt = x_val(2,:);
    
    %Project the ones that are not masked
    if N_zeros > 0
        zopt_pos = zopt;
        aopt_pos = aopt;
        zopt = zeros(1,nz);
        aopt = zeros(1,nz);
        zopt(:,~mask) = vz(:,~mask);
        aopt(:,~mask) = va(:,~mask);
        zopt(:,mask) = zopt_pos;
        aopt(:,mask) = aopt_pos;
    end
    
    %Reshape
    rprox = cat(3, reshape(zopt,size(v,1),size(v,2)), reshape(aopt,size(v,1),size(v,2)));
    
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
            [fs,gs,Hs] = func(gpuArray(x));
            f = gather(fs);
            g = gather(gs);
            H = gather(Hs);
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

            if any(isnan(reshape(H(:,:,j),[],1))) || any(isinf(reshape(H(:,:,j),[],1)))
                d(:,j) = 0;
                gtd(j) = 0;
            else
                
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
            invalid_idx = ( ~isreal(x_new(1,:)) | (x_new(1,:) < 0) | ~isreal(x_new(2,:)) | (x_new(2,:) < 0) ) & ~terminated;
            if any(invalid_idx(:))
                t(invalid_idx) = t(invalid_idx) * tr;
                continue;
            end  
            
            %Function query
            if ~gpu
                fn = func(x_new);
            else
                fns = func(gpuArray(x_new));
                fn = gather(fns);
            end
            invalid_idx = ( (imag(fn) ~=0) | (fn > f) ) & ~terminated;
            if any(invalid_idx(:))
                t(invalid_idx) = t(invalid_idx) * tr;
            else
                break;
            end            
        end  
        
        %No progress if still invalid
        if i == maxStepIters
            t(invalid_idx) = 0;
            x_new =  x + repmat(t,[2,1]) .* d;
            if ~gpu
                fn = func(x_new);
            else
                fns = func(gpuArray(x_new));
                fn = gather(fns);
            end
            invalid_idx = ( (imag(fn) ~=0) | (fn > f) ) & ~terminated;
            t(invalid_idx) = 0;
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
function [f_x, g_x, H_x] = obj_grad_func( z, a, scz, h, N, xt, gm, QE, I, amb, DC, lambda_residual, tau, vz, va )

    %Get model and gradient
    if nargout > 2
        [lambda_model, lambda_grad, lambda_HK] = model_func( z*scz, a, xt, gm );
    elseif nargout > 1
        [lambda_model, lambda_grad] = model_func( z*scz, a, xt, gm );
    else
        [lambda_model] = model_func( z*scz, a, xt, gm );
    end
    
    %Scaling and offset
    lambda_model = QE*(I*lambda_model + amb) + DC;
    if nargout > 1
        lambda_grad = QE*I*lambda_grad;
        lambda_grad(:,:,1) = lambda_grad(:,:,1) * scz;
    end
    if nargout > 2
        lambda_HK = QE*I*lambda_HK;
        lambda_HK(:,:,1) = lambda_HK(:,:,1) * scz^2;
        lambda_HK(:,:,2) = lambda_HK(:,:,2) * scz;
        lambda_HK(:,:,3) = lambda_HK(:,:,3) * scz;
    end
    
    %Numerical issues
    ep = eps();
    lambda_model( lambda_model < ep) = ep;
    
    %Function value
    lcumsum = cumsum( lambda_model(1:end-1,:) ,1 );
    lcumsum = cat(1, zeros([1,size(h,2)]), lcumsum);
    f_x = sum(lambda_model,1) .* (N*ones([1,size(h,2)]) - sum(h,1)) + sum(h .* lcumsum,1) - sum( h.* log( 1-exp(-lambda_model) ), 1 ); 
    
    %Proximal form
    f_x = lambda_residual * f_x + 1/(2*tau) * ( (z - vz).^2 + (a - va).^2 );
    
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
        
        %Proximal form
        g_x = lambda_residual * g_x + 1/tau * cat(1, reshape(z - vz,1,[]), reshape(a - va,1,[]) );
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
        
        %Proximal form
        H_x = lambda_residual * H_x;
        H_x(:,:,1,1) = H_x(:,:,1,1) + 1/tau;
        H_x(:,:,2,2) = H_x(:,:,2,2) + 1/tau;
        
        %Reshape
        H_x = reshape( shiftdim(H_x,2), [2,2, size(h,2)]);
    end
    
return;
