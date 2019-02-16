function [ res ] = admm_linearized(b, mask, x0, sizeI, ...
                                  lambda_residual, lambda_tv_depth, lambda_tv_albedo, ...
                                  max_it, verbose)    


    %Stack params
    lambdas_prior = [lambda_tv_depth, lambda_tv_albedo];
    if isempty(mask)
        mask = ones(size(b));
    end

    %Prox operators
    
    %Prox of F
    ProxF = @(v,lambda_prox) (lambda_residual * b + 1/lambda_prox * v)/(lambda_residual + 1/lambda_prox) .* mask + v .*(1-mask);
    
    %Prox of G
    ProxG = @(v,lambda_prox) proxShrink(v, lambda_prox);
    
    %Penalty matrix A
    Amult = @(x)KMat(x, lambdas_prior, 1); 
    ATmult = @(x)KMat(x, lambdas_prior, -1);
    
    %Objective we try to solve
    objective = @(x) objectiveFunction( x, b, mask, Amult, lambda_residual );
      
    %Algorithm parameters
    if all(lambdas_prior == [1,1])
        L = sqrt(8);
    else
        L = compute_operator_norm(Amult, ATmult, sizeI);
    end
    
    lambda_algorithm = 1.0 * 1/5;
    mu_algorithm = .9 * (lambda_algorithm / L^2);
    
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
        imshow(reshape(x,sizeI(1),[])), title(sprintf('Lin-ADMM iterate %d',0));
    end
    
    %Debug
    if strcmp(verbose, 'all') || strcmp(verbose, 'brief')
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
          'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end
    
    %Do iterations
    for k = 1:max_it
        
        % x-update
        x = ProxF( x - (mu_algorithm / lambda_algorithm) * ATmult( Amult(x) - z + u ), mu_algorithm );

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
            imshow(reshape(x,sizeI(1),[])), title(sprintf('Lin-ADMM iterate %d',k));
        end

        % diagnostics, reporting, termination checks
        history.objval(k)  = objective(x);

        history.r_norm(k)  = norm( reshape( Amult(x) - z, [],1)); %Minus from virtual B matrix
        history.s_norm(k)  = norm( reshape( -(1/lambda_algorithm)* ATmult(z - zold), [], 1)); %Minus from virtual B matrix

        history.eps_pri(k) = sqrt(p_elem)*ABSTOL + RELTOL*max( norm( reshape( Amult(x), [],1 )), norm( reshape(-z, [], 1) ) );
        history.eps_dual(k)= sqrt(n_elem)*ABSTOL + RELTOL*norm( reshape( (1/lambda_algorithm) * ATmult(u), [], 1) );

        if strcmp(verbose, 'all') || strcmp(verbose, 'brief')
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
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
 
return;

function prox = proxShrink(v, lambda_prox)

    %Isotropic
    Amplitude = @(u)sqrt(sum(v.^2,4));
    prox = max( 0, 1 - lambda_prox./ repmat( Amplitude(v), [1,1,1,2] ) ) .* v;

    %Anisotropic
    %prox = max( 0, 1 - lambda_prox./ abs(v) ) .* v;
    
return;

function f_val = objectiveFunction( x, b, mask, Amult, lambda_residual )
    
    f_x = lambda_residual * 1/2 * norm( reshape(mask .* (x - b), [], 1) , 2 )^2;
    g_Ax = sum( abs( reshape( Amult(x), [], 1 ) ), 1 );
    
    %Function val
    f_val = f_x + g_Ax;
    
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

