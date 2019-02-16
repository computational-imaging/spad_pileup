function [ res ] = pd_solve(b, mask, x0, sizeI, ...
                              lambda_residual, lambda_tv_depth, lambda_tv_albedo, ...
                              max_it, tol, verbose)    


    %Stack params
    lambdas_prior = [lambda_tv_depth, lambda_tv_albedo];
    if isempty(mask)
        mask = ones(size(b));
    end

    %Prox operators
    
    %Isotropic
    Amplitude = @(u)sqrt(sum(u.^2,4));
    ProxFS = @(u,s) u./ max(1, repmat( Amplitude(u), [1,1,1,2] ) );
    %ProxFS = @(u,s) (u/(1 + thresh_huber * s)) ./ max(1, repmat( Amplitude(u/(1 + thresh_huber * s), [1,1,1,2] ) )); %Huber
    %}
    
    %{
    %Anisotropic
    Amplitude = @(u)sqrt(u.^2); %Anisotropic
    ProxFS = @(u,s) u./ max(1, Amplitude(u) ); %Anisotropic
    %ProxFS = @(u,s) (u/(1 + thresh_huber * s)) ./ max(1, Amplitude(u/(1 + thresh_huber * s))); %Huber
    %}

    %Prox of F
    ProxG = @(v,lambda_prox) (lambda_residual * b + 1/lambda_prox * v)/(lambda_residual + 1/lambda_prox) .* mask + v .*(1-mask);
    
    
    %Penalty matrix A
    Kmult = @(x)KMat(x, lambdas_prior, 1); 
    KSmult = @(x)KMat(x, lambdas_prior, -1);
      
    %Objective we try to solve
    objective = @(x) objectiveFunction( x, b, mask, Kmult, lambda_residual );
      
    %Algorithm parameters
    if all(lambdas_prior == [1,1])
        L = sqrt(8);
    else
        L = compute_operator_norm(Kmult, KSmult, sizeI);
    end
    
    sigma = 1.0*5;
    tau = .9/(sigma*L^2);
    theta = 1.0;
    
    %Set initial iterate
    f = reshape( x0, sizeI); %Simply start with 0-vector
    g = Kmult(f);
    f1 = f;
    
    %Display it.
    if strcmp(verbose, 'all')
        iterate_fig = figure();
        clf;
        imshow(reshape(f1,sizeI(1),[])), title(sprintf('Local PD iterate %d',0));
        pause(0.1)
    end
    
    %Example of one iterations.
    for i = 1:max_it
        
        fold = f;
        g = ProxFS( g + sigma * Kmult(f1), sigma);
        f = ProxG( f - tau * KSmult(g), tau);
        f1 = f + theta * (f-fold);

        %Display it.
        if strcmp(verbose, 'all')
            figure(iterate_fig);
            clf;
            imshow(reshape(f1,sizeI(1),[])), title(sprintf('Local PD iterate %d',i));
            pause(0.1)
        end

        diff = f - fold;
        diff_norm = norm(diff(:), 2);
        if norm(f(:), 2) > eps()
            diff_norm = diff_norm / norm(f(:), 2);
        end
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('iter %03d, diff\t%2.4f, obj\t%7.2f\n', i, diff_norm, objective(f1))
        end
        if diff_norm < tol
            break;
        end
    end
    
    res = f1;
 
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


