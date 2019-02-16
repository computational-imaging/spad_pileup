function L = compute_operator_norm(A, AS, vec_size)

    % computes the operator norm for a linear operator AS on images with size sx, 
    % which is the square root of the largest eigenvector of AS*A.
    % http://mathworld.wolfram.com/OperatorNorm.html

    %Compute largest eigenvalue (in this case arnoldi, since matlab
    %implementation faster than power iteration)
    opts.tol = 1.0e-3;
    lambda_largest = eigs(@(x)ASAfun(x, A, AS, vec_size), prod(vec_size(:)), 1,'lm', opts);
    L = sqrt(lambda_largest);

return;

function ASAx = ASAfun(x, A, AS,vec_size)
    x_img = reshape(x,vec_size);
    ASAx = AS(A(x_img));
    ASAx = ASAx(:);
return;