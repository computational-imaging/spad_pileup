clear;
close all;

%Dataset
fn = 'motorcycle_GaussianPSF_40.0';
fdata = sprintf('./dataset/middlefield_depth_results/%s.mat', fn);
load(fdata, 'a', 'z', 'data', 'c', 'bucket_size' );
sz = size(data.A);
mask = reshape(data.mask, 1, []);

%Reshape
a = reshape(a,sz);
z = reshape(z,sz)*c*bucket_size/2;
z = z / max(z(:));

%Display
figure();
imagesc(a);
axis image, colormap gray, colorbar;
title('Ground-truth amplitude')

figure();
imagesc(z );
axis image, colormap hsv, colorbar;
title('Ground-truth depth')

x0 = cat(3, z, a);
b = cat(3, z, a);
lambda_residual = 10;
lambda_tv_depth = 1.0;
lambda_tv_albedo = 1.0;
weight = 1 - double( cat(3,reshape(mask,sz),reshape(mask,sz)) );
verbose = 'brief';
max_it = 200;
tol = 1e-4;
[ res ] = admm_linearized(b, weight, x0, [size(a,1), size(a,2), 2], ...
                          lambda_residual, lambda_tv_depth, lambda_tv_albedo, ...
                          max_it, verbose);    
figure();
imshow(reshape(res,size(a,1),[])), title('ResultADMM');

[ res ] = pd_solve(b, weight, x0, [size(a,1), size(a,2), 2], ...
                          lambda_residual, lambda_tv_depth, lambda_tv_albedo, ...
                          max_it, tol, verbose);   
figure();
imshow(reshape(res,size(a,1),[])), title('ResultPD');