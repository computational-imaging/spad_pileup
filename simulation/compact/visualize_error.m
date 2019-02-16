clear;
%close all;

%Dataset
fn = 'motorcycle_GaussianPSF_40.0_pad';
%fn = 'motorcycle_PicoquantBluePSF_40.0_pad_30';

%Dataset
%fn = 'motorcycle_small_GaussianPSF_40.0_pad';
%fn = 'motorcycle_small_GaussianPSF_40.0_pad20x10';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad20x10';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad30x10';
%fn = 'motorcycle_PicoquantBluePSF_40.0_pad_30';

%Parameter test
%fn = 'motorcycle_small_GaussianPSF_40.0_pad';
%fn = 'motorcycle_small_PicoquantBluePSF_40.0_pad20x10';

%Falloff
%fn = 'motorcycle_small_falloff_GaussianPSF_11.5_pad30x10_falloff';
%fn = 'motorcycle_small_falloff_PicoquantBluePSF_11.5_pad30x10_falloff';

%Long tests
%fn = 'motorcycle_small_falloff_GaussianPSF_184.0_pad30x10_falloff_long';
%fn = 'motorcycle_small_falloff_PicoquantBluePSF_184.0_pad30x10_falloff_long';

fdata = sprintf('./dataset/middlefield_depth_results/%s.mat', fn);
load(fdata, 'N_trials', 'd', 'QE', 'I', 'amb', 'DC','sigma_true', 'data', 'buckets', 'psf_name', 'gm',  'a', 'z', 'bucket_size', 'c', 'xt', 'xtpsf', 'T', 'times' );
sz = size(data.A);
mask = reshape(data.mask, 1, []);

%Normalize a to be between [0,1] with scale to I
asc = 1/max(max(a(:)),1);
I = 1/asc * I;
a = asc * a;

%Result
%fres = sprintf('./results_probabilistic_2D/%s_NewtonInit.mat', fn);
fres = sprintf('./results_probabilistic_2D/parameter_test/%s_lambda0.1.mat', fn);
%fres = sprintf('./results_probabilistic_2D/%s_lambda1000.mat', fn);
load(fres);

%Pulse
figure();
imagesc(reshape(a,sz));
axis image, colormap gray, colorbar;
title('Ground-truth amplitude')

figure();
imagesc(reshape(z,sz)*c*bucket_size/2 );
axis image, colormap hsv, colorbar;
title('Ground-truth depth')

%PSF
print_rec = 1:round(size(buckets,2)/10):size(buckets,2); %Plot subset of 100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Naive method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j = print_rec
fprintf('[I = %2.1f], Naive: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_naive(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_naive(1,j)/bucket_size - z(1,j)))
end

%Display
figure();
imagesc(reshape(min(gaussint_naive,1),sz));
axis image, colormap gray, colorbar;
title('Gauss-fit amplitude')

figure();
imagesc(reshape(c*gaussmean_naive/2,sz) );
axis image, colormap hsv, colorbar;
title('Gauss-fit depth')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Apply on coates result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure();
imagesc(reshape(min(log_max_rc,1),sz));
axis image, colormap gray, colorbar;
title('Log-matched Coates amplitude')

figure();
imagesc(reshape(c*log_mean_rc/2,sz) );
axis image, colormap hsv, colorbar;
title('Log-matched Coates depth')

for j = print_rec
fprintf('[I = %2.1f], Coates: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_rc(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_rc(1,j)/bucket_size - z(1,j)))
end

%Display
figure();
imagesc(reshape(min(gaussint_rc,1),sz));
axis image, colormap gray, colorbar;
title('Gauss-fit Coates amplitude')

figure();
imagesc(reshape(c*gaussmean_rc/2,sz) );
axis image, colormap hsv, colorbar;
title('Gauss-fit Coates depth')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Probabilistic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j = print_rec
fprintf('[I = %2.1f], Probabilistic Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv(1,j)*bucket_size/2*1000 - d(1,j)*1000), abs(zv(1,j) - z(1,j)))
end

%Display
figure();
imagesc(reshape(min(av,1),sz));
axis image, colormap gray, colorbar;
title('Probabilistic amplitude')

figure();
imagesc(reshape(c*zv*bucket_size/2,sz) );
axis image, colormap hsv, colorbar;
title('Probabilistic depth')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Probabilistic TV Prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist('zprior') && exist('aprior')
    
    zprior_vec = zprior;
    aprior_vec = aprior;
    zprior = reshape(zprior, sz);
    aprior = reshape(aprior, sz);
    for j = print_rec
        fprintf('[I = %2.1f], Probabilistic TV Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zprior_vec(1,j)*bucket_size/2*1000 - d(1,j)*1000), abs(zprior_vec(1,j) - z(1,j)))
    end

    %Display
    figure();
    imagesc(min(aprior,1));
    axis image, colormap gray, colorbar;
    title('Probabilistic TV amplitude')

    figure();
    imagesc(c*zprior*bucket_size/2 );
    axis image, colormap hsv, colorbar;
    title('Probabilistic TV depth')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
gaussmean_naive(mask) = z(mask)*bucket_size;
gaussmean_rc(mask) = z(mask)*bucket_size;
gaussint_naive(mask) = a(mask);
gaussint_rc(mask) = a(mask);
%}
gaussmean_naive(mask) = xt(end)*bucket_size;
gaussmean_rc(mask) = xt(end)*bucket_size;
zv(mask) = (T-1);
gaussint_naive(mask) = 0;
gaussint_rc(mask) = 0;
av(mask) = 0;

%GT
gta = reshape(a,sz);
gtz = reshape(z,sz)*c*bucket_size/2;

%close all;

%Save images
save_images = true;

%Zprior and Aprior
post_methods_a = [];
post_methods_z = [];
post_methods_a_error = [];
post_methods_z_error = [];
post_methods_string = '';
if exist('zprior') && exist('aprior')
    post_methods_a = aprior;
    post_methods_z = c*zprior*bucket_size/2;
    post_methods_a_error = reshape(min(aprior,1),sz) - gta;
    post_methods_z_error = c*zprior*bucket_size/2 - gtz;
    post_methods_string = '| Probabilistic TV';
end

%Maskerror
maskerror = 1-reshape(mask,sz);

figure();
imagesc( cat(2, reshape(a,sz), reshape(min(gaussint_naive,1),sz), reshape(min(gaussint_rc,1),sz), reshape(min(av,1),sz), post_methods_a ) );
axis image, colormap gray, colorbar;
title(sprintf('AMPLITUDE: GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

if save_images
    cm = get(gcf,'Colormap');
    imgc = cat(2, reshape(a,sz), reshape(min(gaussint_naive,1),sz), reshape(min(gaussint_rc,1),sz), reshape(min(av,1),sz), min(post_methods_a,1) );
    imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
    imwrite(imgc, 'amplitude.png')
end

figure();
imagesc( cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000 );
axis image, colormap hsv, colorbar;
title(sprintf('DEPTH [mm]: GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

if save_images
    cm = get(gcf,'Colormap');
    imgc = cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000;
    imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
    imwrite(imgc, 'depth.png')
end

figure();
imgc = abs( cat(2, reshape(min(gaussint_naive,1),sz) - gta, reshape(min(gaussint_rc,1),sz) - gta, reshape(min(av,1),sz) - gta, post_methods_a_error ) );
imgc = repmat(maskerror,[1,size(imgc,2)/size(maskerror,2)]) .* imgc;
imagesc( imgc );
axis image, colormap hot, colorbar;
title(sprintf('AMPLITUDE: Error --> GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

if save_images
    cm = get(gcf,'Colormap');
    imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
    imwrite(imgc, 'amplitude_error.png')
end

figure();
imgc = abs( cat(2, reshape(c*gaussmean_naive/2,sz) - gtz, reshape(c*gaussmean_rc/2,sz) - gtz, reshape(c*zv*bucket_size/2,sz) - gtz, post_methods_z_error )) * 1000;
imgc = repmat(maskerror,[1,size(imgc,2)/size(maskerror,2)]) .* imgc;
imagesc( imgc );
axis image, colormap hot, colorbar;
title(sprintf('DEPTH [mm]: Error --> GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

if save_images
    cm = get(gcf,'Colormap');
    imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
    imwrite(imgc, 'depth_error.png')
end


mean( maskerror(:) .* abs(c*gaussmean_naive(:)/2*1000 - gtz(:)*1000),1)
mean( maskerror(:) .* abs(c*gaussmean_rc(:)/2*1000 - gtz(:)*1000),1)
mean( maskerror(:) .* abs(c*zv(:)*bucket_size/2*1000 - gtz(:)*1000),1)
mean( maskerror(:) .* abs(c*zprior(:)*bucket_size/2*1000 - gtz(:)*1000),1)