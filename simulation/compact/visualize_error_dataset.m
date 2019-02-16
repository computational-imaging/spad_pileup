clear;
close all;

%Dataset
dsets = cell(0);

%psf_sel = 'PicoquantBluePSF';
psf_sel = 'GaussianPSF';

dsets{1} = 'Motorcycle_falloff_%s_11.3_pad30x10_falloff';
dsets{2} = 'Adirondack_falloff_%s_8.4_pad30x10_falloff';
dsets{3} = 'Jadeplant_falloff_%s_10.9_pad30x10_falloff';
dsets{4} = 'Piano_falloff_%s_7.1_pad30x10_falloff';
dsets{5} = 'Pipes_falloff_%s_21.0_pad30x10_falloff';
dsets{6} = 'Playroom_falloff_%s_17.9_pad30x10_falloff';
dsets{7} = 'Playtable_falloff_%s_5.1_pad30x10_falloff';
dsets{8} = 'Recycle_falloff_%s_5.0_pad30x10_falloff';
dsets{9} = 'Shelves_falloff_%s_17.8_pad30x10_falloff';
dsets{10} = 'Vintage_falloff_%s_2.6_pad30x10_falloff';
dsets{11} = 'Bicycle1_falloff_%s_21.7_pad30x10_falloff';

%Output
ofn = './results_probabilistic_2D_dataset';

%Data folder 
datafolder = '/run/media/fheide/My Passport/BayesianSPAD/results';

%Errors
error_depth = zeros(4,length(dsets));
error_depth_max = zeros(4,length(dsets));
error_albedo = zeros(4, length(dsets));

%Save images
save_images = true;
display_all = false;

%Colorbar
acmax = 1.0;
dcmax = 10;

%Select dataset
for didx = 1:length(dsets)

    %Clear all old data
    clearvars -except dsets didx psf_sel error_depth error_albedo save_images ofn display_all acmax dcmax datafolder error_depth_max
    close all;

    %Print
    fprintf('\n###########################################################\n')
    fprintf('#################### Processing [%d/%d] ##################\n', didx, length(dsets))
    fprintf('###########################################################\n')

    %Falloff
    fn = sprintf( dsets{didx}, psf_sel);
    dataset = fn;
    fdata = sprintf('%s/dataset/middlefield_depth_results_dataset/%s.mat', datafolder, fn);
    load(fdata, 'N_trials', 'd', 'QE', 'I', 'amb', 'DC','sigma_true', 'data', 'buckets', 'psf_name', 'gm',  'a', 'z', 'bucket_size', 'c', 'xt', 'xtpsf', 'T', 'times' );
    sz = size(data.A);
    mask = reshape(data.mask, 1, []);

    %Normalize a to be between [0,1] with scale to I
    asc = 1/max(max(a(:)),1);
    I = 1/asc * I;
    a = asc * a;

    %Result
    fres = sprintf('%s/results_probabilistic_2D_dataset/result_prior_%s_lambda0.020000.mat', datafolder, fn);
    load(fres);

    %Pulse
    if(display_all)
        figure();
        imagesc(reshape(a,sz));
        axis image, colormap gray, colorbar;
        title('Ground-truth amplitude')

        figure();
        imagesc(reshape(z,sz)*c*bucket_size/2 );
        axis image, colormap hsv, colorbar;
        title('Ground-truth depth')
    end

    %PSF
    print_rec = 1:round(size(buckets,2)/10):size(buckets,2); %Plot subset of 100

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Naive method
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for j = print_rec
    fprintf('[I = %2.1f], Naive: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_naive(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_naive(1,j)/bucket_size - z(1,j)))
    end

    %Display
    if display_all
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

    if display_all
        figure();
        imagesc(reshape(min(log_max_rc,1),sz));
        axis image, colormap gray, colorbar;
        title('Log-matched Coates amplitude')

        figure();
        imagesc(reshape(c*log_mean_rc/2,sz) );
        axis image, colormap hsv, colorbar;
        title('Log-matched Coates depth')
    end
    
    for j = print_rec
    fprintf('[I = %2.1f], Coates: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*gaussmean_rc(1,j)/2*1000 - d(1,j)*1000), abs(gaussmean_rc(1,j)/bucket_size - z(1,j)))
    end

    %Display
    if display_all
        figure();
        imagesc(reshape(min(gaussint_rc,1),sz));
        axis image, colormap gray, colorbar;
        title('Gauss-fit Coates amplitude')

        figure();
        imagesc(reshape(c*gaussmean_rc/2,sz) );
        axis image, colormap hsv, colorbar;
        title('Gauss-fit Coates depth')
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Probabilistic
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for j = print_rec
    fprintf('[I = %2.1f], Probabilistic Newton: Error in depth is %3.3f mm, %2.2f bins\n', I, abs(c*zv(1,j)*bucket_size/2*1000 - d(1,j)*1000), abs(zv(1,j) - z(1,j)))
    end

    %Display
    if display_all
        figure();
        imagesc(reshape(min(av,1),sz));
        axis image, colormap gray, colorbar;
        title('Probabilistic amplitude')

        figure();
        imagesc(reshape(c*zv*bucket_size/2,sz) );
        axis image, colormap hsv, colorbar;
        title('Probabilistic depth')
    end


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
        if display_all
            figure();
            imagesc(min(aprior,1));
            axis image, colormap gray, colorbar;
            title('Probabilistic TV amplitude')

            figure();
            imagesc(c*zprior*bucket_size/2 );
            axis image, colormap hsv, colorbar;
            title('Probabilistic TV depth')
        end

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

    close all;

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
    set(gca, 'CLim', [0, acmax]);
    axis image, colormap gray, colorbar;
    title(sprintf('AMPLITUDE: GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

    if save_images
        mkdir( sprintf('%s/%s', ofn, dataset) );
        
        cm = get(gcf,'Colormap');
        imgc = cat(2, reshape(a,sz), reshape(min(gaussint_naive,1),sz), reshape(min(gaussint_rc,1),sz), reshape(min(av,1),sz), min(post_methods_a,1) );
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/amplitude.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_amplitude.png', ofn, dataset), [0, acmax], 5, 'gray' );
    end

    figure();
    imagesc( cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000 );
    axis image, colormap hsv, colorbar;
    title(sprintf('DEPTH [mm]: GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

    if save_images
        cm = get(gcf,'Colormap');
        imgc = cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000;
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/depth.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_depth.png', ofn, dataset), [0, max(imgc(:))], 11, 'hsv' );
    end
    
    %Different visualization
    figure();
    imagesc( cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000 );
    axis image, colormap lines, colorbar;

    if save_images
        cm = get(gcf,'Colormap');
        imgc = cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000;
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/depth_lines.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_depth_lines.png', ofn, dataset), [0, max(imgc(:))], 11, 'lines' );
    end
    
    figure();
    imagesc( cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000 );
    axis image, colormap prism, colorbar;

    if save_images
        cm = get(gcf,'Colormap');
        imgc = cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000;
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/depth_prism.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_depth_prism.png', ofn, dataset), [0, max(imgc(:))], 11, 'prism');
    end
    
    figure();
    imagesc( cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000 );
    axis image, colormap colorcube, colorbar;

    if save_images
        cm = get(gcf,'Colormap');
        imgc = cat(2,reshape(z,sz)*c*bucket_size/2, reshape(c*gaussmean_naive/2,sz), reshape(c*gaussmean_rc/2,sz), reshape(c*zv*bucket_size/2,sz), post_methods_z ) * 1000;
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/depth_colorcube.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_depth_colorcube.png', ofn, dataset), [0, max(imgc(:))], 11 , 'colorcube');
    end

    figure();
    imgc = abs( cat(2, reshape(min(gaussint_naive,1),sz) - gta, reshape(min(gaussint_rc,1),sz) - gta, reshape(min(av,1),sz) - gta, post_methods_a_error ) );
    imgc = repmat(maskerror,[1,size(imgc,2)/size(maskerror,2)]) .* imgc;
    imagesc( imgc );
    axis image, colormap hot, colorbar;
    title(sprintf('AMPLITUDE: Error --> GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

    %Compute error
    error_albedo(1,didx) = psnr(reshape(min(gaussint_naive,1),sz), gta);
    error_albedo(2,didx) = psnr(reshape(min(gaussint_rc,1),sz), gta);
    error_albedo(3,didx) = psnr(reshape(min(av,1),sz), gta);
    if exist('zprior') && exist('aprior')
        error_albedo(4,didx) = psnr(reshape(min(aprior,1),sz), gta);
    end

    if save_images
        cm = get(gcf,'Colormap');
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/amplitude_error.png', ofn, dataset))
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_amplitude_error.png', ofn, dataset), [0, max(imgc(:))], 5, 'hot' );
    end

    figure();
    imgc = abs( cat(2, reshape(c*gaussmean_naive/2,sz) - gtz, reshape(c*gaussmean_rc/2,sz) - gtz, reshape(c*zv*bucket_size/2,sz) - gtz, post_methods_z_error )) * 1000;
    imgc = repmat(maskerror,[1,size(imgc,2)/size(maskerror,2)]) .* imgc;
    imagesc( imgc );
    set(gca, 'CLim', [0, dcmax]);
    axis image, colormap hot, colorbar;
    title(sprintf('DEPTH [mm]: Error --> GT | Naive Gauss-Fit | Coates Gauss-Fit | Probabilistic %s', post_methods_string))

    %Compute error
    error_depth(1,didx) = mean( maskerror(:) .* abs( c*gaussmean_naive(:)/2 - gtz(:) ) * 1000);
    error_depth(2,didx) = mean( maskerror(:) .* abs( c*gaussmean_rc(:)/2 - gtz(:) ) * 1000);
    error_depth(3,didx) = mean( maskerror(:) .* abs( c*zv(:)*bucket_size/2 - gtz(:) ) * 1000);
    if exist('zprior') && exist('aprior')
        error_depth(4,didx) = mean( maskerror(:) .* abs( c*zprior(:)*bucket_size/2 - gtz(:) ) * 1000);
    end

    error_depth_max(1,didx) = max( maskerror(:) .* abs( c*gaussmean_naive(:)/2 - gtz(:) ) * 1000);
    error_depth_max(2,didx) = max( maskerror(:) .* abs( c*gaussmean_rc(:)/2 - gtz(:) ) * 1000);
    error_depth_max(3,didx) = max( maskerror(:) .* abs( c*zv(:)*bucket_size/2 - gtz(:) ) * 1000);
    if exist('zprior') && exist('aprior')
        error_depth_max(4,didx) = max( maskerror(:) .* abs( c*zprior(:)*bucket_size/2 - gtz(:) ) * 1000);
    end
    
    if save_images
        cm = get(gcf,'Colormap');
        imgc = ind2rgb(int32(imgc/max(imgc(:)) * size(cm,1)), cm);
        imwrite(imgc, sprintf('%s/%s/depth_error.png', ofn, dataset) )
        
        %Save colormap
        save_cmap( sprintf('%s/%s/cmap_depth_error.png', ofn, dataset), [0, max(imgc(:))], 11, 'hot' );
    end

end

%Print results
error_albedo
mean(error_albedo,2)

error_depth
mean(error_depth,2)

error_depth_max
mean(error_depth_max,2)

%Save results
save(sprintf('%s/error_list.mat', ofn), 'error_albedo', 'error_depth', 'dcmax', 'acmax');