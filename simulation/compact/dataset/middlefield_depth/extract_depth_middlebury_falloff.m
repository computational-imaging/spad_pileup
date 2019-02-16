%Extract depth and albedo for left view in middlebury testset
clear;
close all;

%SDK
addpath('./MatlabSDK');

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

%Select dataset
for didx = 2:11

    %Clear all old data
    clearvars -except dsets didx
    close all;
    
    %Print
    fprintf('\n###########################################################\n')
    fprintf('#################### Processing [%d/%d] ##################\n', 1, length(dsets))
    fprintf('###########################################################\n')
    
    %Dataset
    dataset = sprintf('./%s-perfect', dsets{didx});
    
    %Calib
    [cam0, cam1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax ] = getcalib( dsets{didx} );
    f = cam0(1);

    %Load the amplitude image
    %amplitude_method = [];
    amplitude_method = 'koltun2013';
    %amplitude_method = 'bell2014';
    if isempty(amplitude_method)
        amplitude_fn = [dataset,'/im0.png'];
        I = im2double(imread(amplitude_fn));
    elseif strcmp(amplitude_method, 'koltun2013')
        amplitude_fn = sprintf('%s/%s_albedo_koltun2013.png', dataset, dsets{didx}  );
        I = im2double(imread(amplitude_fn));
    elseif strcmp(amplitude_method, 'bell2014')
        amplitude_fn = sprintf('%s/%s_albedo_bell2014.png', dataset, dsets{didx}  );
        I = im2double(imread(amplitude_fn));
    end

    %Amplitude
    A = rgb2gray(I);

    %Multiply
    %A = min(A * 1.3, 1); 

    d = readpfm([dataset,'/disp0.pfm']);
    Z = baseline * f ./ (d + doffs); %mm
    zlim =  baseline * f ./ ([vmax,vmin] + doffs); 
    mask = Z < zlim(1) | Z > zlim(2);
    falloff = 1./((Z/1000).^2);

    %Computer falloff
    %ds_factor = 0.18;
    ds_factor = 0.089;

    %Downsampling factor
    d = imresize(d, ds_factor,'nearest');
    Z = imresize(Z, ds_factor,'nearest');
    mask = imresize(mask, ds_factor,'nearest');
    A = imresize(A, [size(d,1), size(d,2)],'bilinear');
    falloff = imresize(falloff, ds_factor,'bilinear');

    %Maximum of falloff
    fprintf( 'Maximum falloff %2.2f, Ratio %2.1f', max(falloff(:)), 1/max(falloff(:))  )

    %Diff
    Z = Z/1000;
    zlim = zlim/1000;

    %Resize
    figure(), imshow(A), title('Amplitude');
    figure(), imagesc(falloff), axis image, colorbar, colormap gray, title('Falloff');
    figure(), imagesc(falloff .* A), axis image, colorbar, colormap gray, title('Falloff Amplitude');
    figure(), imagesc(Z,[zlim(1),zlim(2)]), axis image, colormap hsv, colorbar, title('Depth');
    figure(), imagesc(double(mask)), axis image, colorbar, title('Mask');

    %Save the dataset
    save(sprintf('%s_small_falloff.mat', dsets{didx}), 'Z','A', 'falloff', 'mask', 'zlim', 'amplitude_method');
    
end
