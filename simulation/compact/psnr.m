function [ PSNR ] = psnr( I, reference, psnr_pad )

    %No padding by default
    if nargin < 3
        psnr_pad = 0;
    end

    %Compute PSNR:
    I_diff = reference(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad, :) - ...
                     I(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad, :);
    MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
    if MSE > eps
        PSNR = 10*log10(1/MSE);
    else
        PSNR = Inf;
    end   

end

