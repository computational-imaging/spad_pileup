%Synthetic data test for "Primal dual cross-channel deconvolution"
clear;
close all;

load('./bright_img.mat');
%psf = bright_img(128-15:128+15,10);
%psf = mean(bright_img(128-15:128+15,:),2);
psf = mean(bright_img(128-20:128+20,:),2);
psf = reshape(psf, [1,1,length(psf(:))]);

% normalize
psf = squeeze( psf / sum(psf(:)) );

%Plot psf
figure(),
plot(psf);
title('Temporal PSF of the SPAD')

%Mixure model
xtimes = (0:length(psf(:))-1)';
mixtures = 'gauss3';
fitoptions = fitoptions(mixtures, 'TolX', 1e-8, 'TolFun', 1e-10, 'MaxIter', 1000, 'MaxFunEvals', 10000);
f = fit(xtimes,psf,mixtures, fitoptions);
f

xtimesint = linspace(xtimes(1), xtimes(end), 500);
gmix = @(x,a1,b1,c1,a2,b2,c2,a3,b3,c3) a1*exp(-((x-b1)./c1).^2) + a2*exp(-((x-b2)./c2).^2) + a3*exp(-((x-b3)./c3).^2);
ffit = gmix(xtimesint, f.a1, f.b1, f.c1, f.a2, f.b2, f.c2, f.a3, f.b3, f.c3 );

figure(),
hold on;
plot(xtimes, psf(:), 'ob');
plot(xtimesint, ffit, '-r');
hold off;
title('Finalized fit');



