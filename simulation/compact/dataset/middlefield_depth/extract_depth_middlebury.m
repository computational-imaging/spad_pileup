%Extract depth and albedo for left view in middlebury testset
clear;
close all;

%SDK
addpath('./MatlabSDK');

%Dataset
dataset = './Motorcycle-perfect';

%Calib 2014
cam0=[3979.911 0 1244.772; 0 3979.911 1019.507; 0 0 1];
cam1=[3979.911 0 1369.115; 0 3979.911 1019.507; 0 0 1];
f = cam0(1);
doffs=124.343;
baseline=193.001;
width=2964;
height=2000;
ndisp=270;
isint=0;
vmin=23;
vmax=245;
dyavg=0;
dymax=0;

I = imread([dataset,'/im0.png']);
I = double(I)/255;

d = readpfm([dataset,'/disp0.pfm']);
Z = baseline * f ./ (d + doffs); %mm
zlim =  baseline * f ./ ([vmax,vmin] + doffs); 
mask = Z < zlim(1) | Z > zlim(2);

%Downsampling
%d = imresize(d, 0.18,'nearest');
%Z = imresize(Z,0.18,'nearest');
%I = imresize(I,0.18,'nearest');
%mask = imresize(mask,0.18,'nearest');

d = imresize(d, 0.089,'nearest');
Z = imresize(Z,0.089,'nearest');
I = imresize(I,0.089,'nearest');
mask = imresize(mask,0.089,'nearest');

%Diff
Z = Z/1000;
zlim = zlim/1000;
A = rgb2gray(I);

%Resize
figure(), imshow(A), title('Amplitude');
figure(), imagesc(Z,[zlim(1),zlim(2)]), axis equal, colormap hsv, colorbar;
figure(), imagesc(double(mask)), colorbar;

%Save the dataset
save('motorcycle_small.mat', 'Z','A', 'mask', 'zlim');
