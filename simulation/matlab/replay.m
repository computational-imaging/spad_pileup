load('Data\cornell.mat');
r = im2double(cornell.indirect);

load('Cornell\cornell_indirect_1000R_125L_40I.mat');
P2 = data.P;
H2 = data.H;

load('GatedData\cornell_indirect_5000R_125L_10I_Q30_GW5.mat')
P = data.P;
H = data.H;

load('cornell_indirect_1000R_125L_40I_MAP.mat.mat')
P3 = data.P;
H3 = data.H;

%Normalize
for i = 1:256
    for j = 1:256
        r(:,i,j) = r(:,i,j)/sum(r(:,i,j));
    end
end

% For scaling
rm = max(max(max(r)));
pm = max(max(max(P)));
pm2 = max(max(max(P2)));
pm3 = max(max(max(P3)));

dim1 = 108:148;  %corner
dim2 = 108:148;
L = 150; start = 60;

% dim1 = 70:226;   %bunny
% dim2 = 60:246;
% L = 125; start = 15;

dim1 = 80:250;   %cornell
dim2 = 30:230;
L = 125; start = 15;

t = start:L;

% For error comparison
e1 = 0;
e2 = 0;
e3 = 0;
for i = dim1
    for j = dim2
        e1 = e1+norm(r(:,i,j)-P(:,i,j)/sum(P(:,i,j)));
        e2 = e2+norm(r(:,i,j)-P2(:,i,j)/sum(P2(:,i,j)));
        e3 = e3+norm(r(:,i,j)-P3(:,i,j)/sum(P3(:,i,j)));
    end
end

%Play videos
for i = start:L
    subplot(2,1,1)
    imshow(squeeze(r(i,dim1,dim2)/rm));
    title(['Actual: Frame ' num2str(i)]);
    subplot(2,1,2)
    imshow(squeeze(P(i,dim1,dim2)/pm));
    title(['Gated: Frame ' num2str(i)]);
    pause(1/30);

end

for i = start:L
    subplot(2,1,1)
    imshow(squeeze(P2(i,dim1,dim2)/pm2));
    title(['Coates: Frame ' num2str(i)]);
    subplot(2,1,2)
    imshow(squeeze(P3(i,dim1,dim2)/pm3));
    title(['MAP: Frame ' num2str(i)]);
    pause(1/30);
end

load('Bounds\bounds_cornell_indirect_1000R_125L_40I_m99.mat')
bounds1 = bounds;
bm1 = max(max(max(bounds1)));

load('Bounds\bounds_cornell_indirect_1000R_125L_40I_m0.mat')
bounds2 = bounds;
bm2 = max(max(max(bounds2)));

% Play video of bounds
for i = start:L
    subplot(1,2,1)
    imshow(squeeze(bounds1(i,dim1,dim2)/bm2));
    title(['MAP Bounds: Frame ' num2str(i)]);
    subplot(1,2,2)
    %imshow(ind2rgb(squeeze(round(bounds(i,dim1,dim2)/bm2)*256),jet(size(t,2))),[]);
    imshow(squeeze(bounds2(i,dim1,dim2))/bm2);
    title(['Coates Bounds: Frame ' num2str(i)]);
    pause(1/30);
end



