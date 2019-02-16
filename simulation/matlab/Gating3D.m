% This code takes the shot noise into account, dark counts,
% and jitter

clear;
%figure;
tic;

load('Data\cornell.mat');
r = im2double(cornell.indirect);

gateWindow = 5;
QE = .30;
I = 10;
reps = 5000;%300;
jitter = 0.001;

% dim1 = 108:148;  %corner
% dim2 = 108:148;
% L = 150; start = 75;

% dim1 = 70:226;   %bunny
% dim2 = 60:246;
% L = 125; start = 15;

dim1 = 80:250;   %cornell
dim2 = 30:230;
L = 125; start = 16;
t = start:L;

for i = 1:256
    for j = 1:256
        r(:,i,j) = r(:,i,j)/sum(r(:,i,j));
    end
end
LL = size(t);
H = zeros(size(r));
GWs = size(start:L,2)/gateWindow;
for re = 1:round(reps/GWs)
    disp([num2str(100*re/round(reps/GWs)) '%']);
    for j = dim1
       for k = dim2
            Np = [zeros(15,1); poissrnd(I*r(t,j,k))];  %number of photons
            fail = [zeros(1,15) rand(LL)];
            for w = 1:GWs
                for i = (start+gateWindow*(w-1)):(start-1+gateWindow*w)
                    St = (start+gateWindow*(w-1));
                    Lt = (start-1+gateWindow*w);
                    if(r(i,j,k) > 0)
                        if(fail(i) > (1-QE)^Np(i))
                            
                            jit = round(jitter*randn());
                            if((i+jit)>=St && (i+jit)<= Lt)
                                H(i+jit,j,k) = H(i+jit,j,k)+1;
                                
                                break;
                            end
                        end
                    end
                end
            end
        end
    end
end


P = H;


toc;


data.P = P;
data.H = H;
% figure;
%save('cornell_indirect_5000R_125L_10I_Q30_GW5.mat','data')



