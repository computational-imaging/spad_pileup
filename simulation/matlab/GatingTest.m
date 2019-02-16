
clear;
tic;
count = 1;
gateWindow = 5;  %Length of gating window

reps = 5000;  % Number of shots

L = 100;
t = linspace(0,1,L);


% Select an incident photon function
%r = ones(size(t));
r = ones(1,L);
r = exp(-(t-.5).^2/.05);
r = (t-1).*t.*(1-2.5*t).^2;
%r = exp(-t);
%load('NMF_data2.mat');
%r = NMF.all_Cornell(1:100,im)';
%r = NMF.indirect_Corner(1:100,im)';
r = r/sum(r);
R = cumsum(r);

QE = .30; %Quantum efficiency
I = 10;   %Incident number of photons
jScale = .000001; % percentage of jitter relative to histogram
jitter = jScale*100/2.355;

DC = 0;%.001;  % dark counts per exposure
amb = 0;%.002;

trials = 1;

h(1) = 1-exp(-r(1)*I*QE);
for i = 2:size(r,2)
    h(i) = 1 - ( sum(h(1:i-1)) + (1-sum(h(1:i-1)))*exp(-r(i)*I*QE));
end

GWs = L/gateWindow;   % Number of total windows

for tt = 1:trials
    H(tt,:) = zeros(size(t));
    for re = 1:round(reps/GWs)   % Split shots into each window equally
        Np = poissrnd(I*r) + poissrnd(DC+amb);  %number of photons
        fail = rand(size(t));  % chance of failure

        for i = 1:GWs   % Loop through each window
            for k = (1+gateWindow*(i-1)):(gateWindow*i);
                if(fail(k) > (1-QE)^Np(k))
                   jit = round(jitter*randn());
                   if((k+jit)>0 && (k+jit)<= L)
                        H(tt,k+jit) = H(tt,k+jit)+1;
                        break;
                   end
                end
            end
        end

    end
    H2 = H(tt,:);
end

% Now do Coates
for tt = 1:trials
    H(tt,:) = zeros(size(t));
    for re = 1:round(reps)   % Split shots into each window equally
        Np = poissrnd(I*r) + poissrnd(DC+amb);  %number of photons
        fail = rand(size(t));  % chance of failure
             for k = 1:L
                if(fail(k) > (1-QE)^Np(k))
                   jit = round(jitter*randn());
                   if((k+jit)>0 && (k+jit)<= L)
                        H(tt,k+jit) = H(tt,k+jit)+1;
                        break;
                   end
                end
            end
    end
    H3 = H(tt,:);

    Hc = cumsum(H3);
    P(tt,:) = zeros(size(r));   % Coates solution
    P(tt,1) = 1/(-QE)*log(1-H3(1)/reps);
    for i = 2:L
        P(tt,i) = 1/(-QE)*log((reps-Hc(i))/(reps-Hc(i-1)));
    end

end

plot(1:100,r/max(r),1:100,H2/sum(H2*max(r)),1:100,P/sum(P*max(r)),'k');
legend('Actual','Gating','Coates'); title('Normalized Intensities');
xlabel('k');ylabel('r_k');
toc;



