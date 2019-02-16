% This code creates a histogram of photon detections, and then creates the
% Coates and MAP solutions.


clear;
reps = 10000; % Number of shots

L = 100;
t = linspace(0,1,L);

% Select in incident photon function
%r = ones(size(t));
r = ones(1,L);
r = exp(-(t-.5).^2/.05);
%r = (t-1).*t.*(1-2.5*t).^2;
%r = exp(-2*t);

%load('NMF_data2.mat');
%r = NMF.indirect_Corner(1:100,5)';

r = r/sum(r);
R = cumsum(r);

QE = .30; %Quantum efficiency
I = 10;   %Incident number of photons
jScale = .25; % percentage of jitter relative to histogram
jitter = jScale*100/2.355;

DC = 0;  % dark counts per exposure
amb = 0; % ambient light

h = zeros(size(t)); %expected histogram probabilities
h(1) = 1-exp(-r(1)*I*QE);
for i = 2:size(r,2)
    h(i) = 1 - ( sum(h(1:i-1)) + (1-sum(h(1:i-1)))*exp(-r(i)*I*QE)); 
end

%Create expected histogram probability with jitter
g = normpdf(-(L-1):(L-1),0,jitter);
LL = size(g,2);
g = g/sum(g);
c = conv(h,g);
c = c((L):(2*L-1));


trials = 1; % number of trials
for tt = 1:trials
    H(tt,:) = zeros(size(t)); %Histogram counts
    
    for re = 1:reps
        Np = poissrnd(I*r) + poissrnd(DC+amb);  %number of incoming photons
        fail = rand(size(t));    % failure random number

        for i = 1:size(t,2)
            if(fail(i) > (1-QE)^Np(i))
               jit = round(jitter*randn()); %create jitter
               if((i+jit)>0 && (i+jit)<= L) %make sure it is within bounds
                    H(tt,i+jit) = H(tt,i+jit)+1;
               end
               break;
            end
        end
    end
    
    H2 = H(tt,:);  % Store current histogram
    %H2 = c*reps;   %Perfect histogram, for testing purposes

    para = 0.5;   %Larger means closer to original, lower means smoother
    
    Parameters.N = reps;
    Parameters.QE = QE;
    Parameters.lambda = para;
    Parameters.jitter = jitter;
    Parameters.mult = 0;
    Parameters.iterations = 5;
    Parameters.tol = 1e-10;
    Parameters.meu = 1-para;
    H2 = proxOp(H2',H2',H2',Parameters);
    
    Hc = cumsum(H2);
    
    P(tt,:) = zeros(size(r));   % Coates solution
    P(tt,1) = 1/(-QE)*log(1-H2(1)/reps);
    for i = 2:L
        P(tt,i) = 1/(-QE)*log((reps-Hc(i))/(reps-Hc(i-1)));
    end
    
    % Find MAP solution
    lambda = 1e12;   % Proximity term
    meu = 3e11;      % Smoothness term
    % These terms should not add up to higher than one, (1-meu-lambda)
    % is the weight given to the probability of the histograms matching
    
    Pi = P(tt,:);
    Parameters.lambda = lambda;
    Parameters.meu = meu;
    Parameters.tol = 1e-8;
    Parameters.mult = 1;
    Parameters.iterations = 10;
    Pm(tt,:) = proxOp(Pi',H2,Pi',Parameters);
end

zz = 1.645; %90 percent     %zz = 1.96;  %95 percent
zz = 2.24; %95 percent(combined) -> sqrt(.95) percent

J = -getHess(H2,P',Parameters);
Ji = inv(J);
for ind = 1:L
    rup(ind) = Pm(ind)+zz*sqrt(abs(Ji(ind,ind)));
    rlo(ind) = Pm(ind)-zz*sqrt(abs(Ji(ind,ind)));
end
Jup = -getHess(H2,rup',Parameters);
Jlo = -getHess(H2,rlo',Parameters);
Jui = inv(Jup);
Jli = inv(Jlo);
for ind = 1:L
    rup(ind) = rup(ind)+zz*sqrt(abs(Jui(ind,ind)));
    rlo(ind) = rlo(ind)-zz*sqrt(abs(Jli(ind,ind)));
end

subplot(211);
plot(t,r,t,Pm/sum(Pm),t,rup/sum(rup),t,rlo/sum(rlo),t,P/sum(P))
legend('Actual','MAP','MAP Upper','MAP Lower','Coates');
xlabel('time');ylabel('r_t');
title('Incoming photon distribution')
xlim([-0.02 1.02]);

subplot(212);
plot(t,h,t,c,t,(H/reps),t,H2/reps);
legend('Expected w/o Jitter','Expected w/ Jitter','Measured',...
    'Measured+Smoothed'); 
title('Histogram detection probability')
xlabel('time');ylabel('p_t');
xlim([-0.02 1.02]);




