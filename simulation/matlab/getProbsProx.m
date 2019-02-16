function [ pp ] = getProbsProx(H,r,v,pars)
pp = 0;
L = size(H,1);

R = cumsum(r);
P(1) = 1-exp(-r(1)*pars.QE);
P(2:L) = exp(-R(1:(L-1))*pars.QE).*(1-exp(-r(2:L)*pars.QE));

% Create the convolution of p and jitter function
g = normpdf(-(L-1):(L-1),0,pars.jitter);
LL = size(g,2);
g = g/sum(g);
c = conv(P,g);
c = c((L):(2*L-1));

sg = sign(r);
for i = 1:L
    
    if(sg(i) < 0)
       pp = -Inf;
       break;
    end
    
    if(sg(i) <= 0)   % Only allow positive r values
        p = 0;
    else
        p = c(i);
    end
    
    if(H(i)~=p || H(i)~=0)      % Make sure double zeros not counted
        pp = pp+H(i)*log10(p)+log10(1-p)*(pars.N-H(i));
    end
    

end
if(sum(H) > pars.N)
    pp = -Inf;
end
pp = pars.mult*pp ...   % This is the likelihood term
    - pars.lambda*sum((r-v).^2) ...   % This is the proximity term
- pars.meu*sum(diff(r).^2);           % This is the smoothness term
end




