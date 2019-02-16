function [dr] = gradHistProx(h,r,v,pars)

r2 = r*pars.QE;
L = size(r,1);

R = cumsum(r2);
P(1) = 1-exp(-R(1));
P(2:L) = exp(-R(1:(L-1))).*(1-exp(-r2(2:L)));

g = normpdf(-(L-1):(L-1),0,pars.jitter);
LL = size(g,2);
c = conv(P,g);
c = c((L):(2*L-1));

dr = zeros(size(r));

for j = 1:L
    dPv = zeros(1,L);
    dPv(j) = exp(-R(j));
    for k = (j+1):L
        dPv(k) = exp(-R(k))- exp(-R(k-1));
    end
    
    dPc = [zeros(1,L) dPv];
    dc = conv(dPc,g);
    dc = dc((2*L):(3*L-1));
    
    
    % This if for the function gradient
    for i = 1:L     
       if(c(j) > 0 && c(i) > 0 && c(i) < 1)
           dr(j) = dr(j) + h(i)*dc(i)/c(i);
           dr(j) = dr(j) - (pars.N-h(i))*dc(i)/(1-c(i)); 
       end
    end
    
    
    % This is now for the proximity term in the proximal operator
    dr(j) = pars.mult*dr(j)-pars.lambda*(r(j)-v(j));
    
    % This is for the smooth term
    if(j<L)
         dr(j) = dr(j)-pars.meu*(r(j)-r(j+1));
    end
   
    if(j>1)
         dr(j) = dr(j)-pars.meu*(r(j)-r(j-1));
    end

end

dr(find(isnan(dr))) = 0;

end