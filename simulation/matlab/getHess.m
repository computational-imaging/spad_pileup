% Fisher matrix under regularity conditions


function [F] = getHess(h,r,pars)
r2 = r*pars.QE;
L = size(h,1);
F = zeros(L,L);

R = cumsum(r2);
P(1) = 1-exp(-R(1));
P(2:L) = exp(-R(1:(L-1))).*(1-exp(-r2(2:L)));

g = normpdf(-(L-1):(L-1),0,pars.jitter);
LL = size(g,2);
g = g/sum(g);
c = conv(P,g);
c = c((L):(2*L-1));

for n = 1:L
    %%
    dPn = zeros(1,L);
    dPn(n) = exp(-R(n));
    for k = (n+1):L
        dPn(k) = exp(-R(k))- exp(-R(k-1));
    end
    dPcn = [zeros(1,L) dPn];
    dcn = conv(dPcn,g);
    dcn = dcn((2*L):(3*L-1));

    for j = 1:L
%% 
        dPj = zeros(1,L);
        dPj(j) = exp(-R(j));
        for k = (j+1):L
            dPj(k) = exp(-R(k))- exp(-R(k-1));
        end
        dPcj = [zeros(1,L) dPj];
        dcj = conv(dPcj,g);
        dcj = dcj((2*L):(3*L-1));
%%
        dP2 = zeros(1,L);
        dP2(max(n,j)) = exp(-R(max(n,j)));
        for k = (max(n,j)+1):L
            dP2(k) = exp(-R(k))- exp(-R(k-1));
        end
        dPc2 = [zeros(1,L) dP2];
        dc2 = conv(dPc2,g);
        dc2 = dc2((2*L):(3*L-1));

        for i = 1:L     
           if(c(j) > 0 && c(i) > 0 && c(i) < 1)
               F(n,j) = F(n,j) + h(i)*(-dc2(i)*c(i)-dcj(i)*dcn(i))/(c(i))^2;
               F(n,j) = F(n,j) - ...
                (pars.N-h(i))*(-dc2(i)*(1-c(i))+dcn(i)*dcj(i))/(1-c(i))^2; 
           end
        end
        if(j==n && j ==1)
             F(n,j) = F(n,j)-pars.meu;
        end
        if(j==n && j == L)
             F(n,j) = F(n,j)+pars.meu;
        end
        if( abs(j-n) == 1 )
            F(n,j) = F(n,j)+pars.meu*abs(j-n);
        end
end
end
    
F(find(isnan(F))) = 0;
%F = -F;

end