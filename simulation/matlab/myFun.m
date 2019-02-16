function [f,g,H] = myFun(r,h,v,pars)

f =  -getProbsProx(h,r,v,pars);   %function value

g = -gradProx(h,r,v,pars);    %gradient

%H = getHess(h,r,N,L,QE,meu,jitter);


end
