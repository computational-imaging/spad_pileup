function [ Pout] = proxOp(Pi,H2,v,pars)

Pcur = Pi;
tol = pars.tol;
iters = pars.iterations;
options.Method = 'cg';
options.optTol = 1e-12;
options.progTol = 1e-12; 
options.maxIter = 400;
options.Display = 'off';
options.bbType = 1;
options.c1 = .1;%.1e-9;
options.c2 = .9;
   % options.qnUpdate = 1;
%    options.LS_type = 0;
%    options.LS_multi = 1;
   % options.DerivativeCheck = 'on';

for i = 1:iters
    
    Pnext = minFunc(@myFun,Pcur,options,H2,v,pars);
    if(norm(Pnext-Pcur)<tol)
       break;
   end
   Pcur = Pnext;
   v = Pnext;
end

Pout = Pnext;
end

