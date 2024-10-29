% Name of the NNLS solver:
%
% Primal-Dual interior point method. 
%
% Input
%
% A --- coefficient matrix,
%
% b --- observation vector,
%
% options_general --- specification of options as returned by 'initopt_general',
%
% options_specific --- specification of options as returned by 'opt_primaldual'.

function [out] = primaldual(A, b, eps)

%%%
[~, d] = size(A);
AtA = A'*A;
Atb = A' * b;

diagind = sub2ind([d,d], 1:d, 1:d)';
AtAdiag = diag(AtA);

% 
grad = @(x) (AtA * x - Atb);
  

%%% 
f = @(x) norm(A * x - b)^2;
 
%%% functions to evaluate the (inverse) 'surrogate' duality gap 
%%% and the residual of the KKT system. 
eta = @(x, mu) d/(x' * mu);
residual = @(x, mu) sqrt(norm(grad(x) - mu)^2 + norm(x .* mu - 1/eta(x, mu))^2);

% Initialization
alpha = 0.01; beta=0.95; % parameters for the stepsize selection
FLAG = 1; 
x = ones(d,1);
mu = 1./x;
r = residual(x, mu);
nu = 10;
%%%
gradf = grad(x);    
% 


 while FLAG
     
     etaVal = nu * eta(x, mu); 
     
    
     RHS = -gradf + 1/etaVal./x;
     D = mu./x;
     % compute the Newton descent direction
     AtA(diagind) = (D + AtAdiag);
     descentX = AtA \ RHS;
         
     AtA(diagind) = AtAdiag;
         
     
     
     descentMu = -mu + 1/etaVal./x - (mu./x).*descentX;
     
     % get the stepsize with the Armijo rule
     xold = x;
     muold = mu;
     rold = r;
     FLAGSTEP = 1;
     
     ix1 = find(descentMu < 0);
     if ~isempty(ix1)
         min1 = min(1, min(-mu(ix1)./descentMu(ix1)));
     else
         min1 = 1;
     end
     
     ix2 = find(descentX < 0);
     
     if ~isempty(ix2)
         min2 = min(1, min(-x(ix2)./descentX(ix2)));
     else
         min2 = 1;
     end
      
     t = 0.99*min(min1,min2);
    
     while FLAGSTEP
        x = xold + t * descentX;
        mu = muold + t*descentMu;
        %%% compute new residuals
        r = residual(x, mu);
        if( r > (1-alpha*t)*rold)
        t = beta * t;
        else
            FLAGSTEP = 0;
        end  
     end
     %%% udpate gradient
     gradf = grad(x);
     %%%
     
     

     %%% 
     if(1/eta(x, mu) < eps && r < eps) 
         FLAG =0;
     end
 end

out.xopt = x;
out.err = f(x);


end










  

