%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Quantile regression example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

engel = dlmread('../class-03/engel.txt')/100;

% 1st column: household income, 2nd column: food expenditures

X = engel(:,1);
Y = engel(:,2);
n = numel(Y);

% non-smooth formulation
tau = 2/3;

cvx_begin quiet
variable bet0
variable bet
minimize sum((1 - tau) * max(-(Y - X * bet - bet0), 0) + tau * max(0, (Y - X * bet - bet0)))
cvx_end

cvx_nonsmooth = cvx_optval;

% smooth formulation with ineq constraints
Z = [ones(n,1) X];

cvx_begin quiet
variable u(n)
variable v(n)
variable alph(2)
minimize tau*sum(u) + (1 - tau)*sum(v)
subject to
Y - Z * alph - u + v == zeros(n,1)
u >= 0
v >= 0
cvx_end

% dual problem
cvx_begin quiet
variable eta(n)
maximize dot(Y, eta)
subject to
Z' * eta == 0
eta <= tau
eta >= tau - 1
cvx_end

E = find(eta < tau -1E-4 & eta > tau-1 + 1E-4);

alph_dual = Z(E,:) \ Y(E,:);

% have cvx find the dual variables 

cvx_begin quiet
variable u(n)
variable v(n)
variable alph(2)
dual variable lambda
dual variable mu_u
dual variable mu_v
minimize tau*sum(u) + (1 - tau)*sum(v)
subject to
lambda: Y - Z * alph - u + v == zeros(n,1)
mu_u: u >= 0
mu_v: v >= 0
cvx_end

norm(-lambda - eta) % numerically the same

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% isotonic regression example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 50;
x = sort(rand(n, 1));

y = sqrt(x) + 0.1 * randn(n, 1);

plot(x, y, '*')

%%%
D = zeros(n-1, n);

for i = 1:n-1
    D(i,i) = -1;
    D(i,i+1) = +1;
end

%%% dual optimization problem via gradient descent

Dminustop = @(z) [z(1); diff(z); -z(n-1)];
grad = @(mu) -diff((Dminustop(mu) - y));
L = normest(D).^2;% storage of D actually not needed

maxiter = 10000;
tol = 1E-8;
function_values = zeros(maxiter, 1);
iter = 0;
mucur = zeros(n-1, 1);

while iter < maxiter
    
   munew = mucur - 1/L * grad(mucur);
   munew = munew .* (munew >= 0);
   
   if norm(munew - mucur) < tol
      break; 
   end
   
   mucur = munew;
   
   iter = iter + 1;
   function_values(iter) = 0.5 * norm(y - Dminustop(mucur))^2;
    
    
end



%%%

cvx_begin
variable f(n)
dual variable mucvx
minimize 0.5*sum((y - f).^2)
subject to
mucvx: D * f >= 0
cvx_end

figure 
hold on
plot(x, y, '*r')
plot(x, f, '-*')



norm(f - (D' * mucur + y))
norm(mucur - mucvx)

% check duality gap: 

0.5 * norm(D' * mucur)^2 - cvx_optval

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
