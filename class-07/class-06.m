%%% Example 1: Lasso via Variable Augmentation and Projected Gradient Descent
n = 1000;
d = 500;
s = 10;
X = randn(n, d);
betastar = zeros(d, 1);
betastar(1:s) = rand(s, 1);
sigma = 0.2;
epsilons = randn(n, 1);
y = X * betastar + sigma * epsilons;


Z = [X  -X];

lambda = sigma * sqrt(2 * log(d) * n); 
obj = @(theta) sum((y - Z * theta).^2) + lambda * sum(theta);
grad = @(theta) -2 * Z' * (y - Z * theta) + lambda * ones(2*d, 1);

L = 2 * norm(Z).^2;

% initial solution
maxiter = 1000;
obj_values = zeros(maxiter, 1);
thetacur = zeros(2*d, 1); 

iter = 0;
tol = 1E-8;

while iter < maxiter
    
    gr = grad(thetacur);
    
    z = (thetacur - 1/L * gr);
    zplus = z .* (z >= 0); % truncation at zero 
      
    if norm(zplus - thetacur) < tol
       flag = true;
    end
        
    thetacur = zplus;
    iter = iter + 1;
    obj_values(iter) = obj(thetacur);
    
    if flag
        break;
    end
    
end

plot(1:iter, obj_values(1:iter), '-*')

% check that only positive or negative part are "on" (i.e., non-zero), 
% but not both
dot(thetacur(1:d), thetacur((d+1):end))

% subtract positive and negative parts to convert to the original set 
% of variables from the augmented set of variables
betahat = thetacur(1:d) - thetacur((d+1):end);
sum(abs(betahat) > 0)

% check results with cvx
cvx_begin
variable bet(d)
minimize sum((y - X * bet).^2) + lambda * norm(bet, 1)
cvx_end


norm(bet - betahat)


%%% Example 2: Group Lasso via Proximal Gradient Descent
clear all

d = 1000; % 1000 variables altogether 
K = 20; % 20 groups 
m = d/K; % number of variables per group
% for simplicity groups are given by
% {1,...,50},{51,...,100},...{951,...,1000}.

reg = @(x) sum(sqrt(sum(reshape(x, [m K]).^2, 1))); % function implementating group lasso regularizer
prox = @(xsub, lambda) max((1 - lambda/norm(xsub)), 0) * xsub; % prox operator for this specific regularizer 

% data generation
s = 5; % suppose 5 out of 20 groups have non-zero coefficients   
rprm = randperm(K);
supp = rprm(1:s);

beta = zeros(d, 1);

for j=1:numel(supp)
    beta(m * (supp(j)-1)+1:supp(j)*m) = randn(m,1);
end

beta = beta/norm(beta);

n = 1500;
frac_labelflip = 0.025;
xi = 2*(rand(n, 1) > frac_labelflip) - 1;
X = randn(n, d);
y = xi .* sign(X * beta);

%%% define gradient etc.

f = @(theta) theta(1) * ones(n, 1) + X * theta(2:end);
foo = @(f) sum(log(1 + exp(-y .* f)));
grad = @(theta) [ones(n, 1) X]' * (-y ./ (1 + exp(y .* f(theta))));
L = 0.25 * (norm([ones(n, 1) X]).^2);


maxiter = 2000;
function_values = zeros(maxiter, 1);
function_values_acc = zeros(maxiter, 1); 

%%% (oracle) choice of the regularization parameter 
gradbeta = grad([0; beta]);
gradbeta = gradbeta(2:end);
lambda = max(sqrt(sum(reshape(gradbeta, [m K]).^2, 1))); % from statistical theory (Negahban et al, Statistical Science)

thetacur = zeros(d+1, 1); 

iter = 0;
tol = 1E-6;
flag = false;

while iter < maxiter
    gr = grad(thetacur);
    
    zprox = zeros(d+1, 1);
    z = thetacur - 1/L * gr;
    
    zprox(1) = z(1);
    
    for j=1:K
        ix_in_groupj = 1 + (m * (j-1)+1:j*m);
        zprox(ix_in_groupj) = prox(z(ix_in_groupj), lambda/L);
    end

      
    if norm(zprox - thetacur) < tol % stopping criterion
       norm(zprox - thetacur)
       flag = true;
    end
        
    thetacur = zprox;
    iter = iter + 1;
    function_values(iter) = foo(f(thetacur)) + lambda * reg(thetacur(2 : end)); % note that the intercept is not regularized
    
    if flag
        break;
    end
    
end


% code with Nesterov acceleration

thetacur = zeros(d+1, 1); 
thetaprev = thetacur;

iter = 0;
tol = 1E-6;
flag = false;
  
 while iter < maxiter
     
     aux = thetacur + ((iter - 1) / (iter + 2)) * (thetacur - thetaprev);
     thetaprev = thetacur;
     
     gr = grad(aux);
     
     zprox = zeros(d+1, 1);
     z = aux - 1/L * gr;
     
     zprox(1) = z(1);
     
     for j=1:K
         ix_in_groupj = 1 + (m * (j-1)+1:j*m);
         zprox(ix_in_groupj) = prox(z(ix_in_groupj), lambda/L);
     end
     
     
     if norm(zprox - thetacur) < tol % stopping criterion
         norm(zprox - thetacur)
         flag = true;
     end
     
     thetacur = zprox;
     iter = iter + 1;
     function_values_acc(iter) = foo(f(thetacur)) + lambda * reg(thetacur(2 : end)); % note that the intercept is not regularized
     
     if flag
         break;
     end
    
 end
   
  
figure
hold on
plot(function_values(1:30,:))
plot(function_values_acc(1:30,:))
