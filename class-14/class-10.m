%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1: ADMM for fused lasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% same data set as for homework 4
y = dlmread('cgh.txt');
n = numel(y);

plot(y, '-*')

D = zeros(n-1, n);

for i = 1:n-1
    D(i,i) = -1;
    D(i,i+1) = +1;
end

%%% ADMM

w = zeros(n- 1, 1);
z = zeros(n-1, 1);
f = zeros(n, 1);

rho = 100;

M = (rho * (D' * D) + eye(n));

L = chol(M); %norm(L' * L - M)

%%%
maxiter = 1000;
lambda = 5;
soft = @(x, t) (abs(x) >= t) .* (x - sign(x) * t);
diffadj = @(x) [-x(1); -diff(x); x(end)]; % for efficient implementation of D' * vector

res_primal = zeros(maxiter, 1);
res_dual = zeros(maxiter, 1);
iter = 0;
t0 = tic;
while iter < maxiter
   
    % update f
    rhs = y + (rho * diffadj(z - w));
    
    f = L \ (L' \ rhs);
    
    % update z
    
    Df = diff(f);
    
    zold = z;
    z = soft(Df + w, lambda/rho);
    
    % update w 
    
    w = w + Df - z;
    
    iter = iter + 1;
    
    %%% monitor convergence
    res_primal(iter) = norm(Df - z); 
    res_dual(iter) = rho * norm(diffadj(z - zold));

end
t1 = toc(t0);

figure 
hold on
plot(y, '-*')
plot(f, '-r')

0.5 * norm(y - f).^2 + lambda * norm(diff(f), 1)

cvx_begin
variable fcvx(n)
minimize 0.5* sum((y - fcvx).^2) + lambda*norm(D*fcvx, 1)
cvx_end

norm(fcvx - f, Inf)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2: Quantile Regression via Dykstra's algorithm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 1000;
d = 20;
X = randn(n, d);
beta = rand(d, 1);
err = 0.2 * trnd(2, 1000, 1);
y = X * beta + err;

beta_ls = X \ y;

%%% QR estimator
tau = 0.5;

[U,~,~] = svd(X, 'econ');

%%% dykstra's algorithm
eta = zeros(n, 1);
maxiter = 1000;
tol = 1E-5;
iter = 0;

while iter < maxiter
    
    toproj = (eta - (-y));
    
    w = zeros(n, 1);
    
    while true
        x = (toproj - w) - U * (U' * (toproj - w));
        z = x + w;
        z(z > tau) = tau;
        z(z < (tau-1)) = tau-1;
        w = w + x - z;
        if norm(x - z) < tol
           break; 
        end
    end
    
    eta = z;
    iter
    iter = iter + 1;
    
end

cvx_begin
variable etacvx(n)
minimize dot(-y,etacvx)
subject to
X' * etacvx == 0
etacvx <= tau * ones(n, 1)
etacvx >= (tau-1) * ones(n,1) 
cvx_end

dot(-y,x)
%norm(x - etacvx, 'Inf')




