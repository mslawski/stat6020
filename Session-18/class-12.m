%
% MM-algorithms
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 1: LAD regression 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 1000;
d = 20;
X = randn(n, d);
betastar = rand(d, 1);
err = 0.2 * trnd(2, 1000, 1);
y = X * betastar + err;

beta_ls = X \ y;


maxiter = 1000;
fval = zeros(maxiter, 1);
iter = 0;
beta = beta_ls;
tol = 1E-5;

while iter < maxiter
    
    r = y - X*beta;
    iter = iter + 1;
    fval(iter) = sum(abs(r));
    
    if iter > 1
        if (fval(iter-1)-fval(iter))/fval(iter-1) <  tol
            break;
        end
    end
    
    w = 1./abs(r);
   
    Xw = repmat(sqrt(w), [1 d]) .* X;
    yw = sqrt(w) .* y;
    beta = (Xw \ yw);
    
end

cvx_begin
variable betacvx(d)
minimize norm(y - X*betacvx, 1)
cvx_end

%%% residuals vs. fitted --- least squares

figure
hold on
plot(X*beta_ls, y - X*beta_ls, '*')
%plot(X*beta, y - X*beta, 'rx')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 2: MCP penalty
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;
d = 50;

X = randn(n, d);
betastar = sprand(d, 1, 0.1);

sigma = 0.2;
err =  sigma * randn(n, 1);

y = X*betastar + err;

lambda = 2 * sigma * sqrt(2 * log(d)/n);

gamma = 1.2;

Omega = @(x) sum((lambda * x - (x.^2 / (2 * gamma))) .* (x <= gamma * lambda) + ...
                 0.5 * gamma * lambda^2 .* (x > gamma * lambda));
             
Omegaprime = @(x) (lambda - x/gamma) .* (x < gamma * lambda);             
                         
obj = @(x) 0.5 * mean((y - X * x).^2) + Omega(abs(x));             

w = ones(d, 1) * lambda;            
maxiter = 50;
iter = 0;
fvals = zeros(maxiter, 1);
tol = 1E-5;
betalasso = zeros(d, 1);

while iter < maxiter 

    cvx_begin quiet
    variable bet(d)
    minimize ((0.5/n) * sum((y - X * bet).^2) + sum(w .* abs(bet)))
    cvx_end
    
    if iter==1
       betalasso = bet; 
    end

    iter = iter + 1;
    fvals(iter) = obj(bet);
    w = Omegaprime(abs(bet));
    
    if iter > 1
        if (fvals(iter-1) - fvals(iter))/fvals(iter-1) < tol
           break; 
        end
    end
    

end

betaoracle = zeros(d, 1);
betaoracle(betastar > 0) = X(:,betastar > 0) \ y;

norm(bet - betaoracle) % equal to oracle estimator knowing support in advance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 3: EM-Algorithm I -- Gaussian Data with lost signs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = 2;
sigma = 1;

n = 1000;

u = mu + sigma * randn(n, 1);
x = abs(u);

mucur = 0.5;
sigmacur = sqrt(mean(x.^2) - mucur^2);

nloglik = @(mu_, sigma_) -sum(log(normpdf(x, mu_, sigma_) + normpdf(x, -mu_, sigma_)));

maxiter = 100;
tol = 1E-8;
fvals = zeros(maxiter, 1);
converged = false;
iter = 0;

while ~converged && iter < maxiter

    likpos = normpdf(x, mucur, sigmacur);
    w = likpos./(likpos + normpdf(-x, mucur, sigmacur)); 

    mucur = sum(w.*x + (1-w) .* (-x))/n;
    sigmacur = sqrt(sum(w.*(x - mucur).^2 + (1-w).*(-x - mucur).^2)/n);
    iter = iter+1;
    fvals(iter) = nloglik(mucur, sigmacur);
    
    if iter > 1
        if (fvals(iter-1) - fvals(iter))/fvals(iter-1) < tol
            converged = true;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 4: EM-Algorithm II -- mixture of Poisson
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 1000;
alpha = 0.5;
z = rand(n, 1) > alpha; % latent variables

lambda1 = 3;
lambda2 = 8;

x = (z == 0) .* poissrnd(lambda1, n, 1) + (z==1) .* poissrnd(lambda2, n, 1);

nloglik = @(lam1, lam2, alph) -sum(log(alph * poisspdf(x, lam1) + (1-alph) * poisspdf(x, lam2)));


maxiter = 1000;
tol = 1E-8;
fvals = zeros(maxiter, 1);
converged = false;
iter = 0;

thetacur =  [0.5; mean(x) - 1; mean(x) + 1]; %[0.5; mean(x) - 0.5; mean(x) + 0.5]; %[0.5; mean(x) - 2; mean(x) + 2];
%thetacur = [0.5; 5; 5];
while ~converged && iter < maxiter

    alpha_cur = thetacur(1);
    lambda1_cur = thetacur(2);
    lambda2_cur = thetacur(3);
    
    w1 = alpha_cur * poisspdf(x, lambda1_cur);
    w2 = (1 - alpha_cur) * poisspdf(x, lambda2_cur);
   
    w = w1 ./ (w1 + w2);
    
    thetacur(1) = sum(w)/n;
    thetacur(2) = sum(w .* x) / sum(w);
    thetacur(3) = sum((1-w) .* x) / sum(1-w);
    iter = iter + 1;
    fvals(iter) = nloglik(thetacur(2), thetacur(3), thetacur(1));
    
    if iter > 1
        if (fvals(iter-1) - fvals(iter))/fvals(iter-1) < tol
            converged = true;
        end
    end
end

%%% contour plot of the log-likelihood

alpha0 = 0.5;%thetacur(1);
lambda1_grid = [1:0.1:15];
lambda2_grid = [1:0.1:15];
nloglikvals = zeros(numel(lambda1_grid), numel(lambda2_grid));

for i=1:numel(lambda1_grid)
    for j=1:numel(lambda2_grid)
         nloglikvals(i,j) = nloglik(lambda1_grid(i), lambda2_grid(j), alpha0);     
    end
end
    
contour(lambda1_grid, lambda2_grid, nloglikvals, 500)
