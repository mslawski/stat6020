
%%% setting up CVX

path = pwd;
cd '/Users/martinslawski/mydata/cvx/'
cvx_setup
cd(path)

%%% 


%%% [1] Simple test: 
%%% straight line fit based on the least absolute deviation criterion

x = rand(100, 1);
y = 2*x + 0.2*randn(100,1);

plot(x,y, '*')

cvx_begin
variable a(1)
variable b(1)
minimize sum(abs(y - a - b*x))
cvx_end

line([0 1], [a b + a], 'LineWidth', 2)

%%% [2] linear support vector machine & logistic regression

n = 40;
X = randn(n, 2);
X(1:(n/2),1) = X(1:(n/2),1) + 2;
X(1:(n/2),2) = X(1:(n/2),2) + 2;
X((n/2 + 1):n, 1) = X((n/2 + 1):n,1) - 2;
X((n/2 + 1):n, 2) = X((n/2 + 1):n,2) - 2;
y = [ones(n/2,1); -ones(n/2, 1)];


figure 
hold on
plot(X(y == 1,1), X(y == 1,2), '*r', 'MarkerSize', 12)
plot(X(y == -1,1), X(y == -1,2), '*b', 'MarkerSize', 12)

% fit linear SVM

lambda = 0.001;

cvx_begin
variable w(2)
variable w0(1)
minimize sum(pos(1 - y.*(X*w + w0))) + lambda * sum(w.^2)
cvx_end

% plot corresponding decision boundary
sl = -w(1)/w(2);
int = -w0(1)/w(2);
line([-4 4], [-4*sl + int, 4*sl + int], 'LineWidth', 4)

% fit logistic regression
cvx_begin
variable w(2)
variable w0(1)
minimize sum(log(1 + exp(-y.*(X*w + w0)))) + 10*lambda * sum(w.^2)
cvx_end

sl = -w(1)/w(2);
int = -w0(1)/w(2);
line([-4 4], [-4*sl + int, 4*sl + int], 'LineWidth', 4)


%%% [3] matrix completion / matrix factorization

n = 40;
m = 20;

u = randn(n, 1);
v = rand(m, 1);
X = u*v';

% reveal only 40% of the entries 
nentries = n*m;
frac = 0.4;
nreveal = n * m * frac;
rp = randperm(nentries);
S = rp(1:nreveal);
[S_i, S_j] = ind2sub([n m], S); 
Xobs = X;
Xobs(~ismember(1:nentries, S)) = NaN;

% try to complete incomplete matrix using nuclear norm minimization

cvx_begin
variable Xhat(n,m)
minimize norm_nuc(Xhat)
subject to 
Xhat(S) == Xobs(S)
cvx_end

norm(Xhat - X, 'fro') % exact agreement 

% try to complete matrix using matrix factorization and coordinate
% descent

maxiter = 100;
iter = 0;
objective = zeros(maxiter, 1);

% random init
uhat = randn(n,1);
vhat = randn(m,1);

Xzeroed = Xobs;
Xzeroed(isnan(Xzeroed)) = 0;

while iter < maxiter

    tmp_v = Xzeroed' * uhat;
    for j=1:m
        vhat(j) = tmp_v(j) / sum(uhat(S_i(S_j == j)).^2);
    end
    
    tmp_u = Xzeroed * vhat;
    for i=1:n
        uhat(i) = tmp_u(i) / sum(vhat(S_j(S_i == i)).^2);
    end
    
    iter = iter + 1;
    Xhat = uhat * vhat';
    objective(iter) = sum((Xhat(S) - X(S)).^2);
    
    if objective(iter) < 1E-10
       break; 
    end
end

objective = objective(1:iter);

norm(Xhat - X, 'fro') % almost exact agreement


%%% Extension to higher rank

n = 40;
m = 20;

u1 = randn(n, 1);
v1 = rand(m, 1);
u2 = randn(n, 1);
v2 = rand(m, 1);

X = u1*v1' + u2 * v2';

% reveal only 40% of the entries 
nentries = n*m;
frac = 0.6;
nreveal = n * m * frac;
rp = randperm(nentries);
S = rp(1:nreveal);
[S_i, S_j] = ind2sub([n m], S); 
Xobs = X;
Xobs(~ismember(1:nentries, S)) = NaN;

%%%


% try to complete matrix using matrix factorization and coordinate
% descent

maxiter = 100;
iter = 0;
objective = zeros(maxiter, 1);

% random init
uhat1 = randn(n,1)/10;
vhat1 = randn(m,1)/10;
uhat2 = randn(n,1)/10;
vhat2 = randn(m,1)/10;


Xzeroed = Xobs;
Xzeroed(isnan(Xzeroed)) = 0;

while iter < maxiter
    
    fit2 = uhat2 * vhat2';
    fit2(isnan(Xobs)) = 0;
    Residual = Xzeroed - fit2;

    %%% first low-rank term
    tmp_v = Residual' * uhat1;
    for j=1:m
        vhat1(j) = tmp_v(j) / sum(uhat1(S_i(S_j == j)).^2);
    end
    
    tmp_u = Residual * vhat1;
    for i=1:n
        uhat1(i) = tmp_u(i) / sum(vhat1(S_j(S_i == i)).^2);
    end
    
    %%% second low-rank term
    
    fit1 = uhat1 * vhat1';
    fit1(isnan(Xobs)) = 0;
    Residual = Xzeroed - fit1;
    
    tmp_v = Residual' * uhat2;
    for j=1:m
        vhat2(j) = tmp_v(j) / sum(uhat2(S_i(S_j == j)).^2);
    end
    
    tmp_u = Residual * vhat2;
    for i=1:n
        uhat2(i) = tmp_u(i) / sum(vhat2(S_j(S_i == i)).^2);
    end
    
    
    
    iter = iter + 1;
    Xhat = uhat1 * vhat1' + uhat2 * vhat2';
    objective(iter) = sum((Xhat(S) - X(S)).^2);
    
    if objective(iter) < 1E-10
       break; 
    end
end

objective = objective(1:iter);

norm(Xhat - X, 'fro') % almost exact agreement


