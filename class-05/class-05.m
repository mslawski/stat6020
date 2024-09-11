%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotpath =  '/Users/martinslawski/Dropbox/UVA/teaching/STAT6020/figs/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Part 1: Nesterov's accelerated gradient method %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% logistic regression

d = 10;

beta = sign(randn(d, 1));
beta0 = 0;

n = 50;

X = randn(n, numel(beta));

y = sign(X * beta + beta0);

y = y .* (2*(rand(n, 1) > 0.1)-1); % add some label noise 


% gradient descent

f = @(theta) theta(1) * ones(n, 1) + X * theta(2:end);

foo = @(f) sum(log(1 + exp(-y .* f)));


grad = @(theta) sum([ones(n, 1) X]' * diag(-y ./ (1 + exp(y .* f(theta)))), 2);

stepsize = 0.9 * 2  / (norm([ones(n, 1) X]).^2); 


maxiter = 2000;
iterates = zeros(maxiter, d + 1);
iterates_acc = zeros(maxiter, d + 1); % iterates from accelerated gradient method
function_values = zeros(maxiter, 1);
function_values_acc = zeros(maxiter, 1); % function values from accelerated gradient method


xcur = zeros(d + 1, 1);
xcur_acc_prev = zeros(d + 1, 1);
xcur_acc = zeros(d + 1, 1);

tol = 1E-8;
iter = 0;

% ordinary gradient descent
while iter < maxiter && norm(grad(xcur)) > tol 
   
    xcur = xcur - stepsize * grad(xcur);
    iter = iter + 1;
    iterates(iter,:) = xcur;
    function_values(iter) = foo(f(xcur));
    
end

iterates = iterates(1:iter, :);
function_values = function_values(1:iter, :);

% accelerated gradient descent

xprev_acc= zeros(d + 1, 1);
xcur_acc = zeros(d + 1, 1);

tol = 1E-8;
iter = 0;

while iter < maxiter && norm(grad(xcur_acc)) > tol 
   
    z = xcur_acc + ((iter - 1) / (iter + 2)) * (xcur_acc - xprev_acc);
    xprev_acc = xcur_acc;
    xcur_acc = z - stepsize * grad(z);
    
    iter = iter + 1;
    iterates_acc(iter,:) = xcur_acc;
    function_values_acc(iter) = foo(f(xcur_acc));
    
end


figure
hold on
plot(log(function_values(1:40)), '-*b')
plot(log(function_values_acc(1:40)), '-*k')
legend({'plain gradient descent', 'Nesterov acceleration'}, 'FontSize', 28)
xlabel('Iterations')
ylabel('Objective')

exportgraphics(gcf, [plotpath 'gradient_acceleration.pdf']);


% double check results with cvx
% 
% cvx_begin
% variable beta0_cvx
% variable beta_cvx(d)
% minimize sum(log(1 + exp(-y .* (X * beta_cvx + ones(n,1) * beta0_cvx))))
% cvx_end
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Part 2: Newton descent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hess = @(theta) [ones(n, 1)  X]'  * (repmat(exp(-y .* f(theta)) ./ ((1 + exp(-y .* f(theta))).^2), [1 (d+1)]) .* [ones(n, 1)  X]);

maxiter = 200;
iterates_newt = zeros(maxiter, d + 1); % iterates from accelerated gradient method
function_values_newt = zeros(maxiter, 1); % function values from accelerated gradient method

xcur_newt = zeros(d + 1, 1);

tau = 0.5;
gamma = 0.8;
tol = 1E-8;
iter = 0;
flag = false;
foo_xcur = foo(f(xcur_newt));

while iter < maxiter && norm(grad(xcur_newt)) > tol 
   
    gr = grad(xcur_newt);
    H =  hess(xcur_newt);
    dir = H \ (-gr);
    
    % select step-size via Armijo rule
    
    m = 0;
    flag = false;
    while foo_xcur - foo(f(xcur_newt + gamma^m * dir)) < -tau * gamma^m * dot(dir, gr)  
        m = m+1;
        if gamma^m < tol^2
            flag = true;
           break;
        end
    end
    if  flag
        break; 
    end
    
    
    xcur_newt = xcur_newt + gamma^m * dir;
    iter = iter + 1;
    iterates(iter,:) = xcur_newt;
    foo_xcur = foo(f(xcur_newt));
    function_values_newt(iter) = foo_xcur;
    
end

iterates = iterates(1:iter, :);
function_values_newt = function_values_newt(1:iter, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Part 3: Sub-gradient descent                          %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read-data set: regression problem with outliers 

X = dlmread('class-05/XX.txt', ',');
y = dlmread('class-05/y.txt');

[n, d] = size(X);
Z = [ones(n,1) X];

theta_LS = Z \ y;
fitted = Z*theta_LS;
residuals = y - fitted;

close all
plot(fitted, residuals, '*', 'MarkerSize', 10)

subgrad_loss = @(z) sign(z);
subgrad = @(theta) -Z'*subgrad_loss(y - Z*theta);
obj = @(theta) sum(abs(y - Z * theta));

thetacur = theta_LS; % initialize with least squares solution

maxiter = 10000;
objvals =  zeros(maxiter, 1);
iterates = zeros(maxiter, d+1);
iter = 0;
tol = 1E-8;

while iter < maxiter 
   
    thetacur = thetacur - 1/(iter+1) * subgrad(thetacur); % diminishing stepsize
    iter = iter + 1;
    iterates(iter,:) = thetacur;
    objvals(iter) = obj(thetacur);
    
end

iterates = iterates(1:iter, :);
objvals = objvals(1:iter);

% 

cvx_begin
variable thetacvx(d+1)
minimize sum(abs(y - Z * thetacvx))
cvx_end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Part 4: Frank-Wolfe
%%%% %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example: quadratic objective to be minimized over the l1-norm ball of
% radius B = 2

rho = 0.6; 

Q = [1 rho; rho 1];
c = -2*Q*[2; -1] - 2 * Q * 0.1 * randn(2, 1);

foo = @(x) x' * Q * x + x' * c;
grad = @(x) 2*Q*x + c;

maxiter = 5000;
iterates = zeros(maxiter, 2);
function_values = zeros(maxiter, 1);

iter = 0;
tol = 1E-6;
flag = false;
B = 2;
xcur = zeros(2, 1);
foo_xcur = foo(xcur);

gamma = 0.8;
tau = 0.5;

while iter < maxiter 
   
    g = grad(xcur);
    [~, max_ix] = max(abs(g));
    sign_max_ix = -sign(g(max_ix));
    
    dir = -xcur;
    dir(max_ix) = B * sign_max_ix - xcur(max_ix);
    
    dot_dir_g = dot(dir, g);
    
    if dot_dir_g > -tol
       break; 
    end
    
    
    
    % perform inexact line search using the Armijo rule
    m = 0;
    while foo_xcur - foo(xcur + gamma^m * dir) < -tau * gamma^m * dot_dir_g  
        m = m+1;
        if gamma^m < tol^2
            flag = true;
           break;
        end
    end
    if  flag
        break; 
    end
    
    xcur = xcur + gamma^m * dir;
    %norm(xcur, 1)
    iter = iter + 1;
    iterates(iter,:) = xcur;
    foo_xcur = foo(xcur);
    function_values(iter) = foo_xcur;
    
end

cvx_begin
variable xcvx(2)
minimize transpose(xcvx) * Q * xcvx + dot(xcvx, c)
subject to
norm(xcvx, 1) <= B
cvx_end

cvx_optval


xgrid = -3:0.1:3;
ygrid = xgrid;
z = zeros(numel(xgrid), numel(ygrid));

for i=1:numel(xgrid)
    for j=1:numel(ygrid)
    z(i,j) = foo([xgrid(i); ygrid(j)]);
    end
end


figure
hold on
contour(xgrid, ygrid, z', 50)
colorbar
plot([0;iterates(:,1)], [0;iterates(:,2)], '-*r')

line([2 0], [0 2], 'color', 'black', 'LineWidth', 2)
line([2 0], [0 -2], 'color', 'black', 'LineWidth', 2)
line([0 -2], [-2 0], 'color', 'black', 'LineWidth', 2)
line([-2 0], [0 2], 'color', 'black', 'LineWidth', 2)