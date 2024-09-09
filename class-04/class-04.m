%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotpath =  '/Users/martinslawski/Dropbox/UVA/teaching/STAT6020/figs/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Example 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function to be minimized: quadratic function

f = @(x) 3*x.^2 - 6*x +1;
gradf = @(x) 6*x -6;
grid = 0:0.01:2;

plot(grid, f(grid))

% global minimizer is xstar = 1

% gradient descent with constant step-size

maxiter = 200;
iterates = zeros(maxiter, 1);
function_values = zeros(maxiter, 1);

stepsize = 0.33;  %1;%0.34;% 0.3. %0.4; %1;

xcur = 0;

iter = 0;

while iter < maxiter
   
    xcur = xcur - stepsize * gradf(xcur);
    iter = iter + 1;
    iterates(iter) = xcur;
    function_values(iter) = f(xcur);
    
end

figure
hold on 
plot(grid, f(grid))
plot(iterates, function_values, '-*r')

exportgraphics(gcf, [plotpath 'gradient_zigzag.pdf']);

%%%% Example 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho = 0.7; % .99; 1

Q = [1 rho; rho 1];
c = -2*Q*[2; -1] - 2 * Q * 0.1 * randn(2, 1);


foo = @(x) x' * Q * x + x' * c;
grad = @(x) 2*Q*x + c;

xstar = -(Q \ (c/2));
%foo(xstar)


stepsize = 0.95 * (1 / (1+ rho)); 

maxiter = 1000;
iterates = zeros(maxiter, 2);
function_values = zeros(maxiter, 1);


xcur = [0;0];

tol = 1E-8;
iter = 0;

while iter < maxiter && norm(grad(xcur)) > tol 
   
    xcur = xcur - stepsize * grad(xcur);
    iter = iter + 1;
    iterates(iter,:) = xcur;
    function_values(iter) = foo(xcur);
    
end

iterates = iterates(1:iter, :);
function_values = function_values(1:iter, :);

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

%%%% Example 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% logistic regression

beta = [-1; 1];
beta0 = 0;

n = 50;

X = randn(n, numel(beta));

y = sign(X * beta + beta0);

y = y .* (2*(rand(n, 1) > 0.1)-1); % add some label noise 

figure
hold on
plot(X(sign(y) == 1,1), X(sign(y) == 1,2), '+r', 'MarkerSize', 12)
plot(X(sign(y) == -1,1), X(sign(y) == -1,2), 'xk', 'MarkerSize', 12)

line([-3,3], [-3,3])

% gradient descent

f = @(theta) theta(1) * ones(n, 1) + X * theta(2:end);

foo = @(f) sum(log(1 + exp(-y .* f)));


grad = @(theta) sum([ones(n, 1) X]' * diag(-y ./ (1 + exp(y .* f(theta)))), 2);

stepsize = 0.9 * 2  / (norm([ones(n, 1) X]).^2); 


maxiter = 200;
iterates = zeros(maxiter, 3);
function_values = zeros(maxiter, 1);


xcur = [0;0;0];

tol = 1E-8;
iter = 0;

while iter < maxiter && norm(grad(xcur)) > tol 
   
    xcur = xcur - stepsize * grad(xcur);
    iter = iter + 1;
    iterates(iter,:) = xcur;
    function_values(iter) = foo(f(xcur));
    
end

iterates = iterates(1:iter, :);
function_values = function_values(1:iter, :);

figure
hold on
plot(X(sign(y) == 1,1), X(sign(y) == 1,2), '+r', 'MarkerSize', 16)
plot(X(sign(y) == -1,1), X(sign(y) == -1,2), 'xk', 'MarkerSize', 16)
for i=1:2:10
    interc = -iterates(i,1)/iterates(i,end);
    slope = -iterates(i,2)/iterates(i,end);
    line([-3,3], [-3 * slope + interc, 3 * slope + interc])
    %pause() --- out-comment if you want to see how the hyperplane is
    % adjusted
end


cvx_begin
variable beta0_cvx
variable beta_cvx(2)
minimize sum(log(1 + exp(-y .* (X * beta_cvx + ones(n,1) * beta0_cvx))))
cvx_end


%%% minimizing quadratic function with back-tracking line search
%&& norm(grad(xcur)) > tol 

rho = 0.8;

Q = [1 rho; rho 1];
c = -2*Q*[2; -1] - 2 * Q * 0.1 * randn(2, 1);


foo = @(x) x' * Q * x + x' * c;
grad = @(x) 2*Q*x + c;

% optimal colution, for checking
% xstar = Q \ (-c/2);
% foo(xstar)

maxiter = 20000;
iterates = zeros(maxiter, 2);
function_values = zeros(maxiter, 1);

xcur = [0;0];

tau = 0.5;
gamma = 0.8;
tol = 1E-8;
iter = 0;
foo_xcur = foo(xcur);

while iter < maxiter 
   
    gradxcur = grad(xcur);
    normsq_gradxcur = norm(gradxcur)^2; 
    
    % check Armijo rule
    m = 0;
    while foo_xcur - foo(xcur - gamma^m * gradxcur) < tau * gamma^m * normsq_gradxcur  
        m = m+1;
        if gamma^m < tol^2
            flag = true;
           break;
        end
    end
    if  flag
        break; 
    end
    
       
    xcur = xcur - gamma^m * gradxcur;
    iter = iter + 1;
    iterates(iter,:) = xcur;
    foo_xcur = foo(xcur);
    function_values(iter) = foo_xcur;
    
end

iterates = iterates(1:iter, :);
function_values = function_values(1:iter, :);