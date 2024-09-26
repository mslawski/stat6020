%%% Data read-in and processing

cod_rna = dlmread('cod-rna.txt');
%sum(any(isnan(cod_rna)))

y = cod_rna(:,1);
X0 = cod_rna(:,2:end);


[n,d] = size(X0); 
% centering + scaling of the features
X = X0;

for j=1:d
    X(:,j) = (X(:,j) - mean(X(:,j)))/std(X(:,j));
end

% check: 
% mean(X), std(X)

% add intercept
X = [ones(n,1) X];
%svd(X)


%%% [1] batch gradient descent with constant step-size


f = @(theta) X * theta;

foo = @(f) mean(log(1 + exp(-y .* f)));


grad = @(theta) X' * (-y ./ (1 + exp(y .* f(theta))))/n;

stepsize = 0.9 * 2  / (normest(X).^2) * n; 


maxiter = 200;
function_values = zeros(maxiter, 1);
training_accuracy = zeros(maxiter, 1);

xcur = zeros(d + 1, 1);
tol = 1E-8;
iter = 0;

while iter < maxiter && norm(grad(xcur)) > tol 
   
    xcur = xcur - stepsize * grad(xcur);
    iter = iter + 1;
    function_values(iter) = foo(f(xcur));
    training_accuracy(iter) = mean(sign(X * xcur) ~= y);
end

function_values = function_values(1:iter, :);
training_accuracy = training_accuracy(1:iter, :);

% function values
plot(1:numel(function_values), log(function_values), '-*')
% training accuracy (i.e., misclassification rate) on training set
plot(1:numel(function_values), training_accuracy, '-*r')

%%% [2] stochastic gradient descent with constant step-size

f_i = @(theta, i) sum(X(i,:) .* theta'); 
grad_stoch = @(theta, i) transpose(X(i,:)) * (-y(i) / (1 + exp(y(i) .* f_i(theta,i))));

%%% estimating the constant M_V for the step size selection 

grad_stoch_normssq = zeros(n,1);
for i=1:n
   grad_stoch_normssq(i) =  norm(grad_stoch(zeros(d+1, 1), i))^2; 
end

M_V = (mean(grad_stoch_normssq) - norm(grad(zeros(d+1, 1)))^2) / (norm(grad(zeros(d+1, 1)))^2);

maxiter = 10000;
function_values_s = zeros(maxiter, 1);
training_accuracy_s = zeros(maxiter, 1);

xcur = zeros(d + 1, 1);
stepsize_stoch = stepsize / (M_V + 1);
tol = 1E-8;
iter = 0;

while iter < maxiter %&& norm(grad(xcur)) > tol 
   
    ix = randperm(n);
    xcur = xcur - stepsize_stoch * grad_stoch(xcur, ix(1));
    iter = iter + 1;
    function_values_s(iter) = foo(f(xcur));
    training_accuracy_s(iter) = mean(sign(X * xcur) ~= y);
end

function_values_s = function_values_s(1:iter, :);
training_accuracy_s = training_accuracy_s(1:iter, :);

plot(1:numel(function_values_s), log(function_values_s), '-*')
plot(1:numel(training_accuracy_s), training_accuracy_s, '-r') 

% Comparison

% 1 --- assuming that 1 iteration of SGD is (only) 100x cheaper than
% a full gradient iteration
figure
hold on 
plot((1:numel(function_values))*100, log(function_values), '-*')
plot(1:numel(function_values_s), log(function_values_s), '-*')

% 2 --- asumming that 1 iteration of SGD is 1000 x cheaper than a full
% gradient iteration

figure
hold on 
plot((1:numel(function_values))*100, log(function_values), '-*')
plot(1:numel(function_values_s), log(function_values_s), '-*')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [3] stochastic gradient descent with diminishing step size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_i = @(theta, i) dot(X(i,:), theta'); 
grad_stoch = @(theta, i) transpose(X(i,:)) * (-y(i) / (1 + exp(y(i) .* f_i(theta,i))));


maxiter = 10000;
function_values_s_d = zeros(maxiter, 1);
training_accuracy_s_d = zeros(maxiter, 1);

xcur = zeros(d + 1, 1);

beta = 0.1; % this is a guess for the strong convexity parameter
% determine gamma to satisfy the 2nd condition on slide 13:
gamma =  max((1/beta) / stepsize_stoch  - 1, 0); 
tol = 1E-8;
iter = 0;

while iter < maxiter %&& norm(grad(xcur)) > tol 
   
    ix = randperm(n);
    xcur = xcur - grad_stoch(xcur, ix(1)) * ((1/beta) / (gamma + iter + 1));
    iter = iter + 1;
    function_values_s_d(iter) = foo(f(xcur));
    training_accuracy_s_d(iter) = mean(sign(X * xcur) ~= y);
end


plot(1:numel(function_values_s_d), log(function_values_s_d), '-*')
plot(1:numel(function_values), training_accuracy, '-*r')

% comparison to fixed step size

figure
hold on
plot(1:numel(function_values_s), log(function_values_s), '-*')
plot(1:numel(function_values_s_d), log(function_values_s_d), '-*')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [4] stochastic gradient descent with mini-batches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_ids = @(theta, ids) X(ids,:) * theta; 
grad_stoch_mb = @(theta, ids) transpose(X(ids,:)) * (-y(ids) ./ (1 + exp(y(ids) .* f_ids(theta,ids)))) / numel(ids);


maxiter = 1000;
batchsize = 50; % batchsize 50 (still less than 0.1% of full data set)
function_values_s_mb = zeros(maxiter, 1);
training_accuracy_s_mb = zeros(maxiter, 1);

xcur = zeros(d + 1, 1);
stepsize_stoch_mb = 5 * stepsize / (M_V + 1); 
% we take a 5x larger step-size, assuming less noise in the mini-batch
% gradient
tol = 1E-8;
iter = 0;

while iter < maxiter %&& norm(grad(xcur)) > tol 
   
    ix = randperm(n);
    xcur = xcur - stepsize_stoch_mb * grad_stoch_mb(xcur, ix(1:batchsize));
    iter = iter + 1;
    function_values_s_mb(iter) = foo(f(xcur));
    training_accuracy_s_mb(iter) = mean(sign(X * xcur) ~= y);
end



plot(1:numel(function_values_s_mb), log(function_values_s_mb), '-*')
plot(1:numel(training_accuracy_s_mb), training_accuracy_s_mb, '-*')

% comparison to batch size = 1

figure
hold on
plot(1:numel(function_values_s), log(function_values_s), '-*')
plot(1:numel(function_values_s_mb), log(function_values_s_mb), '-*')

% w/rescaling (by #samples used for computing the batch gradient)

figure
hold on
plot(1:numel(function_values_s), log(function_values_s), '-*')
plot((1:numel(function_values_s_mb))*50, log(function_values_s_mb), '-*')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [5] SVRG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we load in a smaller data set for faster demonstration

X = dlmread('../hw/X_HW.txt');
y = dlmread('../hw/y_HW.txt');

n = numel(y);

X = [ones(n, 1) X];
d = size(X, 2);

% huber loss
M = 1;
loss = @(z) z.^2 .* (abs(z) <= M) + M*(2*abs(z) - M) .* (abs(z) > M);
grad_loss = @(z) 2 * z .* (abs(z) <= M) + 2*M*sign(z) .* (abs(z) > M);
grad = @(theta, ids) -X(ids,:)'*grad_loss(y(ids) - X(ids,:)*theta)/numel(ids);
obj = @(theta) mean(loss(y - X * theta));

%%% 1 --- constant step size

L = 2 * norm(X).^2 /n;
stepsize = 1.5/L;

% estimate for constant M_V in stochastic gradient
grad_stoch_normssq = zeros(n,1);
for i=1:n
   grad_stoch_normssq(i) =  norm(grad(zeros(d, 1), i))^2; 
end

% note that in the first example, the constant "3" does not appear.
% I blew up the constant a little bit to get better results (less fluctuation). Note that
% estimating M_V at a single point may not give the best results.
M_V = 3*(mean(grad_stoch_normssq) - norm(grad(zeros(d,1), 1:n))^2) / (norm(grad(zeros(d,1), 1:n))^2);

%

epochs = 100;
batchsize = 1;
function_values = zeros(epochs * n, 1);

%%%

xcur = zeros(d, 1);
stepsize_stoch = stepsize / (M_V + 1);
function_values = zeros(epochs * n, 1);
iter = 0;

for ii=1:epochs 
   
    ix = randperm(n);
    
    for i=1:n
    
        xcur = xcur -  stepsize_stoch* grad(xcur, ix(i));
        iter = iter + 1;
        
        function_values(iter) = obj(xcur);

       
    end
    
        
end

plot(1:iter, function_values, '-*')

%%% 2 --- diminishing step size

xcur = zeros(d, 1);
stepsize_stoch = stepsize / (M_V + 1);
beta =  (min(svd(X))^2/n); % estimate for strong convexity constant
% determine gamma to satisfy the 2nd condition on slide 13: 
gamma = max((1/beta) / stepsize_stoch  - 1, 0);
function_values_d = zeros(epochs * n, 1);
iter = 0;

for ii=1:epochs 
   
    ix = randperm(n);
    
    for i=1:n
    
        xcur = xcur -  ((1/beta) / (gamma + iter))* grad(xcur, ix(i));
        iter = iter + 1;
        
        function_values_d(iter) = obj(xcur);

       
    end
    
        
end

plot(1:iter, function_values_d, '-*')
%function_values(end)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SVRG 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% step size computation --- based on slide 17, setting the right hand side
% equal to .9, m = n, and solving the resulting quadratic equation. 

alpha = -0.01315789474*(-9*n*beta + sqrt(81*n^2*beta^2 - 1520.*L*beta*n))/(L*beta*n);

% check if condition is satisfied:
1/(1 - 2 * alpha * L) * (1 / (n * beta * alpha) + 2 * L * alpha) %~0.9 < 1


xcur = zeros(d, 1);
tol = 1E-8;
iter = 0;

function_values_svrg = zeros(epochs * n, 1);

for ii=1:epochs 
   
    ix = randperm(n);
    
    xinit = xcur;
    grad_batch = grad(xcur, 1:n);
    
    for i=1:n
    
        corr  = grad(xinit, ix(i)) - grad_batch;
        xcur = xcur -  alpha * (grad(xcur, ix(i)) - corr);
        iter = iter + 1;
        
        function_values_svrg(iter) = obj(xcur);
       
    end
    
end

figure
hold on
plot(log(function_values))
plot(log(function_values_d))
plot(log(function_values_svrg))

%%% SAGA

xcur = zeros(d, 1);
tol = 1E-8;
iter = 0;
alpha = 1/(2*(beta * n + L));


all_grads_loss = grad_loss(y - X*xcur);
all_grads = -(X .* repmat(all_grads_loss, [1 d]))';
mean_all_grads = mean(all_grads, 2);

function_values_saga = zeros(epochs * n, 1);


for ii=1:epochs 
   
    ix = randperm(n);
    
    for i=1:n
    
        g_cur = grad(xcur, ix(i));
        
        corr  = (all_grads(:,ix(i)) - mean_all_grads);
        xcur = xcur -  alpha * (g_cur - corr);
        iter = iter + 1;
        function_values_saga(iter) = obj(xcur);
       
        % update gradient memory and running average
        mean_all_grads = mean_all_grads + (g_cur - all_grads(:,ix(i)))/n;
        all_grads(:,ix(i)) = g_cur;
        
        
    end
    
end

figure
hold on
plot(log(function_values))
plot(log(function_values_d))
plot(log(function_values_svrg))
plot(log(function_values_saga))

%%% iterate averaging

xcur = zeros(d, 1);
avg = xcur;
beta = min(svd(X))^2/n;
gamma = max((1/beta) / stepsize_stoch  - 1, 0);
function_values_avg = zeros(epochs * n, 1);
iter = 0;

for ii=1:epochs 
   
    ix = randperm(n);
    
    for i=1:n
    
        xcur = xcur -  ((1/beta) / (gamma + iter^(2/3)))* grad(xcur, ix(i));
        iter = iter + 1;
        avg = avg * (1 - 1/iter) + xcur/iter;
        
        function_values_avg(iter) = obj(avg);

       
    end
    
        
end


figure
hold on
plot(function_values)
plot(function_values_d)
plot(function_values_svrg)
plot(function_values_saga)
plot(function_values_avg)









