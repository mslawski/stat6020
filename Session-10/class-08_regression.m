%%% Optimization for (shallow) neural networks:
%%% Regression

%%% generate some data for a regression problem (noisy sine) and visualize  

x = (0.01:0.001:1)';
d = size(x,2);
n = numel(x);
xi = 0.25 * randn(n, 1);
y = sin(x*2*pi) + xi;

plot(x, y, '*')

%%% Setting up 

L = 10;%%% number of hidden layers
%lambda = 0;
lambda = 0.0001; %%% parameter weight decay

% activation function used at the output layer
sigma2 = @(x) x; % identity --- since we are in the regression setting
Dsigma2 = @(x) ones(numel(x), 1); % derivative is the identity map 

%%% The RELU function is not smooth, hence we used a smooth approximation
%%% so gradient descent can be applied. 
mu = 0.03;
sigma1 = @(x) (mu * log(exp(x / mu) + 1) - mu*log(2)); % smoothing of RELU function

% visual verification of the approximation
%grx = -3:0.01:3;
%plot(grx, sigma1(grx), '-', 'LineWidth', 4)

% derivative of sigma1
Dsigma1 = @(x) exp(x/mu)./(1 + exp(x/mu));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Actual Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxiter = 10000;
fvals = zeros(maxiter, 1);

% initialization 
Xtilde = [ones(n, 1) x]; 
Wtilde = randn(d+1, L) * 0.1;
wtilde = randn(L+1,1) * 0.1;

Xtildet_bdiag = kron(eye(L), Xtilde');

% function for generating the first hidden layer 
z1 = @(Wtilde) Xtilde * Wtilde;

% function for generating the second hidden layer 
z2 = @(Htilde, wtilde) Htilde * wtilde;  

% loss
loss = @(yhat) sum(0.5*(yhat - y).^2);
Dloss = @(yhat) (yhat - y);

% function evaluation (forward propagation)
Z1cur = z1(Wtilde);
Htilde = [ones(n,1) sigma1(Z1cur)];
z2cur = z2(Htilde, wtilde);
yhat = sigma2(z2cur);
iter = 1;
fvals(iter) = mean(loss(yhat)) + ((lambda/2) * sum(wtilde(2:end).^2) + (lambda/2) * sum(sum(Wtilde(2:end,:).^2)));

% gradient descent step-size 
stepsize = 0.1;

while iter < maxiter

    % compute gradient (back propagation)
    dL = Dloss(yhat);
    dsigma2 = Dsigma2(z2cur);
    weights = (dL .* dsigma2);
    grad_wtilde = (Htilde' *  weights/ n) + lambda * [0; wtilde(2:(L+1))];
    
    grad_Wtilde_p1 = (repmat(weights, [1 L]) .* Dsigma1(Z1cur)) .* (repmat(wtilde(2:(L+1)), [1 n])');
    grad_Wtilde_p2 =  Xtildet_bdiag  * grad_Wtilde_p1(:);
    grad_Wtilde = reshape(grad_Wtilde_p2, [(d+1) L])/n + lambda * [zeros(1, L);Wtilde(2:end,:)];
    % gradient update 
    %stepsize = min(0.1, 1/iter);
    wtilde = wtilde - stepsize * grad_wtilde;
    Wtilde = Wtilde - stepsize * grad_Wtilde;
    % compute function value (forward propagation)
    Z1cur = z1(Wtilde);
    Htilde = [ones(n,1) sigma1(Z1cur)];
    z2cur = z2(Htilde, wtilde);
    yhat = sigma2(z2cur);
    iter = iter + 1;
    fvals(iter) = mean(loss(yhat)) + ((lambda/2) * sum(wtilde(2:end).^2) +  (lambda/2) * sum(sum(Wtilde(2:end,:).^2)));

  
end

plot(fvals, '-*')

%%% visualize function learned

figure 
hold on
plot(x, sin(x*2*pi), '-r')
plot(x, yhat, '*', 'MarkerSize', .5, 'Color', 'blue')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



