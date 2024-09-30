%%% Optimization for (shallow) neural networks:
%%% Binary Classification

%%% Classification example -- the XOR problem
n = 1000;
x = 2*randn(n, 2) - 1;
y = (sign(x(:,1).*x(:,2)) + 1)/2;
d = size(x,2);
 
figure 
hold on
plot(x(y == 0,1), x(y == 0,2), '*r')
plot(x(y == 1,1), x(y == 1,2), '*b')

%%% Setting up 

L = 10; %%% number of hidden layers
%lambda = 0;
lambda = 0.01; %%% parameter weight decay

% loss function: cross entropy aka negative log-likelihood of a logistic
% regression model. 
loss = @(yhat) -(y .* log(yhat) + (1- y) .* log(1 - yhat));
% D --- short for derivative: derivative of the loss function
Dloss = @(yhat) -(y./yhat - (1-y)./(1 - yhat));


% activation function and derivative at the output layer (layer 2)
sigma2 = @(x) exp(x)./(1 + exp(x)); % logistic --- classification
Dsigma2 = @(x) sigma2(x) .* (1 - sigma2(x));

%%% Numerical Derivative check --- useful to know: 
%DSigma2(1.4)
%delta = 0.001;
%(sigma2(1.4 + delta) - sigma2(1.4))/delta

%%% activation function for generating the hidden layer

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

% 
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


% function evaluation (forward propagation)
Z1cur = z1(Wtilde);
Htilde = [ones(n,1) sigma1(Z1cur)];
z2cur = z2(Htilde, wtilde);
yhat = sigma2(z2cur);
iter = 1;
fvals(iter) = mean(loss(yhat)) + ((lambda/2) * sum(wtilde(2:end).^2) + (lambda/2) * sum(sum(Wtilde(2:end,:).^2)));

% gradient descent step-size 
stepsize = 0.03;

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

% sequence of function values
plot(fvals, '-*')
% misclassification rate on the training set
mean((yhat > 0.5) ~= y)

%%% visualize the regions that were learned
x1gr = (-1:0.01:1)';

counter = 1;
Xgr = zeros(numel(x1gr).^2, 2);
for i=1:numel(x1gr)
    for j=1:numel(x1gr)
      Xgr(counter,:) = [x1gr(i) x1gr(j)];  
      counter = counter + 1;
    end
end

ngr = size(Xgr, 1); 


Z1gr =  [ones(ngr, 1) Xgr] * Wtilde;
Htildegr = [ones(ngr,1) sigma1(Z1gr)];
z2gr = z2(Htildegr, wtilde);
yhatgr = sigma2(z2gr);

figure
hold on
plot(Xgr(yhatgr < 0.5,1), Xgr(yhatgr < 0.5,2), '*r')
plot(Xgr(yhatgr > 0.5,1), Xgr(yhatgr > 0.5,2), '*b')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




