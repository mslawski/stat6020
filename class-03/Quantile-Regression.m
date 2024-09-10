%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotpath =  '/Users/martinslawski/Dropbox/UVA/teaching/STAT6020/figs/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% classic data set on quantile regression

engel = dlmread('engel.txt')/100;

% 1st column: household income, 2nd column: food expenditures

X = engel(:,1);
Y = engel(:,2);

plot(X, Y, '*')

beta_LS = dot(X - mean(X), Y - mean(Y))/sum((X - mean(X)).^2);
beta0_LS = mean(Y) - mean(X) * beta_LS;

line([1 50], [1*beta_LS + beta0_LS 50*beta_LS + beta0_LS], 'color', 'red')

%%% different quantile regression straight lines

taus = [0.1 0.25 0.5 0.75 0.9];
beta0s_QR = zeros(numel(taus), 1);
betas_QR = zeros(numel(taus), 1);

for i=1:numel(taus)
    tau = taus(i);
   cvx_begin quiet
   variable bet0
   variable bet
   minimize sum((1 - tau) * max(-(Y - X * bet - bet0), 0) + tau * max(0, (Y - X * bet - bet0)))
   cvx_end
   beta0s_QR(i) = bet0;
   betas_QR(i) = bet;
   line([1 50], [1*bet + bet0 50*bet + bet0])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






