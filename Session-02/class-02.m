%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setup CVX
path = pwd;
cd '/Users/martinslawski/mydata/cvx/'
cvx_setup
cd(path)

% directory for storing plots
plotpath =  '/Users/martinslawski/Dropbox/UVA/teaching/STAT6020/figs/';

fs_lab = 12*1.5;
fs_glob = 18*1.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Example 1: Loss functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 50;
x = sort(rand(n, 1));

y = sqrt(x) + 0.1 * randn(n, 1);

% (a) Squared Loss 

beta = sum((x - mean(x)) .* (y - mean(y))) ./ sum(((x - mean(x))).^2);
beta0 = mean(y) - mean(x) * beta;

% (b) Absolute Loss (CVX)

cvx_begin
variable beta0_abs
variable beta_abs
minimize sum(abs(y - beta0_abs - beta_abs * x))
cvx_end

% (c) Kullback Leibler Divergence (coordinate descent)


fval = inf;
tol = 1E-10;
iter = 0;
% initialize beta
beta_KL = beta;
sumx = sum(x);
fvals = [];
    
while true
    
   
    % find iterate for beta0_KL given beta_KL
    foo0 = @(beta0) sum(-y .* log(beta_KL * x + beta0)) + n * beta0 + sumx * beta_KL;
    
    [beta0_KL, fval_new] = fminbnd(foo0, 0.01, 1); % assume we optimize between 0.01 and 1
    
    if fval_new < fval - tol
        fval = fval_new;
        fvals = [fvals; fval];
    else
        break;
    end
    
    % find iterate for beta_KL given beta0_KL
    foo = @(beta) sum(-y .* log(beta * x + beta0_KL) + x * beta) + n * beta0_KL;
    
    [beta_KL, fval_new] = fminbnd(foo, 0.01, 1); % assume we optimize between 0.01 and 1
    
    if fval_new < fval - tol
        fval = fval_new;
        fvals = [fvals; fval];
    else
        break;
    end

    iter = iter + 1;
    
end

%%% plot function values 

plot(1:numel(fvals), log(fvals), '-*')
diff(fvals)

box on
set(gca,'FontSize', fs_glob);
xlabel('Iterations',  'FontSize',  fs_lab * 1.5);
ylabel('log(Function Values)', 'FontSize',  1.25 * fs_lab)

saveas(gcf, [plotpath 'functionvalues_KL.fig']);
exportgraphics(gcf, [plotpath 'functionvalues_KL.pdf']);


% double-check result from (c) using CVX

cvx_begin
variable beta0_KL_cvx
variable beta_KL_cvx
minimize sum(-y .* log(beta0_KL_cvx + x * beta_KL_cvx) + (x * beta_KL_cvx + beta0_KL_cvx))
cvx_end

% results match perfectly

%%% Plots results 

plot(x, y, '*')

box on
%set(gcf, 'units','normalized','outerposition',[0 0 1 1])
set(gca,'FontSize', fs_glob);
xlabel('X',  'FontSize',  fs_lab * 1.5);
ylabel('Y', 'FontSize',  1.25 * fs_lab)
set(gca, 'XTick', [0.1:0.1:1])
ylim([0 1.1])

% draw straight lines

% squared
line([0, 1], [beta0, beta0 + beta])
% absolute
line([0, 1], [beta0_abs, beta0_abs + beta_abs], 'color', 'black')
% KL
line([0, 1], [beta0_abs, beta0_KL + beta_KL], 'color', 'red')
%

legend('Data', 'Squared', 'Absolute', 'KL', 'Location', 'southeast')
saveas(gcf, [plotpath 'isotonic_lines.fig']);
exportgraphics(gcf, [plotpath 'isotonic_lines.pdf']);


%%% non-convex optimization problem: capped ell_1 loss

c = 0.05;
loss = @(t) sum(abs(t) .* (t <= c) + c .* (t > c));

%%% evaluate optimization landscape

beta0grid = 0:0.01:1;
betagrid = beta0grid;

feval = zeros(numel(beta0grid), numel(betagrid));

for i=1:numel(beta0grid)
    for j=1:numel(betagrid)
       feval(i, j) = loss(abs(y - beta0grid(i) - betagrid(j) * x)); 
    end
end

contour(beta0grid, betagrid, feval, 20)
colorbar
exportgraphics(gcf, [plotpath 'contours.pdf']);

% contour plot

plot(betagrid, feval(39, :))
exportgraphics(gcf, [plotpath 'function_beta0.pdf']);


%%% try coordinate descent for this problem

fval = inf;
tol = 1E-10;
iter = 0;
% initialize beta
beta_clipped = beta; %0.7; %0.5; --- other potential statring values
sumx = sum(x);
fvals = [];
    
while true
    
   
    % 
    foo0 = @(beta0) loss(abs(y - beta0 - beta_clipped * x));
    
    [beta0_clipped, fval_new] = fminbnd(foo0, 0.01, 0.4); % assume we optimize between 0.01 and 0.4
    
    if fval_new < fval - tol
        fval = fval_new;
        fvals = [fvals; fval];
    else
        break;
    end
    
    foo = @(beta) loss(abs(y - beta0_clipped - beta * x));
    
    [beta_clipped, fval_new] = fminbnd(foo, 0.5, 1); % assume we optimize between 0.5 and 1
    
    if fval_new < fval - tol
        fval = fval_new;
        fvals = [fvals; fval];
    else
        break;
    end

    iter = iter + 1;
    
end

% does not really work 


