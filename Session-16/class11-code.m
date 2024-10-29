%
% Semidefinite programming
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 1: MAXCUT for a small graph
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 6;

%%% adjacency matrix of the graph

A = zeros(n);

A(1,2) = 1;
A(2,3) = 1;
A(3,4) = 1;
A(3,6) = 1;
A(4,5) = 1;
A(5,6) = 1;

A = A + A';

%%% approach 1: global optimum by enumeration (2^6 = 64 possibilities)

Xs = 2*(dec2bin(0:2^n-1, 6)-'0')-1;

maxcuts = zeros(2^n, 1);

for i=1:2^n
    
    xi = Xs(i,:);
    
    maxcuts(i) = -(xi * A * xi')/2;
      
end

maxcuts = maxcuts + sum(sum(A));

max(maxcuts)

find(maxcuts == max(maxcuts)) % note that the two solutions are equivalent

%%% approach 2: SDP relaxation

C = -A/2;

cvx_begin sdp
variable Y(n,n) symmetric
maximize trace(C * Y)
subject to 
Y >= 0
diag(Y) == ones(n, 1)
cvx_end

sum(sum(A)) + cvx_optval % same as maxcut optimal function value

% the SDP solution here already has rank 1 and integral, 
% and hence delivers the optimal solution already. 
% No randomized routing necessary. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 2: Sparse PCA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% underlying covariance matrix: vv' + W, where v is sparse and W is 
% Wishart "noise"

d = 50;
s = 5;
v = [2; -3;  1; -1; 4; zeros(d-s, 1)];

n = 100;

Sigma = v*v' + 10 * wishrnd(eye(d), n)/n;

% approach 1: computing top eigenvector and thresholding 

[V, lambda] = eig(Sigma);

% approach 2: SDP relaxation as in D'Aspremont et al.

cvx_begin sdp
variable Y(d,d) symmetric
maximize trace(Y * Sigma)
subject to 
Y >= 0
trace(Y) == 1
sum(abs(Y(:))) <= s
cvx_end 

[V_Y, lambda_Y] = eig(Y);

figure
hold on
plot(1:d, v / norm(v))
plot(1:d, V(:,end))
plot(1:d, V_Y(:,end))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example 3: Mixture of Regressions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data: mixtures of simple linear regressions
n = 500;

X1 = [ones(n/2, 1) rand(n/2, 1)];
beta1 = [-1; 2];
y1 = X1 * beta1;

X2 = [ones(n/2, 1) rand(n/2, 1)];
beta2 = [1; -1];
y2 = X2 * beta2;

y = [y1; y2];
X = [X1; X2];

y = y + 0.2 * randn(n, 1);

plot(X(:,2), y, '*')

%%% approach by Chen et al. (2018)
d = 2;
bigX = zeros(n, d^2);
for i=1:n
   bigX(i,:) = reshape(X(i,:)' * X(i,:), 1, d^2); 
end

lambda = n * 0.14;

cvx_begin
variable K(2,2) symmetric
variable g(2)
minimize norm_nuc(K)% )- 
subject to 
sum(abs(-bigX * K(:) + 2*(y.* (X * g)) - y.^2)) <= lambda
cvx_end

J = g*g' - K;

[V, lam] = eig(J);

beta1hat = g + sqrt(lam(end,end)) * V(:,end);
beta2hat = g - sqrt(lam(end,end)) * V(:,end);
