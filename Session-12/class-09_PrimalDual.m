%%% Primal Dual Method for NNLS (Non-negative Least Squares)

n = 100;
d = 20;
x0 = full(sprand(d, 1, 0.3));

A = rand(n, d);
b = A*x0 + 0.1 * randn(n, 1);

out = primaldual(A, b, 1E-8);

  
cvx_begin
variable xcvx(d)
minimize sum((A * xcvx - b).^2)
subject to
xcvx >= 0
cvx_end

norm(xcvx - out.xopt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
