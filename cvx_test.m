addpath(genpath('Users/martinslawski/mydata/cvx/'))

cvx_begin
variable a(5)
minimize sum(-log(a))
subject to 
sum(a) == 5
cvx_end