###################################################################################################
###
### Random effects model for outlier-sensitive regression: EM algorithm + numerical integration
### 
###################################################################################################

# data set from HW #1
ls()
X <- as.matrix(read.table("../hw/X_HW.txt", sep = ","))
y <- as.numeric(read.table("../hw/y_HW.txt")$V1)

# this is the ``ground truth parameter" -- Huber regression from HW #1 achieves an RMSE of .1941, down from .4296 (least squares) 
betastar <- c(-1, 0.1644, 0.5607, -0.6907,
              0.2636, 0.0975, -0.3998,
              -0.1326, 0.1048, 1.0941, 0.8468)

    
X <- cbind(1, X)
d <- ncol(X)
n <- nrow(X)

###

# assumed PDF of random effects
f_gamma <- function(gamma, tau) exp(-abs(gamma)/tau)/(2*tau)
# assumed PDF given random effects 
f_ymu <- function(res, sigmasq) dnorm(res, sd = sqrt(sigmasq))
# resulting negative log-likelihood
nloglik <- function(beta, sigmasq, tau){

    mu <- X %*% beta
    res <- y - mu

    all_ints <- sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasq) * f_gamma(gamma, tau), lower = -20, upper = 20)$value)

    sum(-log(all_ints))
    
}

# gradient 
grad_nloglik <- function(beta, sigmasq, tau){

    mu <- X %*% beta
    res <- y - mu

    all_ints <- sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasq) * f_gamma(gamma, tau), lower = -20, upper = 20)$value)

    grad_beta <- t(X) %*% (sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasq) * (res - gamma)/sigmasq * f_gamma(gamma, tau), lower = -20, upper = 20)$value)/(-all_ints))

   grad_sigmasq <-  sum(sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasq) * ((res - gamma)^2/(2 * sigmasq^2) - 0.5/sigmasq)  * f_gamma(gamma, tau), lower = -20, upper = 20)$value)/(-all_ints))

    grad_tau <- sum(sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasq) * f_gamma(gamma, tau) * (abs(gamma)/tau^2 - 1/tau), lower = -20, upper = 20)$value)/(-all_ints))

    return(c(grad_beta, grad_sigmasq, grad_tau))
    
}     

### initialization 

# least squares solutions for beta, sigma
lm0 <- lm(y ~ X - 1)
betacur <- coef(lm0) # intercept already included in X
sigmasqcur <- mean(residuals(lm0)^2)  
taucur <- sqrt(sigmasqcur)


nloglik(betacur, sigmasqcur, taucur) # 293.8056 --- negative log-likelihood at starting point 

### test derivatives:
delta <- 1E-4

# 1 --- beta
(nloglik(c(betacur[1], betacur[2] + delta, betacur[3:d]), sigmasqcur, taucur) - nloglik(c(betacur[1], betacur[2] - delta, betacur[3:d]), sigmasqcur, taucur))/(2 * delta) # correct 

# 2 --- sigmasq

(nloglik(betacur, sigmasqcur + delta, taucur) - nloglik(betacur, sigmasqcur - delta, taucur))/(2 * delta)

# 3 --- tau

(nloglik(betacur, sigmasqcur, taucur + delta) - nloglik(betacur, sigmasqcur, taucur - delta))/(2 * delta) # correct

#

grad_nloglik(betacur, sigmasqcur, taucur)[c(2, d+1, d+2)]


###################################################################################################
### run optim (generic purpose optimization routine in R) 
###################################################################################################

obj <- function(theta){

    beta <- theta[1:d]
    sigmasq <- theta[d+1]
    tau <- theta[d+1]
    nloglik(beta, sigmasq = sigmasq, tau)
    
}    

grad_obj <- function(theta){

    beta <- theta[1:d]
    sigmasq <- theta[d+1]
    tau <- theta[d+1]
    grad_nloglik(beta, sigmasq = sigmasq, tau)

}


# Nelder-Mead method 
opt_nm <- optim(par = c(betacur, sigmasqcur, taucur), fn = obj, method = "Nelder-Mead", control = list(maxit = 1E3)
                )
#sqrt(sum((opt_nm$par[1:d] - betastar)^2))


# bfgs method
opt_bfgs <- optim(par = c(betacur, sigmasqcur, taucur), fn = obj,  method = "L-BFGS-B", control = list(maxit = 1E3),
                  lower = c(rep(-Inf, d), 0.02, 0), upper =  rep(Inf, d+2))
#sqrt(sum((opt_bfgs$par[1:d] - betastar)^2))


###################################################################################################
# In-house implementation of gradient descent
###################################################################################################

iter <- 1
maxiter <- 1000
objs <- numeric(maxiter)
objs[iter] <- nloglik(betacur, sigmasqcur, taucur)
sigmasq_min <- 0.01 # (lower bound on sigma to avoid numerical issues)
tau_min <- 0.01
gamma <- .9 # for Armijo in back-tracking line search
eta <- .25 # 2nd parameter in Armijo in back-tracking line search 
tol <- 1E-8

for(iter in 1:(maxiter-1)){

    gr <- grad_nloglik(betacur, sigmasqcur, taucur)
    gr_sigmasq <- gr[d+1]
    gr_tau <- gr[d+2]
    
    stepsize_bound <- c((sigmasqcur - sigmasq_min)/gr_sigmasq, (taucur - tau_min)/gr_tau) # maximum step size for sigmasq not to drop below sigmasq_min
    if(any(stepsize_bound > 0)) stepsize <- min(min(stepsize_bound[stepsize_bound > 0]), 1)
    

    dec <- 0
    while(stepsize > tol){

        stepsize <- stepsize * (gamma^dec)
        
        beta_new <- betacur - stepsize * gr[1:d]
        sigmasq_new <- sigmasqcur - stepsize * gr[d+1]
        tau_new <- taucur - stepsize * gr[d+2]
        upd <- c(beta_new, sigmasq_new, tau_new) - c(betacur, sigmasqcur, taucur)
        obj_new <- nloglik(beta_new, sigmasq_new, tau_new) 
        
        if( obj_new - objs[iter] > eta * sum(gr * upd) ) dec <- dec + 1
        else{

            betacur <- beta_new
            sigmasqcur <- sigmasq_new
            taucur <- tau_new
            iter <- iter + 1
            objs[iter] <- obj_new
            break
        }     
    }

    if(stepsize < tol) break 

}

objs <- objs[1:iter]
objs[iter]

# RMSE
sqrt(sum((betacur[1:d] - betastar)^2)) 

###################################################################################################
### EM algorithm --- using numerical integration 
###################################################################################################

iter <- 1
betacur <- coef(lm0) # intercept already included in X
sigmasqcur <- mean(residuals(lm0)^2)  
taucur <- sqrt(sigmasqcur)
res <- y - X %*% betacur
maxiter <- 100
objs <- numeric(maxiter)
objs[iter] <-  nloglik(betacur, sigmasqcur, taucur)
tol <- 1E-4

while(iter < maxiter){

iter <- iter + 1    

### E-step
integrals0 <- sapply(res, function(res) integrate(function(gamma) f_ymu(res - gamma, sigmasqcur) * f_gamma(gamma, taucur), lower = -5, upper = 5)$value)
integrals1 <- sapply(res, function(res) integrate(function(gamma) gamma  * f_ymu(res - gamma, sigmasqcur) * f_gamma(gamma, taucur), lower = -5, upper = 5)$value)
integrals1abs <- sapply(res, function(res) integrate(function(gamma) abs(gamma)  * f_ymu(res - gamma, sigmasqcur) * f_gamma(gamma, taucur), lower = -5, upper = 5, subdivisions = 1E3)$value)   
integrals2 <- sapply(res, function(res) integrate(function(gamma) gamma^2  * f_ymu(res - gamma, sigmasqcur) * f_gamma(gamma, taucur), lower = -5, upper = 5)$value)   

# M-step
beta_new <- solve(crossprod(X), crossprod(X, y - integrals1/integrals0))
res <- y - X %*% betacur
sigmasq_new <- sum((res^2 + (integrals2/integrals0) - 2 * res * (integrals1/integrals0)))/n     
tau_new <- mean(integrals1abs/integrals0)

# objective
objs_new <- nloglik(beta_new, sigmasq_new, tau_new)

if(objs_new < objs[iter - 1] - tol){
    objs[iter] <- objs_new
    betacur <- beta_new
    sigmasqcur <- sigmasq_new
    taucur <- tau_new
}
else break    
    
}

objs <- objs[1:(iter-1)]


# RMSE --- .1932 (similar to the solution from HW #1)
sqrt(sum((betacur[1:d] - betastar)^2)) 

###################################################################################################
###                                                                                             ### 
###################################################################################################
