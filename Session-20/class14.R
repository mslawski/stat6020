######################################################################################################
### 
######################################################################################################

ls()


######################################################################################################
### [1] Rejection Sampling --- Gamma distribution
######################################################################################################


a <- 2.5
b <- 1.4

# the optimum is attained for x >= 1, hence we use the expression
# below to find the maximizer of the density ratio

xstar <- (a + 2 - 1)/b

opt <- optimize(function(x) dgamma(x,log = TRUE, shape = a, rate = b) - log(2*dcauchy(x)), lower = 0, upper = xstar * 5, maximum = TRUE)

M <- exp(opt$objective)

### acceptance ratio 1/M

Mbar_fun <- function(x) exp( (a + 2 - 1) * log(x) - b * x + log(pi) + a  * log(b) - lgamma(a) )
Mbar <- Mbar_fun(xstar)

### rejection sampling algorithm

set.seed(1047)
npropose <- 5*1E4

Y <- abs(rcauchy(npropose))
X <- Y[(dgamma(Y, shape = a, rate = b)/(Mbar * 2 * dcauchy(Y))) > runif(npropose)]

# check how many samples got accepted --- acceptance rate is about 25%
length(X)
5E4/Mbarvv

# check for correctness

# check 1 --- means
mean(X)
a/b

# check 2 --- agreement of histograms and PDF
hist(X, nclass = 100, freq = FALSE)
plot(function(x) dgamma(x, shape = a, rate = b), from = 0, to = 10, add = TRUE, col = "blue") 


######################################################################################################
### [2] Rejection sampling for Bayesian inference
######################################################################################################

set.seed(231)

theta0 <- -1.3

n <- 15
X <- rnorm(n, mean = -1.6, sd = 1)

# the MLE is

theta_mle <- mean(X)

#
log_likelihood <- function(theta) sapply(theta, function(z) sum(dnorm(X - z, log = TRUE)))
log_prior <- function(theta) dcauchy(theta - theta0, log = TRUE)
log_posterior_unnorm <- function(theta) log_likelihood(theta) + log_prior(theta) 

C <- integrate(function(z) exp(log_posterior_unnorm(z)), low = -5, up = 5)$value

plot(function(theta) exp(log_posterior_unnorm(theta))/C, from = -4, to = 3, n = 1E3)
plot(function(theta) exp(log_prior(theta)), from = -4, to = 3, n = 1E3, add = TRUE, col = "blue")


M <- exp(log_likelihood(theta_mle))

nsamples <- 1E4
batchsize <- 1E4
store_samples <- c()
acc_count <- 0   # numer of accepted samples
gen_count <- 0   # number of samples generated from proposal  

while(TRUE){

    Y <- rcauchy(batchsize) + theta0
    acc <- exp(log_likelihood(Y))/M > runif(batchsize)
    store_samples <- c(store_samples, Y[acc])
    gen_count <- gen_count + batchsize
    acc_count <- acc_count + sum(acc)

    if(length(store_samples) >= nsamples)
        break
}

length(store_samples)
acc_count / gen_count # acceptance probability is fairly decent

hist(store_samples, nclass= 100, prob = TRUE)    
plot(function(theta) exp(log_posterior_unnorm(theta))/C, from = -4, to = 3, n = 1E3, add = TRUE, col = "red")

mean(store_samples) # posterior mean
median(store_samples)
theta_mle # MLE is only slightly different from posterior median/mean

# for comparison, compute posterior mean via numerical integration
C1 <- integrate(function(z) exp(log_posterior_unnorm(z)) * z, low = -5, up = 5)$value
C1 / C0

######################################################################################################
### [3] Importance Sampling: Monte Carlo estimates of Cauchy tail probabilities.
######################################################################################################

# "naive"

samples_naive <- replicate(n = 100, rcauchy(n = 1000))
est_naive <- apply(samples_naive, 1, function(z) mean((abs(z) > 2))/2) 

MSE_naive <- mean((est_naive - (1 - pcauchy(2)))^2)

# improvement
samples_unif <- replicate(n = 100, runif(n = 1000, 0, 2))
est_unif <- apply(samples_unif, 1, function(z) 0.5 - 2 * mean(dcauchy(z))) 
MSE_unif <- mean((est_unif - (1 - pcauchy(2)))^2)

MSE_naive/MSE_unif

# major improvement
samples_unif2 <- replicate(n = 100, runif(n = 1000, 0, 1/2))
est_unif2 <- apply(samples_unif2, 1, function(z) 0.5 * mean(dcauchy(z))) 
MSE_unif2 <- mean((est_unif2 - (1 - pcauchy(2)))^2)

MSE_naive/MSE_unif2

######################################################################################################
### [4] Importance Sampling: posterior mean in the Cauchy-Normal Bayes problem
######################################################################################################

### Suppose we want to estimate the posterior mean in the Cauchy-Normal
### Bayes problem, using the prior as a proposal. 

Y <- rcauchy(1E4) + theta0
denom <- exp(log_posterior_unnorm(Y))/dcauchy(Y, theta0)
num <-  denom * Y

mean(num)/mean(denom)

# from rejection sampling 
mean(store_samples)

######################################################################################################
### [5] Gamma example revisited: Adaptive Rejection Sampling (ARS)
######################################################################################################
set.seed(439)

library(ars)

a <- 2.5
b <- 1.4

# derivative of the log PDF 
logfprime <- function(x) (a - 1) * (1/x) - b 

# alternative --- numerical differentiation

#fprime <- function(x, delta = 0.0001) (dgamma(x + delta, shape = a, rate = b) -  dgamma(x - delta, shape = a, rate = b))/(2*delta)
#logfprime <- function(x) fprime(x) / dgamma(x, shape = a, rate = b)

mysample1 <- ars(1E4, function(x) dgamma(x, log = TRUE, shape = a, rate = b), fprima = logfprime, x= c(.5,1,1.5,2), lb=TRUE, xlb=0)

mean(mysample1)
a / b

hist(mysample1, nclass = 100, prob = TRUE)
plot(function(x) dgamma(x, shape = a, rate = b), from = 0, to = 10, add = TRUE, col = "red")
