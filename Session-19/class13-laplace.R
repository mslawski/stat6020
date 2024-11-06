######################################################################
# Bayesian Computation:
#
# Laplace approximation in the Cauchy-Normal model 
#
######################################################################
ls()
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





# gold standard --- find posterior mean by numerical integration
C0 <- integrate(function(z) exp(log_posterior_unnorm(z)), low = -5, up = 5)$value

plot(function(theta) exp(log_posterior_unnorm(theta))/C0, from = -4, to = 3, n = 1E3)
plot(function(theta) exp(log_prior(theta)), from = -4, to = 3, n = 1E3, add = TRUE, col = "blue")

C1 <- integrate(function(z) exp(log_posterior_unnorm(z)) * z, low = -5, up = 5)$value



Epost <- C1/C0

C2 <- integrate(function(z) exp(log_posterior_unnorm(z)) * (z - Epost)^2, low = -5, up = 5)$value

Varpost <- C2/C0

# Laplace approximation for posterior variance 

# Step 1: define logs of the function to be integrate 
h0 <- function(theta) log_posterior_unnorm(theta)
h2 <- function(theta) log_posterior_unnorm(theta) +  log((theta - Epost)^2) 

# Step 2: Laplace approximation for the normalization constant

opt0 <- optimize(h0, interval = c(-5, 5), maximum = TRUE) # this finds the posterior mode
theta_hat <- opt0$maximum

# approximate 2nd derivative numerically:
delta <- 0.001
sec_deriv0 <- (h0(theta_hat + delta) - 2 * h0(theta_hat) +   h0(theta_hat - delta)) / delta^2  

C0hat <- sqrt(2 * pi) * exp(h0(theta_hat))/sqrt(-sec_deriv0)  # this is rather close 

# Step 3: Laplace approximation for the integral defining the variance

opt2 <- optimize(h2, interval = c(-5, 5), maximum = TRUE) 
theta2_hat <- opt2$maximum

sec_deriv2 <- (h2(theta2_hat + delta) - 2 * h2(theta2_hat) +   h2(theta2_hat - delta)) / delta^2  

C2hat <- sqrt(2 * pi) * exp(h2(theta2_hat))/sqrt(-sec_deriv2)
# unfortunately, variance is under-appromixated quite a bit 
