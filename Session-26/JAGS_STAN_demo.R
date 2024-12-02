###############################################################################
### Illustration of JAGS (via rjags) and STAN (via rstan)
###
### Test data: Old Faithful Geyeser waiting times (for Gaussian mixture model)
###
################################################################################

library(MASS)
data(geyser)

X <- geyser$waiting

hist(geyser$waiting,nclass=30,xlab="Waiting time",prob=TRUE,main="")


################################################################################
### JAGS -- specification without latent variables
################################################################################

library(rjags)
load.module("mix")

jm <- jags.model(file = "mixture_model.txt", data = list(X = X, n = length(X)),
                 n.chains = 1, n.adapt=2000, quiet=FALSE, inits = list(mu0 = 55, mu1 = 80, sigma0inv = 1/10, sigma1inv = 1/10,  delta = 0.5))

samp <- coda.samples(jm, variable.names = c("mu0", "mu1", "sigmainvs", "delta"),
                     n.iter = 10000)

summary(samp)

################################################################################
### JAGS -- specification with latent variables
################################################################################

jm_2 <- jags.model(file = "mixture_model_m.txt", data = list(X = X, n = length(X)),
                 n.chains = 1, n.adapt=2000, quiet=FALSE, inits = list(mu0 = 55, mu1 = 80, sigma0inv = 1/10, sigma1inv = 1/10,  delta = 0.5))


samp_2 <- coda.samples(jm_2, variable.names = c("mu0", "mu1", "sigmainvs", "delta"),
                       n.iter = 10000)

summary(samp_2)

# note that the sampler using the latent variable representation (data
# augmentation) provides reasonable results (unlike the first attempt),
# and runs much faster.

# Note that the the entires of "sigmainvs" correspond to the precision
# (i.e., reciprocal variance). To convert to the scale of standard deviations:   

cnames <- colnames(samp_2[[1]])
apply(sqrt(1/samp_2[[1]][,grep("sigma", cnames)]), 2, summary) 

################################################################################
### STAN --- discrete variables are not allowed, hence only
###          specification w/o latent variables 
################################################################################

library(rstan)
fit <- stan(file = 'mixture_model.stan', data = list(X = X, N = length(X), K = 2))
print(fit)
