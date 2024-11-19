######################################################################
# [1] Limit Behavior of Discrete Markov Chains 
######################################################################

P <- matrix(nrow = 3, ncol = 3,
            data = c(0.2, 0.3, 0.5,
                     0.5, 0.1, 0.4,
                     0.3, 0.7, 0), byrow = TRUE)

# computes the k-th power of P using recursion
powerP <- function(P, k){
  if(k == 1) return(P)
  else{      
      P %*% powerP(P, k -1)
 }     
}

powerP(P, 5)
powerP(P, 10)
powerP(P, 20)

# rows become and all identical --- invariant distribution of the markov chain:
# as k --> infty: P ~= 1 \pi^*, where pi^* is a row vector representing the stationary distribution.
# Hence, for any starting distribution pi_0, we have pi_0 P ~= pi^*.

pistar <- t(solve(t(P - diag(1, nrow = nrow(P)) + 1), rep(1, nrow(P))))

# compute total variation distance from stationary distribution

pi0 <- c(1/3, 1/3, 1/3)
sum(abs(pi0 %*% powerP(P, 5) - pistar))
sum(abs(pi0 %*% powerP(P, 10) - pistar))
sum(abs(pi0 %*% powerP(P, 20) - pistar))

# different starting distributions
pi0 <- c(1, 0, 0)
sum(abs(pi0 %*% powerP(P, 5) - pistar))
sum(abs(pi0 %*% powerP(P, 10) - pistar))
sum(abs(pi0 %*% powerP(P, 20) - pistar))

pi0 <- c(0, 1/2, 1/2)
sum(abs(pi0 %*% powerP(P, 5) - pistar))
sum(abs(pi0 %*% powerP(P, 10) - pistar))
sum(abs(pi0 %*% powerP(P, 20) - pistar))

# for the last chain track TV distance with t
ts <- 1:20
plot(ts, log(sapply(ts, function(t)  sum(abs(pi0 %*% powerP(P, t) - pistar)))))

# the TV distance decays linearly on a log-scale
# ---> convergences is exponentially fast (``geometrically ergodic")

# sampling from the invariant distribution

set.seed(223)

burnin <- 50 # burn-in period
nsamples <- 1E4

samp <- 1
S <- c(1,2,3)
for(i in 1:burnin){
  samp <- sample(S, size = 1, prob = P[samp,])
}

samples <- numeric(nsamples)
for(i in 1:nsamples){
    samp <- sample(S, size = 1, prob = P[samp,])
    samples[i] <- samp
}

table(samples)/nsamples
pistar
mean(samples)
sum(pistar * S)
var(samples)
acf(samples)

# for comparison, i.i.d. samples

samples_iid <- sample(S, size = nsamples, prob  = pistar, replace = TRUE)
mean(samples_iid)
var(samples_iid)

# note that the samples from the Markov chain are correlated --- auto-correlation function
acf(samples)
acf(samples_iid)

# example of a periodic Markov chain

P <- matrix(nr = 3, nc = 3, data = c(0,1,0, 0.5, 0, 0.5, 0, 1, 0), byrow = TRUE)

all.equal(powerP(P,1), powerP(P,3))
all.equal(powerP(P,2), powerP(P,4))


######################################################################
# [2] Discrete Metropolis Hastings  
######################################################################
ls()

#Requested Stationary distribution -> the target distribution
pi <- c(0.2,0.5,0.3)

#Proposal distribution - random walk
H <- matrix(c(0,1,0,0.5,0,0.5,0,1,0),3,3,byrow=T)

#Build the acceptance matrix
A <- matrix(NA,3,3)
for (i in 1:3) {
  for (j in 1:3) {
    A[i,j] <- min(1,pi[j]*H[j,i] /(pi[i] * H[i,j]))
  }
}
#Define 0/0 to be equal to 0
A[is.na(A)] <- 0
A

#Transmission Matrix
P <- A*H
diag(P) <- 1 - apply(P,MARGIN=1,sum) + diag(P)
P
#check for stochastic matrix:
#rowSums(P)

#Check detailed balance
idx <- as.matrix(expand.grid(1:3,1:3))
all((pi[idx[,1]] * P[idx]) == (pi[idx[,2]] * P[idx[,c(2,1)]]))

#Check the stationary distribution , should be equal to pi
matrix(rep(1,3),1,3) %*% solve(diag(rep(1,3)) - P + matrix(1,3,3))


######################################################################
# Metropolis-Hastings algorithm for discrete state space.
# Just use the transition matrix determined by the Metropolis-Hastings
# algorithms, i.e. just simulate from a Markov Chain
#
# Params:
#  X0 - initial state of the chain
#  P  - transition matrix
#  N  - length of chain to simulate
######################################################################
metropolis.hastings <- function(X0,P,N) {
  #Start of the Markov Chain
  X <- X0
  for (n in 1:N) {
    #Perform one Step of the Chain with transition matrix P
    Xnew <- sample(1:3,size=1,p=P[X[n],])
    X <- c(X,Xnew)
  }
  return(X)
}

######################################################################
# The same as metropolis.hastings, but without providing P directly
# instead the propose and accept steps are done manually
# Note: metropolis.hastings and metropolis.hastings2 are equivalent.
#
# Params:
#  X0 - initial state of the chain
#  N  - length of chain to simulate
######################################################################

metropolis.hastings2 <- function(X0,H,A,N) {
  #Start of the Markov Chain
  X <- X0
  for (n in 1:N) {
    #Proposal
    Y    <- sample(1:3,size=1,p=H[X[n],])
    #Accept?
    if (runif(1) <= A[X[n],Y]) {
      X <- c(X,Y)
    } else {
      X <- c(X,X[n])
    }
  }
  return(X)
}

#Simulate 1e4 values of the Markov chain
X <-metropolis.hastings(1,P,N=2e4)
#Remove-Burnin
X <- X[-c(1:5e3)]
#Show frequencies.
table(X)/length(X)
#Plot of the running empirical frequencies
pi.empirical <- cbind(cumsum(X == 1)/1:length(X),
                      cumsum(X == 2)/1:length(X),
                      cumsum(X == 3)/1:length(X))
matplot(pi.empirical,type="l")
legend(1e3,1.0,c("1","2","3"),col=1:3,lty=1:3)
  
#Compute the mean of pi using MCMC
mean(X)
#The true value
sum(pi*(1:3))


##############################################################################
# [3] Random Walk Metropolis Hastings for the Cauchy-Normal Bayes problem  
##############################################################################

set.seed(231)

theta0 <- -1.3

n <- 15
x <- rnorm(n, mean = -1.6, sd = 1)

# the MLE is

theta_mle <- mean(x)

#
log_likelihood <- function(theta) sapply(theta, function(z) sum(dnorm(x - z, log = TRUE)))
log_prior <- function(theta) dcauchy(theta - theta0, log = TRUE)
log_posterior_unnorm <- function(theta) log_likelihood(theta) + log_prior(theta)

# MCMC --- note that because of symmetry of the RW proposal, the acceptance probability simplies greatly 

X <- theta_mle # starting value for the Markov chain 

burnin <- 1E3 # number of samples to reach stationary distribution (estimate)
nsamp <- 2E4 + burnin
tau <- .5 # standard deviation of the RW proposal (this parameter is subject to tuning) 
samp <- numeric(nsamp)
acc  <- 0

for(i in 1:nsamp){

    Y <- rnorm(1, mean = X, sd = tau)

    # acceptance proability
    A <- exp(log_posterior_unnorm(Y) - log_posterior_unnorm(X))

    if(A > runif(1)){

        X <- Y
        acc <- acc + 1 # track number of accepted samples 
        
    }    

    samp[i] <- X 
}    

### diagnostics

samp_ret <- samp[-c(1:burnin)]
n_ret <- nsamp - burnin

# sampling path --- should be unsystematic pattern / no trends or cycles
plot(samp_ret, type = "l")
# auto-correlation function --- should see significant decay
acf(samp_ret)

# convergence of the mean
plot(1:n_ret, cumsum(samp_ret)/(1:n_ret), type = "l")

#
mean(samp_ret)

# histogram check (using numerical integration)

C0 <- integrate(function(z) exp(log_posterior_unnorm(z)), lower = -5,upper = 3)$value

hist(samp_ret, nclass = 100, prob = TRUE)
plot(function(z) exp(log_posterior_unnorm(z))/C0, from = -5, to = 3, col = "red", add = TRUE, n=1E3)

##############################################################################
# [4] MH for Gaussian Mixture Estimation
##############################################################################

library(MASS)
data(geyser)

#Show a histogram of the values
hist(geyser$waiting,nclass=30,xlab="Waiting time",prob=TRUE,main="")

######################################################################
# Density of the mixture distribution
#
# Params:
#  - mu - vector of the two means
#  - sigma - vector of the two sigma's
#  - delta - the mixture parameter
######################################################################

dMixture <- function(x,mu,sigma,delta) {
  delta*dnorm(x,mu[1],sigma[1]) + (1-delta)*dnorm(x,mu[2],sigma[2])
}

######################################################################
# Log density of the prior distribution. 
######################################################################
logdPrior <- function(mu,sigma,delta) {
  res <- dunif(delta,log=T) - log(sigma[1]) - log(sigma[2])
  return(res)
}

######################################################################
# Log density of the (unnormalized) posterior
######################################################################
logdUPost <- function(theta,x) {
  #Unpack theta
  mu    <- theta[1:2]
  sigma    <- theta[3:4]
  delta <- theta[5]
  #Posterior \propto Likelihood * Prior
  sum(log(dMixture(x,mu,sigma,delta))) + logdPrior(mu,sigma,delta)
}


######################################################################
# Random Walk Metropolis-Hastings to sample from posterior
#
# Params:
#  size - sample size
#  rwSigma - vector of length 5 with the random walk variances
######################################################################
rpost.mh1 <- function(size,rwSigma) {
  #Allocate result
  res <- matrix(NA,size,6)
  #Start value
  theta <- c(70,70,10,10,0.5)
  res[1,] <- c(theta,logdUPost(theta,x))
  accept <- numeric(5)
  
  #Loop (as usual)
  for (i in 1:(size-1)) {
    #Which component
    j <- sample(1:5,size=1)
    
    #Propose a new candidate (componentwise)
    thetanew <- theta
    thetanew[j] <- rnorm(1,theta[j],rwSigma[j])
    
    #Accept? - actually its min(1,alpha) but that can be saved
    alpha<- exp(logdUPost(thetanew,x)-logdUPost(theta,x))
    if (!is.na(alpha) && (runif(1) <= alpha)) {
      accept[j] <- accept[j] + 1
      theta <- thetanew
    }
    res[i+1,] <- c(theta,logdUPost(theta,x))
  }
  cat("Acceptance Rates (in %): ", accept/size * 100,"\n")
  return(res)
}

######################################################################
# Random Walk Metropolis-Hastings to sample from posterior
#
# Params:
#  size - sample size
#  rwSigma - vector of length 5 with the random walk variances
######################################################################
rpost.mh2 <- function(size,rwSigma) {
  #Allocate result
  res <- matrix(NA,size,6)
  #Start value
  theta <- c(70,70,10,10,0.5)
  res[1,] <- c(theta,logdUPost(theta,x))
  accept <- numeric(5)
  
  #Loop (as usual)
  for (i in 1:(size-1)) {
    #Propose a new candidate (all at once!)
    thetanew <- theta
    thetanew <- rnorm(5,theta,rwSigma)
    
    #Accept? - actually its min(1,alpha) but no need for the min
    alpha<- exp(logdUPost(thetanew,x)-logdUPost(theta,x))
    if (!is.na(alpha) && (runif(1) <= alpha)) {
      accept <- accept + 1
      theta <- thetanew
    }
    res[i+1,] <- c(theta,logdUPost(theta,x))
  }
  cat("Acceptance Rates (in %): ", accept/size * 100,"\n")
  return(res)
}

######################################################################
# Output analysis
#
# Params:
#   samples :  no Of Samples \times 6 matrix with
#              (mu1,mu2,sigma1,sigma2,delta,unorm pi)
#   burnin  :  no of Samples to throw away (if burnin=0 it probably
#              still throws away the first sample :()
######################################################################
analyze <- function(samples,burnin=0) {
  par(mfrow=c(3,2))
  #Which samples to use
  afterBurnin <- (burnin+1):dim(samples)[1]

  #Show the marginal posteriors.
  plot(samples[afterBurnin,1],type="l",ylab=expression(mu[1]))
  plot(samples[afterBurnin,2],type="l",ylab=expression(mu[2]))
  plot(samples[afterBurnin,3],type="l",ylab=expression(sigma[1]))
  plot(samples[afterBurnin,4],type="l",ylab=expression(sigma[2]))
  plot(samples[afterBurnin,5],type="l",ylab=expression(delta))
  plot(samples[afterBurnin,6],type="l",ylab="log(un. posterior)")
  par(mfcol=c(1,1))

  ##Marginal Posterior mean
  cat("Posterior mean  : ",apply(samples[afterBurnin,],2,mean)[-6],"\n")

  #Marginal Posterior median
  cat("Posterior median: ",apply(samples[afterBurnin,],2,median)[-6],"\n")

  #Posterior mode (joint) (monte-carlo optimization)
  cat("Posterior mode  : ",samples[which.max(samples[,6]),][-6],"\n")
}



#Set seed
set.seed(1234)
options("warn"=-1) ### suppresses warnings --- log's of negative numbers

#Load the data
x <- geyser$wait

rwSigma <- c(2,2,2,2,0.1)
samples <- rpost.mh1(10000,rwSigma)
analyze(samples,burnin=0)

rwSigma <- c(0.5,0.5,0.5,0.5,0.05)
samples <- rpost.mh1(10000,rwSigma)
analyze(samples,burnin=0)


rwSigma <- c(0.5,0.5,0.5,0.5,0.05)
samples <- rpost.mh2(10000,rwSigma)

analyze(samples,burnin=0)
analyze(samples,burnin=3000)

##############################################################################
# [5] Gibbs Sampler for the bivariate Normal distribution 
##############################################################################

library(mvtnorm)

######################################################################
# Sample size values from the bivariate normal with correlation rho.
#
# Params:
#  size - sample size.
#  rho  - the correlation of the bivariate normal.
######################################################################

rnorm2.gibbs<-function(size, rho) {
  #Allocate matrix
  res <- matrix(ncol = 2, nrow = size)
  #Start value
  x <- 0
  y <- 0
  res[1, ] <- c(x, y)

  #Loop
  for (i in 2:size) {
    #x-component
    x <- rnorm(1, rho * y, sqrt(1 - rho^2))
    #y-component -- important, the new x-value is used
    y <- rnorm(1, rho * x, sqrt(1 - rho^2))
    #put current iteration in result matrix
    res[i, ] <- c(x, y)
  }
  #Done
  return(res)
}

######################################################################
# Alternative: draw direct using factorization
######################################################################

rnorm2.factor <- function (size, rho) 
{
  x <- rnorm(size, 0, 1)
  y <- rnorm(size, rho * x, sqrt(1 - rho^2))
  cbind(x, y)
}

######################################################################
# Illustrate the moves of a two-stage Gibbs sampler in the normal
# example
######################################################################
plot.moves <- function(sample,rho,howmany=10,cWise=TRUE,...) {
  #Size
  cex <- 1
  #Level curves
  Sigma <- matrix(c(1,rho,rho,1),2,2)
  #Create and evaluate grid
  x.grid <- seq(-2,2,length=50)
  y.grid <- seq(-2,2,length=50)
  grid <- expand.grid(x.grid,y.grid)

  #Only eval if really necessary
  #if (is.null(dens)) {
    dens <<- matrix(apply(grid,MAR=1,dmvnorm,mean=rep(0,2),sigma=Sigma),50,50)
  #}
  #Show contour plot
  contour(x.grid,y.grid,dens,col=3,xlab="x",ylab="y",nlevels=5,...)

  #Draw every move (first the x-axis move, then the y-axis)
  for (i in 1:howmany) {
    if (cWise) {
      lines(c(sample[i,1],sample[i+1,1]),c(sample[i,2],sample[i,2]),lty=2)
      lines(c(sample[i+1,1],sample[i+1,1]),c(sample[i,2],sample[i+1,2]),lty=2)
      text(sample[i,1],sample[i,2],substitute(theta[i],list(i=i)),col=2,cex=cex)
      text(sample[i+1,1],sample[i,2],substitute(theta[i]^"*",list(i=i)),col=2,cex=cex)
    } else {
      lines(c(sample[i,1],sample[i+1,1]),c(sample[i,2],sample[i+1,2]),lty=2)
      text(sample[i,1],sample[i,2],substitute(theta[i],list(i=i)),col=2,cex=cex)
    }
    
  }
}


######################################################################
# Function to illustrate a sample using time series plots, scatter
# plots etc.
######################################################################
show.sample <- function(sample) {
  plot(sample,col=1:10000)
  par(mfrow=c(3,2))
  plot(ts(sample[,1]))
  plot(ts(sample[,2]))
  acf(sample[,1])
  acf(sample[,2])
  hist(sample[,1],40,xlab="x",main="")
  hist(sample[,2],40,xlab="y",main="")
  par(mfrow=c(1,1))
}

#Draw values
set.seed(12345)
size <- 1e4

rho  <- 0.4
gibbs  <- rnorm2.gibbs(size,rho)
plot.moves(gibbs,rho=rho,howmany=6,cWise=TRUE,main="Gibbs-Sampler")

set.seed(12345)
rho <- 0.98
gibbs  <- rnorm2.gibbs(size,rho)
factor <- rnorm2.factor(size,rho)

#Show the moves
plot.moves(gibbs,rho=rho,howmany=6,cWise=TRUE,main="Gibbs-Sampler")
plot.moves(factor,rho=rho,howmany=6,cWise=FALSE,main="f(x)f(y|x) sampler")


#Show the sample
show.sample(gibbs)
show.sample(factor)

##############################################################################
# [6] Gibbs Sampler for Poisson change point detection
##############################################################################

coal <- read.table("coal.txt",header=T)
with(coal,plot(year,disasters,type="h",xlab="Jahr",ylab="Number of Accidents"))

###################################################
###################################################
######################################################################
# Function to sample from the posterior of the changepoint model
# using a Gibbs sampler.
#
# Params:
#  size - number of samples to draw 
#  x    - the data drawn from the changepoint model (vector)
#
# Returns:
#   size x 4 matrix containing (lambda1,lambda2,alpha,theta) for each
######################################################################
rpost.gibbs <- function(size,x) {
  #Times points
  n <- length(x)
  time <- 1:n
  #Allocate result matrix
  samples <- matrix(NA,size,4)
  dimnames(samples) <- list(NULL,c("lambda1","lambda2","alpha","theta"))

  #Start values
  lambda <- c(1,1)
  alpha <- 1
  theta <- round(n/2)
  samples[1,] <- c(lambda,alpha,theta)

  #Do the Gibbs sampler
  for (i in 2:size) {
    #The full conditionals for lambda[1],lambda[2] and alpha
    lambda[1] <- rgamma(1,3+sum(x[time[time<=theta]]),theta+alpha)
    lambda[2] <- rgamma(1,3+sum(x[time[time>theta]]),(n-theta)+alpha)
    alpha     <- rgamma(1,16,10+sum(lambda))
    #Calculate the probs for the full conditional of theta
    #theta.prob <- sapply(time,function(theta) 
    #  exp(theta*(lambda[2]-lambda[1])) *
    #    exp(sum(x[time[time<=theta]]*log(lambda[1]/lambda[2])))
    #})
    theta.prob <- exp(time * (lambda[2]-lambda[1])) *
      exp(cumsum(x)*log(lambda[1]/lambda[2]))
    #Do the draw
    theta <- sample(time,size=1,prob=theta.prob)

    #Save result.
    samples[i,] <- c(lambda,alpha,theta)
  }
  return(samples)
}



###################################################
###################################################
samples <- rpost.gibbs(1e4,x=coal$disaster)
par(mfrow=c(2,2))
plot(samples[,"lambda1"],type="l")
plot(samples[,"lambda2"],type="l")
acf(samples[,"lambda1"])
acf(samples[,"lambda2"])
par(mfcol=c(1,1))



###################################################
###################################################
burnin <- 1:150
par(mfrow=c(2,2))
#theta
plot(samples[-burnin,"theta"],type="l",ylab=expression(theta))
hist(samples[-burnin,"theta"],nclass=100,xlab=expression(theta))
#lambda1
plot(density(samples[-burnin,"lambda1"]),main="")
#lambda2
plot(density(samples[-burnin,"lambda2"]),main="")
par(mfcol=c(1,1))

#Calc mean for all
apply(samples[-burnin,],MAR=2,mean)
apply(samples[-burnin,],MAR=2,median)
