###################################################
### 
######################################################################

set.seed(12345)

#Load necessary libs
library(cubature) # multivariate numerical integration 
library(coda)  # convergence diagnostics for MCMC



###################################################
### 
###################################################
######################################################################
# Unormalized posterior in the Chen et al (2000) paper
#
# Params:
#  x = c(z,theta)
#  sigma = the observation is y ~ N(theta*z, sigma^2),
#  log = if TRUE return log posterior
#
# Returns:
#  value of the posterior at psi
######################################################################
dpost.un <- function(x,sigma=0.05,log=FALSE) {
  z <- x[1]
  theta <- x[2]

  #Flat prior for theta
  ifelse(log,
         dnorm(y,mean=theta*z,sd=sigma,log=TRUE) + dnorm(z,theta,1,log=TRUE),
         dnorm(y,mean=theta*z,sd=sigma) * dnorm(z,theta,1) * 1)
}

######################################################################
# Function to create contour plot of the posterior for one sigma
#
# Params:
#  sigma - 
#  gridsize - the size of the grid will be gridsize * gridsize
######################################################################

oneplot <- function(sigma,gridsize=100,...) {
  grid.z <- seq(0,2.5,length=gridsize)
  grid.theta <- seq(0,2.5,length=gridsize)
  unpost <- matrix(NA,nrow=length(grid.z),ncol=length(grid.theta))

  #Evaluate the grid
  for (i in 1:length(grid.z)) {
    for (j in 1:length(grid.theta)) {
      unpost[i,j] <- dpost.un(c(grid.z[i],grid.theta[j]),sigma=sigma)
    }
  }

  #Simple 2d Riemann approximation of the normalization constant
  #if (is.null(norm)) {
  #  dz <- diff(grid.z)[1]
  #  dtheta <- diff(grid.theta)[1]
  #  norm <- sum(unpost * (dz*dtheta))

  #Better: use cubintegrate to calculate normalization constant
  norm <- cubintegrate(f = dpost.un, lower=rep(0,2),rep(2.5,2),,sigma=sigma)$integral
  
  post <- unpost/norm
  
  contour(grid.z,grid.theta,post,main=substitute(sigma*"="*z,list(z=sigma)),xlab="z",ylab=expression(theta),...)

  invisible()
}

#The data 
y <- 1


sigma <- c(1,0.5,0.2,0.05)
#pdf("hmc_example_contours_1.pdf")
par(mfrow=c(2,2),mar=c(4,4,3,1))
foo <- sapply(sigma,oneplot)
#dev.off()


###################################################
### 
###################################################

pdf("hmc_example_contours_2.pdf")
par(mar=c(mar=c(4,4,1,1)))
par(mfrow = c(1,1))
oneplot(sigma=0.05,levels=0.1,xlim=c(0.4,2.5))

#Metropolis-Hastings
psi <- c(2,0.55)
points(psi[1],psi[2],col=2,pch=16)
noAcc <- 0

for (i in 1:40) {
  #Proposal and its acceptance prob
  psi.star <- psi + rnorm(2,sd=0.1)
  acc <- dpost.un(psi.star)/dpost.un(psi)

  #If accepted draw a line
  if (runif(1) < acc) {
    segments(psi[1],psi[2],psi.star[1],psi.star[2],col=2)
    psi <- psi.star
    noAcc <- noAcc+1
  }
}

dev.off()

###################################################
###  
###################################################

#Example for the standard normal
thetaIdx <- 1
vIdx   <- 2

#Energy function
E <- function(theta) { 1/2 * theta^2 }

#gradient of the energy function
#in Maxima:  E: 1/2*(x-1)^2; diff(E,x);
gradE <- function(theta) { theta }

#Hamiltonian function                  
H <- function(psi) {
  E(psi[thetaIdx]) +   sum(1/2*psi[vIdx]^2)
}

#educational version of the leapfrog algorithm 
leapfrog.edu <- function(psi,eta=0.2,L=10,col=2,arrows=FALSE) {
  theta <- psi[thetaIdx]
  v   <- psi[vIdx]

  #This is just for drawing
  points(theta,v,col=col,pch=16)
  theta.old <- theta
  v.old <- v
  
  #Leapfrog steps (the non efficient way)
  for (j in 1:L) {
    v   <- v - eta/2 * gradE(theta)
    theta  <- theta + eta * v
    v    <- v - eta/2 * gradE(theta)

    #Line drawing 
    if (arrows) {
      arrows(theta.old,v.old,theta,v,col=col,length=.1)
    } else {
      segments(theta.old,v.old,theta,v,col=col)
    }

    points(theta,v,col=col)
    theta.old <- theta
    v.old <- v
  }

  return(c(theta,v))
}


#Grid
gridsize <- 100
grid.theta <- seq(-4,4,length=gridsize)
grid.v   <- seq(-4,4,length=gridsize)
Hgrid <- matrix(NA,nrow=length(grid.theta),ncol=length(grid.v))

for (i in 1:length(grid.theta)) {
  for (j in 1:length(grid.v)) {
    Hgrid[i,j] <- H(c(grid.theta[i],grid.v[j]))
  }
}

pdf("illustration_leapfrog.pdf")
options(width = 60)
par(mar=c(mar=c(4,4,1,1)))
contour(grid.theta,grid.v,Hgrid,main=expression("Contour of H("*theta*","*v*")"),xlab=expression(theta),ylab=expression(v))
                  
##Start value
psi <- c(1,1)
leapfrog.edu(psi,eta=0.08,L=50,arrows=TRUE)

psi <- c(-2.2,2.2)
leapfrog.edu(psi,eta=0.4,L=12,col=3,arrows=TRUE)
dev.off()


###################################################
### 
###################################################


######################################################################
#Hamiltonian function in the example
# 
# Params:
#  psi - c(x,theta, rho)
#  sigma - the value of sigma to use
######################################################################
H <- function(psi,sigma=0.05) {
  -dpost.un(psi[1:2],log=TRUE,sigma) + 1/2 * sum(psi[3:4]^2)
}

######################################################################
#Gradient of E 
# f : -(-1/2/sigma2*(1-theta*z)^2 - 1/2*(z-theta)^2); diff(f,z);diff(f,theta);
#
#Params:
# bmx - c(x,theta)
# sigma - 
######################################################################
gradE <- function(x,sigma=0.05) {
  z <- x[1]
  theta <- x[2]
  c( - theta*(1-theta*z)/sigma^2 + z -theta, -z*(1-theta*z)/sigma^2-z+theta)
}
  
######################################################################
# The leapfrog algorithm
#
# Params:
#  psi     - current position
#  epsilon - step length of leapfrog algo
#  L       - number of steps
######################################################################
leapfrog <- function(psi, eta=0.005,L=1000) {
  theta <- psi[1:2]
  v   <- psi[3:4]

  v <- v - eta/2 * gradE(theta)
  for (j in 1:L) {
    theta <- theta + eta * v
    v <- v - eta * gradE(theta)
  }
  v <- v + eta/2 * gradE(theta)

  return(c(theta,v))
}

######################################################################
# Sample from dpost using hybrid mcmc
#
# Params:
#  n - number of samples
#  psi0 - start value c(x,theta)
######################################################################
rhybridmcmc <- function(n,theta0=c(2,0.55), eta=0.005,L=1000) {
  psi <- c(theta0,0,0)
  log <- matrix(NA,nrow=n,ncol=length(psi))
  log[1,] <- psi
  t <- 2
  
  while (t<=n) {
    #Momentum variables -- gibbs step
    v <- rnorm(2,0,1)

    #Leapfrog
    psi.star <- leapfrog(c(psi[1:2], v), eta = eta, L=L)
    
    #Accept it?
    acc <- exp( -(H(psi.star) - H(psi)))
    if (runif(1) < acc) psi <- psi.star

    #Bookkeeping
    log[t,] <- psi
    t <- t+1
  }
  return(log[,1:2])
}

theta <- rhybridmcmc(n=40)

pdf("HMC_run.pdf")
par(mar=c(mar=c(4,4,1,1)))
options(width = 60)
oneplot(sigma=0.05,levels=0.1,xlim=c(0.4,2.5))
lines(theta[,1],theta[,2],col=2)
points(theta[,1],theta[,2],col=2)
dev.off()
#convert to coda object
theta <- mcmc(theta)


