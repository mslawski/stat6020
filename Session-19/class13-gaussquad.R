###################################################
### 
###################################################
######################################################################
######################################################################
ls()
options(width=60)
set.seed(12345)

#Load necessary libs
library(statmod)
library(polynom)
library(orthopolynom)



###################################################
### 
###################################################
orthopoly.plot <- function(p.list,main,xlim,ylim) {
  #Create a plot
  legstr <- NULL
  plot(NA,xlim=xlim,ylim=ylim,type="n",xlab="",ylab="",main=main)
  for (k in 1:length(p.list)) {
    lines(p.list[[k]],xlim=xlim,len=1000,col=k)
    legstr <- c(legstr,eval(substitute(expression(p[k](x)),list(k=k-1))))
  }
  #Annotate
  lines(c(-1e99,1e99),rep(0,2),col=1,lty=2)
  legend(x="bottom",legstr,col=1:length(p.list),ncol=3,lty=1)
}


make.monic <- function(p) { 
  a <- coef(p)
  as.polynomial(a/a[length(a)])
}
p.list <- sapply(legendre.polynomials( 5, normalized=FALSE),make.monic)
orthopoly.plot(p.list,main="Legendre ",xlim=c(-1,1),ylim=c(-1.2,1.2))


###################################################
### 
###################################################
p.list <- sapply(hermite.h.polynomials( 5, normalized=FALSE),make.monic)


orthopoly.plot(p.list,main="Hermite",xlim=c(-2,2),ylim=c(-6,6))



###################################################
### 
###################################################
library(statmod)
######################################################################
# Illustrate roots of the orthogonal polynomials, i.e. show the roots
# for a sequence of poly orders.
#
# Params:
#  xlim - limits on the xaxis c(lower,upper)
#  jmax - max poly order
#  type - "hermite" or "legendre". See also the help of gauss.quad
######################################################################

show.roots <- function(xlim,jmax=20,type="hermite") {
  plot(NA,xlim=xlim,ylim=c(0,jmax),ylab="order",main=type,xlab="")
  points(0,1)
  for (order in 2:jmax) {
    quad <- gauss.quad(order,type)
    points(quad$nodes,rep(order,order))
  }
}

# PLOT
#pdf("roots_legendre.pdf")
#show.roots(xlim=c(-1,1),jmax=20,type="legendre")
#dev.off()


###################################################
### 
###################################################
#pdf("roots_hermite.pdf")
#show.roots(xlim=c(-6,6),jmax=20,type="hermite")
#dev.off()

###################################################
### chunk number 6: 
###################################################
gauss.quad(4+1,"legendre")
gauss.quad(4+1,"hermite")



###################################################
### 
###################################################
m <- 2


###################################################
### 
###################################################
######################################################################
#Integrate f(x) from -1 to 1
######################################################################
f <- function(x) {
 x^(2*m+1)
}


###################################################
### 
###################################################
#Use Gauss-Legendre
gq <- gauss.quad(m+1,"legendre")
sum(gq$weights * f(gq$nodes))
#The competitor
integrate(f,lower=-1,upper=1)


###################################################
### 
###################################################
######################################################################
# Function to compute the integral \int_a^b f(x) dx using a
# m+1 Point Gauss-Legendre
#
# Params:
#   f  - function to evaluate
#   a  - lower limit
#   b  - upper limit
#   m  - number of quadrature points (+1)
######################################################################
int.gauss.legendre <- function(f,a,b,m) {
  #Compute m+1 Gauss quadrature
  quad <- gauss.quad(m+1,"legendre")
  #Insert substitution transforming the problem into an
  #ordinary Gauss-Legendre from -1 to 1
  (b-a)/2 * sum(quad$weights * f((b-a)/2*quad$nodes + (b+a)/2))
}

#Integrate f(x) from -5 to 1
int.gauss.legendre(f,-5,3,m=4)
#The competitor
integrate(f,lower=-5,upper=3)


# test function from Newton-Cotes:

testf <- function(x) 4*sin(x)^2 + 0.5 * x^2 * cos(x)
plot(testf, from = 0, to = 2*pi)
# closed form solution
anti1 <- function(x) 4 * (0.5*x - sin(2*x)/4)
anti2 <- function(x) 0.5 * (x^2 * sin(x) + 2*x * cos(x) - 2*sin(x))
valstar <- anti1(2*pi)- anti1(0) + anti2(2*pi) - anti2(0)

abs(int.gauss.legendre(testf,0,2*pi,m=10) - valstar)
# we have an excellent approximation using only 10 support points

###################################################
### chunk number 11: 
###################################################
######################################################################
# Integration for an integrant where a Gaussian weight function
# is appropriate. I.e. use substitution transferring problem
# into a Gauss-Hermite quadrature.
#
# Params:
#   f  - function to evaluate f(x,...)
#   a  - lower limit
#   b  - upper limit
#   m  - number of quadrature points (+1)
######################################################################

int.gauss.norm <- function(f,mu,sigma,m,...) {
  #Compute nodes and weights in a 
  quad <- gauss.quad(m+1,"hermite")

  #Result of a substitution formula to get back
  #to Gauss-Hermite case
  Ap <- quad$weights * exp(quad$nodes^2) * sqrt(2) *sigma
  z <- mu + sqrt(2) * sigma * quad$nodes

  #Approximate the integral
  return(sum(Ap * f(z,...)))
}

mu <- 3
sigma <- 2
E4<- int.gauss.norm(function(x) x^4 * dnorm(x, mu, sigma),mu,sigma,m=3)
integrate(function(x) x^4 * dnorm(x, mu, sigma), -Inf, Inf)$value


#E  <- int.gauss.norm(function(x) x*dnorm(x,mu,sigma),mu,sigma,m=0) 

#E
#E2 - E^2



