######################################################################
#  
######################################################################

testf <- function(x) 4*sin(x)^2 + 0.5 * x^2 * cos(x)

plot(testf, from = 0, to = 2*pi)

# closed form solution

anti1 <- function(x) 4 * (0.5*x - sin(2*x)/4)
anti2 <- function(x) 0.5 * (x^2 * sin(x) + 2*x * cos(x) - 2*sin(x))

valstar <- anti1(2*pi)- anti1(0) + anti2(2*pi) - anti2(0)

# R numerical integration routine

integrate(testf, 0, 2 * pi)

######################################################################

######################################################################
# Newton Cotes formulas
######################################################################

######################################################################
# Riemann -- integrate using the riemann integral
#
# Params:
#  f - a function taking one parameter, i.e. of type f(x)
#  a - lower limit
#  b - upper limit
#  n - number of intervals to use (i.e. n+1 points are used)
#  ... - extra arguments for f
######################################################################

riemann <- function(f,a,b,m, type = c("left", "right", "mid"), ...) {
  #Width of each interval
  h    <- (b-a)/m
  #Define the grid 
  x <- seq(a,b,by=h)
  #Function values
  type <- match.arg(type)   
  # left -- left endpoints, right -- right end points, mid -- midpoint  
  fval <- switch(type, left = f(x[-(m+1)], ...),
                 right = f(x[-1], ...),
                 mid = f((x[1:(m-1)] + x[2:m])/2, ...))
    
  sum(fval * h)
 
 
}


######################################################################
# Trapezoidal -- integrate using the trapezoidal rule
#                     \int_a^b f(x) dx
# Params:
#  f - a function taking one parameter, i.e. of type f(x)
#  a - lower limit
#  b - upper limit
#  n - number of intervals to use (i.e. n+1 points are used)
#  ... - extra arguments for f
######################################################################

trapezoidal <- function(f,a,b,n,...) {
  #Width of each interval
  h    <- (b-a)/n
  #Define the grid 
  x <- seq(a,b,by=h)
  #Function values
  fval <- f(x,...)
  #Trapezoidal rule
  sum(1/2*h*fval[c(1,n+1)],h*fval[2:n])
}


######################################################################
# Integrate using Simpon's rule, i.e. approximate \int_a^b f(x) dx
#
# Params:
#  f - a function taking one parameter, i.e. of type f(x)
#  a - lower limit
#  b - upper limit
#  n - number of intervals to use (has to be even!!)
#  ... - extra arguments for f
######################################################################

simpson <- function(f,a,b,n,...) {
  #Check that n is even
  if ( n %% 2 != 0) { return("Error: n is not even!")}
  
  #Width of each interval
  h    <- (b-a)/n
  #Define the grid 
  x <- seq(a,b,by=h)
  #Function values
  fval <- f(x,...)

  #Indices to take
  i <- 1:(n/2)
  #Simpon's rule
  h/3 * sum(c(fval[2*i-2+1],4*fval[2*i-1+1],fval[2*i+1]))
}




######################################################################
#
# Benchmarking
#
######################################################################

abs( riemann(testf,a=0,b=2 * pi,m=100, "left") - valstar)
abs( riemann(testf,a=0,b=2 * pi,m=100, "right") - valstar)
abs( riemann(testf,a=0,b=2 * pi,m=100, "mid") - valstar)

abs( riemann(testf,a=0,b=2 * pi,m=1000, "left") - valstar)
abs( riemann(testf,a=0,b=2 * pi,m=1000, "right") - valstar)
abs( riemann(testf,a=0,b=2 * pi,m=1000, "mid") - valstar)


abs(trapezoidal(testf,a=0,b=2*pi,n=100) - valstar)
abs(trapezoidal(testf,a=0,b=2*pi,n=1000) - valstar)

abs(simpson(testf,a=0,b=2*pi,n=100) - valstar)
abs(simpson(testf,a=0,b=2*pi,n=1000) - valstar)
