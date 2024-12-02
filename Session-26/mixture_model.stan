data { 
	int<lower=1> K;  // number of mixture components 
	int<lower=1> N;  // number of data points 
	array[N] real X; // observations 
} 

parameters { 
simplex[K] theta; // mixing proportions
ordered[K] mu; // locations of mixture components 
vector<lower=0>[K] sigma; // scales of mixture components 

}

model { 
vector[K] log_theta = log(theta); // cache log calculation 
sigma ~ lognormal(0, 100); 
mu ~ normal(0, 100); 
for (n in 1:N){ 
vector[K] lps = log_theta; 
for (k in 1:K) { 
lps[k] += normal_lpdf(X[n] | mu[k], sigma[k]); 
} 
target += log_sum_exp(lps); 
} 

}