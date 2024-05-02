import numpy as np
#from model.graphs import SBM, balanced_partition_pin_pout
from model.graph import *
from experiment import rct
from model import pom
from experiment.estimators import estimator

### Network parameters
n = 1000         # population size
nc = 50         # number of clusters
p_in = 0.5      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
p_out = 0       # probability of edge between units of different clusters

### Other parameters
beta = 2        # potential outcomes model degree
p = 0.5         # treatment budget (marginal treatment probability)     
gr = 10          # number of graph repetitions
r = 100       # number of RCT repetitions
cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

### Experiment
bias = np.zeros(gr)
variance = np.zeros(gr)
for g in range(gr):
    G = SimpleSBM(n, nc, p_in, p_out)
    TTE_true, outcomes = pom.ppom(beta, G, cf)
    #print(TTE_true)

    P = rct.sequential_treatment_probs(beta, p)
    Z = rct.staggered_Bernoulli(n, P, r)
    Y = outcomes(Z)

    #print(np.sum(outcomes(np.ones((1,n,1))) - outcomes(np.zeros((1,n,1))))/n)
    
    TTE_est = estimator.polynomial_estimate(Z, Y, P) 
    bias[g] = np.sum(TTE_est - TTE_true)/r
    variance[g] = np.sum((TTE_est - TTE_true)**2)/r

avg_bias = np.sum(bias)/gr
avg_variance = np.sum(variance)/gr
print("Bernoull Staggered Rollout, over {} graphs and {} RCT repetitions per graph: \nAverage Bias: {} \nVariance: {}".format(gr, r, avg_bias, avg_variance))