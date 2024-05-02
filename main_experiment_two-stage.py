import numpy as np
from model.graphs import SBM, balanced_partition_pin_pout
from experiment import rct
from model import pom
from experiment.estimators import estimator

### Network parameters
n = 1000        # population size
nc = 50         # number of clusters
p_in = 0.5      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
p_out = 0       # probability of edge between units of different clusters
partitions, probs = balanced_partition_pin_pout(n, nc, p_in, p_out)

### Other parameters
beta = 1        # potential outcomes model degree
p = 0.5         # treatment budget (marginal treatment probability) 
q = p           # treatment prob within clusters (conditional treatment probability)
gr = 10         # number of graph repetitions
r = 100         # number of RCT repetitions

cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

### Experiment
bias = np.zeros(gr)
variance = np.zeros(gr)
for g in range(gr):
    G = SBM(n, partitions, probs)
    TTE_true, fy = pom.ppom(beta, G, cf)
    
    selected_clusters = rct.select_clusters_bernoulli(nc, p/q)
    selected = [i for j in selected_clusters for i in partitions[j]]

    Q = rct.sequential_treatment_probs(beta, q)
    Z = rct.clustered_staggered_Bernoulli(n, Q, r, selected)
    Y = fy(Z)
    
    TTE_est = estimator.clustered_polynomial_estimate(Z, Y, Q, p)
    bias[g] = np.sum(TTE_est - TTE_true)/r
    variance[g] = np.sum((TTE_est - TTE_true)**2)/r

avg_bias = np.sum(bias)/gr
avg_variance = np.sum(variance)/gr
print("Cluster Staggered Rollout, over {} graphs and {} RCT repetitions per graph: \nAverage Bias: {} \nVariance: {}".format(gr, r, avg_bias, avg_variance))