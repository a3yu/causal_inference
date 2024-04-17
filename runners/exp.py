import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model.graph import *
from experiment import rct
from model import pom
from experiment.estimators import estimator
import seaborn as sns
import matplotlib.pyplot as plt

n_iters = [50 * i for i in range(1, 8)]

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
bias = np.zeros((7, gr))
variance = np.zeros((7, gr))
for i, n in enumerate(n_iters):
    for g in range(gr):
        G = SimpleSBM(n, nc, p_in, p_out)
        TTE_true, outcomes = pom.ppom(beta, G, cf)
        #print(TTE_true)

        P = rct.sequential_treatment_probs(beta, p)
        Z = rct.staggered_Bernoulli(n, P, r)
        Y = outcomes(Z)
        
        TTE_est = estimator.polynomial_estimate(Z, Y, P) 
        bias[i][g] = np.sum(TTE_est - TTE_true)/r
        variance[i][g] = np.sum((TTE_est - TTE_true)**2)/r

avg_bias = np.sum(bias, axis=1) / gr
avg_variance = np.sum(variance, axis=1) / gr

sns.lineplot(x=n_iters, y=avg_bias)
plt.fill_between(n_iters, avg_bias - avg_variance, avg_bias + avg_variance, color='blue', alpha=0.3)
plt.xlabel('n')
plt.ylabel('Relative Bias')
plt.show()
