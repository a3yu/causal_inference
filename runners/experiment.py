'''
Basic univariate experiment setting that sets up parameters and runs a simulation.
Results are plotted in seaborn.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model.graph import *
from experiment import rct
from model import pom
from experiment import estimator
import seaborn as sns
import matplotlib.pyplot as plt

def run_experiment(parameter_type, varying_list):
    '''
    Performs a univariate experiment based on the input parameter. 
    Computes and returns the average bias and variance.
    parameter_type: 'p', 'n', 'beta'
    varying_list: contains the values of the parameter to vary by at each iteration.
    '''
    ### Setup the experiment values
    n = 500
    nc = 50         # number of clusters
    p_in = 0.5      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
    p_out = 0       # probability of edge between units of different clusters

    ### Other parameters
    beta = 2        # potential outcomes model degree
    p = 0.5         # treatment budget (marginal treatment probability)     
    gr = 10          # number of graph repetitions
    r = 100       # number of RCT repetitions
    cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients
    
    bias = np.zeros((len(varying_list), gr))
    variance = np.zeros((len(varying_list), gr))

    def run_iteration():
        '''
        Performs one iteration of the experiment based on the current parameter values.
        '''
        for g in range(gr):
            G = SimpleSBM(n, nc, p_in, p_out)
            TTE_true, outcomes = pom.ppom(beta, G, cf)

            P = rct.sequential_treatment_probs(beta, p)
            Z = rct.staggered_Bernoulli(n, P, r)
            Y = outcomes(Z)
            
            TTE_est = estimator.polynomial_estimate(Z, Y, P) 
            bias[i][g] = np.sum(TTE_est - TTE_true)/r
            variance[i][g] = np.sum((TTE_est - TTE_true)**2)/r
 
    # Iterate through each value in the list and run the iteration
    for i, x in enumerate(varying_list):
        if parameter_type == 'p':
            p = x
        elif parameter_type == 'beta':
            beta = x
        elif parameter_type == 'n':
            n = x
        run_iteration()
        
    avg_bias = np.sum(bias, axis=1) / gr
    avg_variance = np.sum(variance, axis=1) / gr

    return avg_bias, avg_variance


## SAMPLE USAGE
# Vary parameter n
n_iters = [50 * i for i in range(1, 8)]
avg_bias, avg_variance = run_experiment('n', n_iters)

# Vary parameter beta
beta_iters = [1, 2, 3]
beta_bias, beta_var = run_experiment('beta', beta_iters)

# Vary parameter p
p_iters = [0.1, 0.3, 0.5, 0.7, 1]
p_bias, p_var = run_experiment('p', p_iters)

## PLOT
sns.set_theme(style="darkgrid")
sns.lineplot(x=n_iters, y=avg_bias)
plt.fill_between(n_iters, avg_bias - avg_variance, avg_bias + avg_variance, color='blue', alpha=0.3)
plt.xlabel('n')
plt.ylabel('Relative Bias')
plt.show()
