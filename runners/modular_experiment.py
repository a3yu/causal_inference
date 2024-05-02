import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.graph import SimpleSBM
from experiment import rct
from model import pom
from experiment.estimators import estimator

def run_experiment(parameters, update_param, stop_condition):
    '''
    parameters: dict containing initial experiment parameters
    update_param: lambda function that takes parameters and a control variable to update the parameters
    stop_condition: lambda function that takes the current parameters and determines the end condition
    '''
    bias = []
    variance = []
    current_params = parameters.copy()

    while not stop_condition(current_params):
        n = current_params['n']
        print(n)
        nc = current_params['nc']
        p_in = current_params['p_in']
        p_out = current_params['p_out']
        beta = current_params['beta']
        p = current_params['p']
        gr = current_params['gr']
        r = current_params['r']
        cf = current_params['cf']

        bias_temp = np.zeros(gr)
        variance_temp = np.zeros(gr)

        for g in range(gr):
            G = SimpleSBM(n, nc, p_in, p_out)
            TTE_true, outcomes = pom.ppom(beta, G, cf)

            P = rct.sequential_treatment_probs(beta, p)
            Z = rct.staggered_Bernoulli(n, P, r)
            Y = outcomes(Z)
            
            TTE_est = estimator.polynomial_estimate(Z, Y, P) 
            bias_temp[g] = np.sum(TTE_est - TTE_true)/r
            variance_temp[g] = np.sum((TTE_est - TTE_true)**2)/r

        # Compute averages
        avg_bias = np.mean(bias_temp)
        avg_variance = np.mean(variance_temp)

        bias.append(avg_bias)
        variance.append(avg_variance)

        # Update parameters for next iteration
        current_params = update_param(current_params)

    return bias, variance

# Parameters dictionary
params = {
    'n': 500,
    'nc': 50,
    'p_in': 0.5,
    'p_out': 0,
    'beta': 2,
    'p': 0.5,
    'gr': 10,
    'r': 100,
    'cf': lambda i, S, G: pom.uniform_coeffs(i, S, G)
}

# Update function
update_param = lambda params: {**params, 'n': params['n'] + 50}

# Stop condition based on parameters
stop_condition = lambda params: params['n'] > 1000  # Stop when number of units exceeds 1000

# Run the experiment
bias, variance = run_experiment(params, update_param, stop_condition)

print(bias, variance)
