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
from experiment import estimator

def run_experiment(parameters, update_param, stop_condition, track_param):
    '''
    parameters: dict containing initial experiment parameters
    update_param: lambda function that takes parameters and a control variable to update the parameters
    stop_condition: lambda function that takes the current parameters and determines the end condition
    track_param: string specifying which parameter to track for plotting
    '''
    bias = []
    variance = []
    current_params = parameters.copy()
    tracked_values = []  # For plotting, collect tracked parameter values

    while not stop_condition(current_params):
        n = current_params['n']
        print(n)
        tracked_values.append(current_params[track_param])  # Collect values for plotting based on track_param
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

    return tracked_values, bias, variance

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
update_param = lambda params: {**params, 'n': params['n'] + 50, 'p': params['n']/1000}

# Stop condition based on parameters
stop_condition = lambda params: params['n'] > 1000  # Stop when number of units exceeds 1000

# Run the experiment tracking 'n'
tracked_param = 'n'  # This can be changed to any parameter you want to track
tracked_values, bias, variance = run_experiment(params, update_param, stop_condition, tracked_param)

# Plotting
plt.figure(figsize=(10, 5))
sns.lineplot(x=tracked_values, y=bias, label='Bias')
plt.fill_between(tracked_values, np.array(bias) - np.array(variance), np.array(bias) + np.array(variance), color='blue', alpha=0.3)
plt.title(f'Bias and Variance Evolution with Varying {tracked_param}')
plt.xlabel(tracked_param)
plt.ylabel('Bias')
plt.legend()
plt.show()
