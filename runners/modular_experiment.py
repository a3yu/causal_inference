import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.graph import SimpleSBM
from experiment import rct
from model import pom
from experiment import estimator

def run_experiment_poly(parameters, update_param, stop_condition, track_param):
    '''
    parameters: dict containing initial experiment parameters
    update_param: lambda function that takes parameters and a control variable to update the parameters
    stop_condition: lambda function that takes the current parameters and determines the end condition
    track_param: string specifying which parameter to track for plotting
    '''
    bias = []
    variance = []
    current_params = parameters.copy()
    tracked_values = [] 

    while not stop_condition(current_params):
        n = current_params['n']
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

def run_experiment_dm(parameters, update_param, stop_condition, track_param):
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
        tracked_values.append(current_params[track_param])  # Collect values for plotting based on track_param
        nc = current_params['nc']
        p_in = current_params['p_in']
        p_out = current_params['p_out']
        beta = current_params['beta']
        p = current_params['p']
        gr = current_params['gr']
        r = current_params['r']
        cf = current_params['cf']

        dm_bias = np.zeros(gr)
        dm_variance = np.zeros(gr)
        for g in range(gr):
            G = SimpleSBM(n, nc, p_in, p_out)
            TTE_true, outcomes = pom.ppom(beta, G, cf)
            P = rct.sequential_treatment_probs(beta, p)
            Z = rct.staggered_Bernoulli(n, P, r)
            Y = outcomes(Z)
            TTE_dm = estimator.dm_estimate(Z, Y)
            dm_bias[g] = np.sum(TTE_dm - TTE_true) / r
            dm_variance[g] = np.sum((TTE_dm - TTE_true)**2)/r

        avg_bias = np.mean(dm_bias)
        avg_variance = np.mean(dm_variance)
        bias.append(avg_bias)
        variance.append(avg_variance)
        
        current_params = update_param(current_params)

    return tracked_values, bias, variance


