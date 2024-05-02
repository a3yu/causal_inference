import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modular_experiment import run_experiment_poly, run_experiment_dm
from experiment import rct
from model import pom
from masterplot import plot_bias_variance

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
stop_condition = lambda params: params['n'] > 800

# Run the experiment tracking 'n'
tracked_param = 'n'  # This can be changed to any parameter you want to track
dm_tracked_values, dm_bias, dm_variance = run_experiment_dm(params, update_param, stop_condition, tracked_param)
poly_tracked_values, poly_bias, poly_variance = run_experiment_poly(params, update_param, stop_condition, tracked_param)


dm_results = {
    'tracked_values': dm_tracked_values,
    'bias': dm_bias,
    'variance': dm_variance
}
poly_results = {
    'tracked_values': poly_tracked_values,
    'bias': poly_bias,
    'variance': poly_variance
}

tracked_data = {
    'DM Estimator': dm_results,
    'Poly Estimator': poly_results
}

plot_bias_variance(tracked_data)