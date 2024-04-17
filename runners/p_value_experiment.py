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

# Simulation parameters
p_values = [0.1, 0.3, 0.5, 0.7, 1]
n = 500
nc = 50          # Number of clusters
p_in = 0.5       # Probability of edge within same cluster
p_out = 0        # Probability of edge between different clusters
beta = 2         # Degree of potential outcomes model
gr = 10          # Number of graph repetitions
r = 100          # Number of RCT repetitions
cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)  # Function to generate coefficients

# Prepare dataframe to store all results
results = pd.DataFrame()

# Main simulation loop
for p in p_values:
    for g in range(gr):
        G = SimpleSBM(n, nc, p_in, p_out)
        TTE_true, outcomes = pom.ppom(beta, G, cf)

        P = rct.sequential_treatment_probs(beta, p)
        Z = rct.staggered_Bernoulli(n, P, r)
        Y = outcomes(Z)

        TTE_est = estimator.polynomial_estimate(Z, Y, P)
        bias = np.mean(TTE_est - TTE_true)
        variance = np.var(TTE_est - TTE_true)
        results = results._append({'p_value': p, 'bias': bias, 'variance': variance}, ignore_index=True)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=results, x='p_value', y='bias', ci='sd', estimator='mean', markers=True, dashes=False)
plt.title('Average Bias vs. P with Shaded Confidence Interval')
plt.xlabel('P values')
plt.ylabel('Average Bias')
plt.show()
