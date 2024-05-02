import sys
import os
import numpy as np
import time
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.graph import *
from experiment import rct
from model import pom
from experiment import estimator

### Network parameters
n = 1000         # population size
nc = 50          # number of clusters
p_in = 0.5       # probability of edge between units within same cluster
p_out = 0        # probability of edge between units of different clusters

### Other parameters
beta = 2         # potential outcomes model degree
p = 0.5          # treatment budget (marginal treatment probability)
gr = 10          # number of graph repetitions
r = 100          # number of RCT repetitions
cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

def run_experiment(g):
    start_time = time.time()  # Start timing this function

    np.random.seed(g)  # Seed for reproducibility in parallel processing
    G = SimpleSBM(n, nc, p_in, p_out)  # Generate graph
    graph_time = time.time()
    print("Graph generation time: {:.2f} seconds".format(graph_time - start_time))

    TTE_true, outcomes = pom.ppom_parallel(beta, G, cf)
    pom_time = time.time()
    print("POM computation time: {:.2f} seconds".format(pom_time - graph_time))

    P = rct.sequential_treatment_probs(beta, p)
    Z = rct.staggered_Bernoulli(n, P, r)
    Y = outcomes(Z)
    rct_time = time.time()
    print("RCT processing time: {:.2f} seconds".format(rct_time - pom_time))

    TTE_est = estimator.polynomial_estimate(Z, Y, P)
    estimator_time = time.time()
    print("Estimator processing time: {:.2f} seconds".format(estimator_time - rct_time))

    bias = np.sum(TTE_est - TTE_true) / r
    variance = np.sum((TTE_est - TTE_true)**2) / r
    end_time = time.time()
    print("Bias and Variance calculation time: {:.2f} seconds".format(end_time - estimator_time))
    print("Total function time: {:.2f} seconds".format(end_time - start_time))

    return bias, variance

def main():
    start = time.time()  # Start timing the entire process
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, range(gr)))
    
    biases = [result[0] for result in results]
    variances = [result[1] for result in results]
    
    avg_bias = np.mean(biases)
    avg_variance = np.mean(variances)
    
    end = time.time()
    print("Total execution time: {:.2f} seconds".format(end - start))
    print("Bernoulli Staggered Rollout, over {} graphs and {} RCT repetitions per graph: \nAverage Bias: {}\nVariance: {}".format(gr, r, avg_bias, avg_variance))

if __name__ == "__main__":
    main()
