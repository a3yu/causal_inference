import numpy as np
from model.graphs import SBM, balanced_partition_pin_pout
from experiment import rct
from model import pom
from experiment.estimators import estimator

def main():
    beta = 1        # potential outcomes model degree
    p = 0.5         # treatment budget (marginal treatment probability) 
    q = 1           # treatment prob within clusters (conditional treatment probability)
    
    n = 1000        # population size
    nc = 50         # number of clusters
    avg_deg = 10    # average graph degree
    p_in = 0.4      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
    p_out = (avg_deg - p_in*(n/nc))/(n - (n/nc))        # probability of edge between units of different clusters
    
    cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

    gr = 5         # number of graph repetitions
    r = 100          # number of RCT repetitions

    partitions, probs = balanced_partition_pin_pout(n, nc, p_in, p_out)
    avg_rel_bias = np.zeros(gr)
    for g in range(gr):
        G = SBM(n, partitions, probs)
        TTE_true, fy = pom.ppom(beta, G, cf)
        
        selected_clusters = rct.select_clusters_bernoulli(nc, p/q)
        selected = [i for j in selected_clusters for i in partitions[j]]

        Q = rct.sequential_treatment_probs(beta, q)
        Z = rct.clustered_staggered_Bernoulli(n, Q, r, selected)
        Y = fy(Z)
        
        TTE_est = estimator.clustered_polynomial_estimate(Z, Y, Q, p)
        avg_rel_bias[g] = np.sum((TTE_est - TTE_true)/TTE_true)/r
    
    avg_rel_bias_graphs = np.sum(avg_rel_bias)/gr
    print("The average relative bias of the estimator, across {} graphs and {} RCT repetitions per graph is {}".format(gr, r, avg_rel_bias_graphs))

if __name__ == '__main__':
    main()