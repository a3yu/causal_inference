import numpy as np
from model.graphs.graphs import SBM, balanced_partition_pin_pout
from experiment.rcts import rct
from model.poms import pom
from experiment.estimators import estimator

def main():
    n = 1000          # population size
    beta = 1        # potential outcomes model degree
    p = 0.5         # treatment budget (marginal treatment probability     

    nc = 50         # number of clusters
    avg_deg = 10     # average graph degree
    p_in = 0.5      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
    p_out = (avg_deg - p_in*(n/nc))/(n - (n/nc))        # probability of edge between units of different clusters
    cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

    gr = 1         # number of graph repetitions
    r = 200         # number of RCT repetitions

    partitions, probs = balanced_partition_pin_pout(n, nc, p_in, p_out)
    for g in range(gr):
        G = SBM(n, partitions, probs)
        TTE_true, fy = pom.ppom(beta, G, cf)
        print(TTE_true)
        #selected = rct.select_clusters_bernoulli(nc, p/q)

        P = rct.sequential_treatment_probs(beta, p)
        Z = rct.staggered_Bernoulli(n, P, r)
        #print(Z)
        Y = fy(Z)
        #print(Y)
        print(np.sum(fy(np.ones((1,n,1))) - fy(np.zeros((1,n,1))))/n)
        
        TTE_est = estimator.polynomial_estimate(Z, Y, P) 
        #print(TTE_est) 
        avg_rel_bias = np.sum((TTE_est - TTE_true)/TTE_true)/r
        print("Under graph {}, the average relative bias of the estimator is {}".format(g, avg_rel_bias))

if __name__ == '__main__':
    main()