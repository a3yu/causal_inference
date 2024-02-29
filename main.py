import random
import numpy as np
from model.model import Model
from model.graphs.sbm import SBM
from experiment.rcts import rct
def main():
    # G = sbm(...)
    # Y = pom(G, coeffs, beta)
    # TTE_gt = np.sum(Y(np.ones(n)) - Y(np.zeros(n))) / n
    # Z = staggered_bern(n, P, r)
    # TTE_est = poly_interp(Z, Y, P)
    P = np.array([0.5, 0.7])
    Z = rct.staggered_Bernoulli(5, P, 3)
    print(Z)
    

if __name__ == '__main__':
    main()