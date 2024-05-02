import numpy as np
from model.graphs.graphs import SBM
from experiment import rct
from model import pom
def main():
    G = SBM(10, [[0,1,2,3,4],[5,6,7,8,9]], [[0.5, 0.5], [0.5, 0.5]])
    # Y = pom(G, coeffs, beta)
    # TTE_gt = np.sum(Y(np.ones(n)) - Y(np.zeros(n))) / n
    # Z = staggered_bern(n, P, r)
    # TTE_est = poly_interp(Z, Y, P)
    P = np.array([0.5, 0.7])
    Z = rct.staggered_Bernoulli(3, P, 5)
    
    a = pom.inner(Z, [[0,1,2],[0,1],[2]],[[0,1,2,3,4,0],[0,1,2,0,0,0],[0,1,0,0,0,0]],1 )
    print(Z)
    print(Z.shape)

if __name__ == '__main__':
    main()