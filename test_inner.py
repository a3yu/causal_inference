from model.poms.pom import *
from experiment.rcts.rct import staggered_Bernoulli

import numpy as np

def main():
    beta = 2
    r = 10
    P = np.array([0,0.25,0.5])

    G = [[0,1,2],[1,3],[2,0,3],[3]]
    n = len(G)

    C = [[1,4,5,-2,3,-6,1],[3,-3,0,4],[5,-2,-2,1,4,0,3],[6,2]]
    
    Z = staggered_Bernoulli(n,P,r)

    YM = inner_matt(Z,G,C,beta)
    print("Matt:")
    print(YM)
    YB = inner_benson(Z,G,C,beta)
    print("Benson:")
    print(YB)
    YA = inner_aedin(Z,G,C,beta)
    print("Aedin:")
    print(YA)
       
if __name__ == '__main__':
    main()
