import numpy as np
from itertools import combinations

def inner(z, Z, G, C, beta):
    '''
    Z: Txnxr RCT design tensor
    G: Graph
    C (list[np array]): coefficients, n x m
    '''

    # construct A (nxmxTxr) from G:
    T,n,r = Z.shape
    m = len(C)
    A = np.empty((n, m))
    # A[i][j] = prod_{k in S_j} of z_k
    for i in range(n):
        # S = [0 for _ in range(m)] 
        S = np.empty(m, T, r) # mxTxr slice
        neighbours = G[i]
        subset_idx = 0
        for size in range(beta + 1): # generate subsets
            for subset in combinations(neighbours, size):
                # for each Txr slice, check for z[t][k][r] = 0
                for t in range(T):
                    for rep in range(r):
                        prod = 0 if any(Z[t][k][rep] for k in subset) else 1
                        S[t][subset_idx][r] = prod
                        subset_idx += 1
        A[i] = S
    
    Y = np.empty((n,T,r))
    for i in range(n):
        Y[i] = C[i] @ A[i]
    return Y


"""
Returns a POM function z -> [Y0[z], Y1[z], ..., Yn[z]]
"""
def linear_pom(G, alpha, z):
    '''
    G: weighted graph
    alpha: baseline effects
    z: treatment vector
    '''
    return lambda z: G.dot(z) + alpha


def poly_pom(G, beta, alpha, a1, a2=None, a3=None, a4=None):
    '''
    Returns a polynomial POM with poly coefficients a1, a2, a3, a4 (a2-a4 optional).
    G: weighted graph
    beta: degree
    alpha: baseline effects
    z: tratment vector
    '''
    def f(z, gz):
        result = alpha + a1*z
        if beta >= 2 and a2 is not None:
            result += a2*np.multiply(gz, gz)
        if beta >= 3 and a3 is not None:
            result += a3*np.power(gz, 3)
        if beta >= 4 and a4 is not None:
            result += a4*np.power(gz, 4)
        return result
    
    if beta == 0:
        return lambda z: alpha + a1*z
    elif beta == 1:
        return lambda z: alpha + a1*G.dot(z)
    else:
        g = lambda z: G.dot(z) / np.array(np.sum(G, axis=1)).flatten()
        return lambda z: f(G.dot(z), g(z))
