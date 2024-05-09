'''
Potential outcomes models to compute the total treatment effect and an outcomes function,
given a graph and coefficients model.
'''
import numpy as np
from itertools import chain, combinations

def ppom(beta, G, cf):
    '''
    Polynomial POM
    beta (int): coefficient parameter
    G: Adjacency list representation of graph (length = n)
    cf: lambda i, S, G: coeff
    Returns TTE and the POM function: lambda Z: Y
    '''
    n = len(G)
    C = coeffs(beta, G, cf)
    TTE = sum([sum(C[i]) - C[i][0] for i in range(n)])/n
    return TTE, lambda Z: inner(Z, G, C, beta)


def coeffs(beta, G, cf):
    '''
    Creates the coefficients on a graph with a user-defined function
    cf: lambda i, S, G: coeff
    Returns: C (list[list[int]]): nxm, C[i] are the coeffs of the ith person
    '''
    n = len(G)
    C = [[] for _ in range(n)]
    for i in range(n):
        neighbours = G[i]
        for subset in _subsets(neighbours, beta):
            coeff = cf(i, subset, G)
            C[i].append(coeff)
    return C


def uniform_coeffs(i, S, G):
    '''
    i : unit/node
    S : subset of i's neighborhood
    G : graph/network 
    '''
    a = 2
    b = 3
    ciS = np.random.uniform(0,a) / (b**len(S))
    return ciS


def coeffs_new(i, S, G, r):
    '''
    i : unit/node
    S : subset of i's neighborhood
    G : graph/network 
    r : magnitude of rel to direct effects.
    Returns coefficient for the subset S of i's neighborhood
    '''
    if(S == [] or S==[i]):
        return np.random.uniform(0,1)
    else:
        return np.random.uniform(0,r)*np.size(G[i])/np.sum(np.size(G[S]))
        

def _subsets(S, beta):
    '''
    Returns the subsets of S up to size beta.
    '''
    return list(chain.from_iterable(combinations(S, k) for k in range(beta+1)))


def inner(Z, G, C, beta):
    '''
    Matt's implementation of the function that maps treatment assignments to potential outcomes
        Z: Tensor of treatment assignments: T x n x r 
        G: Adjacency list representation of graph (length = n)
        C: List of lists of model coefficients (length = n, length of C[i] = mi)
    Returns Y: T x n x r
    '''

    T,n,r = Z.shape

    # Note: This function will work best if the shapes of Z and Y are T x r x n, I'll transpose for now
    Z = np.transpose(Z,(0,2,1))

    Y = np.empty((T,n,r))
    for i in range(n):
        Sis = _subsets(G[i],beta)                    # vector of lists of elements in each subset 
        A = np.empty((T,r,len(Sis)))                 # indicates full treatment of each subset at each time/replication 
        for j,Si in enumerate(Sis):
            A[:,:,j] = np.prod(Z[:,:,Si], axis=2) 
        Y[:,i,:] = A @ C[i] 

    return Y
