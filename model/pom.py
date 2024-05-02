'''
Potential outcomes models
'''
import numpy as np
from itertools import chain, combinations

def ppom(beta, G, cf):
    '''
    Polynomial POM
    Returns TTE and the POM function: lambda Z: Y
    '''
    n = len(G)
    C = coeffs(beta, G, cf)
    TTE = sum([sum(C[i]) - C[i][0] for i in range(n)])/n
    return TTE, lambda Z: inner_matt(Z, G, C, beta)

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
        
    

def inner_benson(Z, G, C, beta):
    '''
    Z: Txnxr RCT design tensor
    G: Graph
    C (list[np array]): coefficients, n x m
    Returns: Y: nxTxr POM tensor
    '''
    T, n, r = Z.shape
    Y = np.zeros((n, T, r))
    
    for i in range(n):
        subsets = _subsets(G[i], beta)
        for t in range(T):
            for j in range(r):
                Y[i, t, j] = sum(C[i][subsets.index(S)] * np.prod(Z[t, [k for k in S], j]) for S in subsets)
    
    return Y

def _subsets(S, beta):
    '''
    Returns the subsets of S up to size beta.
    '''
    return list(chain.from_iterable(combinations(S, k) for k in range(beta+1)))

def inner_matt(Z, G, C, beta):
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
            
def inner_aedin(Z, G, C, beta):
    T, n, r = Z.shape
    # Matt's edit to make inputs consistent
    m = max([len(Ci) for Ci in C])
    for Ci in C:
        Ci += [0]*(m-len(Ci))

    # initialise the subsets for each person
    subsets = [_subsets(G[i], beta) for i in range(n)]
    m = max([len(C[i]) for i in range(n)]) #NOTE: len(C[0]) ?
    # R x N matrix where whole[r][n] = the index that person n gets treated in the rth repetition
    
    whole = np.empty((r, n))
    for y in range(r):
        people = np.empty(n)
        for x in range(n):
            slice_z = Z[:, x, y]
            idx = np.where(slice_z == 1)[0]
            people[x] = idx[0] if idx.size != 0 else -1
        whole[y] = people
    
    A = np.empty((n, m, T, r))
    for i in range(n):
        for j in range(m):
            for k in range(r):
                if len(subsets[i])>j:
                    if len(subsets[i][j])>0:
                        my_array = np.zeros(T)
                        maxim = np.max(whole[k, subsets[i][j]])
                        if(maxim != -1):
                            my_array[int(maxim):] = 1
                            A[i, j, :, k] = my_array
                        else:
                            A[i, j, :, k] = my_array
                    else:
                        my_array = np.ones(T)
                        A[i, j, :, k] = my_array
                else: 
                    my_array = np.ones(T)
                    A[i, j, :, k] = my_array  
    Y = np.empty((n, T, r))
    Y = np.einsum('nmtr,nm->ntr', A, C)
    return Y    
