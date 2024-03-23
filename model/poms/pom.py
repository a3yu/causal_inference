import numpy as np
from itertools import chain,combinations

def inner_benson(Z, G, C, beta):
    '''
    Z: Txnxr RCT design tensor
    G: Graph
    C (list[np array]): coefficients, n x m
    '''

    # construct A (nxmxTxr) from G:
    T,n,r = Z.shape

    # Matt's edit to make inputs consistent
    m = max([len(Ci) for Ci in C])
    for Ci in C:
        Ci += [0]*(m-len(Ci))


    m = len(C[0])
    A = np.empty((n, m, T, r))
    # A[i][j] = prod_{k in S_j} of z_k
    for i in range(n):
        # S = [0 for _ in range(m)] 
        S = np.empty((m, T, r)) # mxTxr slice
        neighbours = G[i]
        subset_idx = 0
        for size in range(beta + 1): # generate subsets
            for subset in combinations(neighbours, size):
                # for each Txr slice, check for z[t][k][r] = 0
                for t in range(T):
                    for rep in range(r):
                        prod = 0 if any(Z[t][k][rep] for k in subset) else 1
                        S[subset_idx][t][rep] = prod
                subset_idx += 1
            
        A[i] = S
    
    Y = np.empty((n,T,r))
    for i in range(n):
        Y[i] = np.einsum('i,ijk->jk', C[i], A[i])
    return Y

def _subsets(S, beta):
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
    '''
    Z: Txnxr RCT design tensor
    G: Graph: adjacency list of incoming edges for each individual
    C (list[np array]): coefficients, n x m
    '''
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
    A = [] #rxnxmxT
    big = []
    for j in range(r):
        for i in range(n):
            person = []  # Initialize person as a list for each subset
            for k in range(m):
                if len(subsets[i]) <= k or len(subsets[i][k]) == 0: #NOTE what to do with emptyset?
                    person.append(np.zeros(T))
                elif np.min(whole[j, subsets[i][k]]) == -1:
                    person.append(np.zeros(T))
                else:
                    my_array = np.zeros(T)
                    my_array[int(np.max(whole[j, subsets[i][k]])):] = 1
                    person.append(my_array)
            big.append(person)  # Append person list to big
        A.append(big)
    # transpose to nxmxTxr
    A = np.transpose(A, (1, 2, 3, 0))
    print(A.shape)
    Y = np.empty((n, T, r))
    for i in range(n):
        Y[i] = np.einsum('i,ijk->jk', C[i], A[i])
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
