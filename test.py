from itertools import chain
from itertools import combinations
import experiment.rcts.rct as rct
from model.graphs.graphs import SBM
import numpy as np

def _powerset(iterable, beta):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(beta+1)))
    

def inner(Z, G, C, beta):
    '''
    Z: Txnxr RCT design tensor
    G: Graph: adjacency list of incoming edges for each individual
    C (list[np array]): coefficients, n x m
    '''
    T, n, r = Z.shape
    # initialise the subsets for each person
    subsets = [_powerset(G[i], beta) for i in range(n)]
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
    
    
def main():
    G = SBM(3, [[1,2], [3]], [[0.5, 0.5], [0.5, 0.5]])
    P = np.array([0.5, 0.7])
    Z = rct.staggered_Bernoulli(3, P, 5)
    A = np.array([[[1, 0, 1, 0],[0, 1, 0, 0],[1, 1, 0, 0]],[[1, 1, 1, 1],[1, 1, 0, 0],[1, 1, 1, 0]]])
    inner(Z, [[0,1,2],[0,1],[2]],[[0,1,2,3,4,0],[0,1,2,0,0,0],[0,1,0,0,0,0]],1 )
    
    
if __name__ == '__main__':
    main()
