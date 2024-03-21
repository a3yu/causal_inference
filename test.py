from itertools import chain
from itertools import combinations
import experiment.rcts.rct as rct
from model.graphs.graphs import SBM
import numpy as np

def powerset(iterable, beta):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(beta+1)))
    

def innerIndices(Z, G, C, beta):
    '''
    Z: Txnxr RCT design tensor
    G: Graph
    C (list[np array]): coefficients, n x m
    '''
    time = Z.shape[0]
    subsets = []
    for i in range(len(G)):
        neighbours = G[i]
        subsets.append(powerset(neighbours, beta))
    
    whole = []  # Initialize as list to store 1D arrays
    for y in range(Z.shape[2]):
        people = np.array([], dtype=int)
        for x in range(Z.shape[1]):
            slice_z = Z[:, x, y]
            idx = np.where(slice_z == 1)[0]
            if idx.size == 0:
                people = np.append(people, -1)
            else:
                people = np.append(people, idx[0])
        whole.append(people)  # Append the 1D array to the list
    # R x N matrix where [r][n] is the index that person n gets treated in the rth reptition
    whole = np.array(whole)  # Convert the list of 1D arrays into a 2D array
    print(whole)
    rep = []
    big = []
    for j in range(len(whole)):
        for i in range(len(subsets)):
            person = []  # Initialize person as a list for each subset
            for k in range(len(C[0])):
                if len(subsets[i]) > k:
                    if len(subsets[i][k]) != 0:
                        if(np.min(whole[j, subsets[i][k]])== -1):
                            person.append(np.zeros(time))
                        else:
                            my_array = np.zeros(time)
                            my_array[np.max(whole[j, subsets[i][k]]):] = 1
                            person.append(my_array)  
                    else:
                        person.append(np.zeros(time)) 
                else:
                   person.append(np.zeros(time)) 
            big.append(person)  # Append person list to big
        rep.append(big)
        # construct A (nxmxTxr) from G:
        #  rxnxmxt
    rep = np.transpose(rep, (1, 2, 3, 0))
    Y = np.empty((Z.shape[1],Z.shape[0],Z.shape[2]))
    for i in range(Z.shape[1]):
        Y[i] = np.einsum('i,ijk->jk', C[i], rep[i])
    print(Y)
    return Y
    
    
    
        
def main():
    G = SBM(3, [[1,2], [3]], [[0.5, 0.5], [0.5, 0.5]])
    P = np.array([0.5, 0.7])
    Z = rct.staggered_Bernoulli(3, P, 5)
    A = np.array([[[1, 0, 1, 0],[0, 1, 0, 0],[1, 1, 0, 0]],[[1, 1, 1, 1],[1, 1, 0, 0],[1, 1, 1, 0]]])
    innerIndices(Z, [[0,1,2],[0,1],[2]],[[0,1,2,3,4,0],[0,1,2,0,0,0],[0,1,0,0,0,0]],1 )
    
    
if __name__ == '__main__':
    main()