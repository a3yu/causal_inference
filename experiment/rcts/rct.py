import numpy as np

class RCT:
    def __init__(self) -> None:
        '''
        Instantiate an instance of a RCT factory.
        '''

    def bernoulliRD(self, n, p):
        '''
        n: size of graph
        p: probability of receiving treatment (E[# of clusters])
        '''
        design = (np.random.rand(n) < p) + 0
        return np.where(design == 1)[0]


    def CRD(self, n, k):
        '''
        n: size of graph
        k: number of clusters to select
        '''
        design = np.zeros(shape=(n))
        design[:k] = np.ones(shape=(k))
        rng = np.random.default_rng()
        rng.shuffle(design)
        return np.where(design==1)[0]


    def staggered_Bernoulli(self, n, P, K):
        '''
        Returns a nxtxk tensor specifying the treatment selections of staggered rollout Bernoulli trials.
        n: size of population
        P (1d np array): probabilities for each rollout timestep
        K: number of trials
        '''
        T = P.size
        design = np.empty((T, n, k))
        for k in range(K): # repetitions of the staggered rollout trial
            Z = np.empty((T, n))
            U = np.random.rand(n)
            for t in range(T): # doing one rollout trial
                Z[t,:] = (U < P[t])+0
            design[k,:,:] = Z
        
        return design