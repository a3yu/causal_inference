import numpy as np

def bernoulliRD(n, p):
    '''
    n: size of graph
    p: probability of receiving treatment (E[# of clusters])
    '''
    design = (np.random.rand(n) < p) + 0
    return np.where(design == 1)[0]


def CRD(clusters, k):
    '''
    clusters: number of clusters
    k: number of clusters to select
    '''
    design = np.zeros(shape=(clusters))
    design[:k] = np.ones(shape=(k))
    rng = np.random.default_rng()
    rng.shuffle(design)
    return np.where(design==1)[0]


def staggered_Bernoulli(n, P, r):
    '''
    Returns a Txnxr tensor specifying the treatment selections of staggered rollout Bernoulli trials.
    n: size of population
    P (1d np array): probabilities for each rollout timestep
    r: number of trials (repetitions)
    '''
    T = P.size
    U = np.random.rand(T, n, r) # initialise random tensor
    P_broad = P.reshape(T, 1, 1) # broadcast P
    design = (U < P_broad).astype(int) # compare each rxT slice with the p's given in U
    
    return design


def clustered_staggered_Bernoulli(n, P, r, selected):
    design = staggered_Bernoulli(n, P, r)
    # masking
    mask = np.zeros_like(design, dtype=bool)
    mask[:, selected, :] = True # for the individuals dim, set each i in selected to 1
    return np.where(mask, design, 0)