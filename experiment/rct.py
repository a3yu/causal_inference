'''
Randomised Control Trials
'''
import numpy as np

def bernoulliRD(n, p):
    '''
    Bernoulli(p) randomized control trial on n units

    n: number of units
    p: probability of selection

    returns binary treatment assignment vector (numpy array with n elements)
    '''
    design = (np.random.rand(n) < p) + 0
    return design


def completeRD(n, k):
    '''
    Complete(n,k) randomized control trial

    n: number of units
    k: number of units to select

    returns binary treatment assignment vector (numpy array with n elements)
    '''
    design = np.zeros(shape=(n))
    design[:k] = np.ones(shape=(k))
    rng = np.random.default_rng()
    rng.shuffle(design)
    return design

def select_clusters_bernoulli(n, p):
    '''
    Bernoulli(p) randomized control trial on n units

    n: number of units
    p: probability of selection

    returns the labels/indices of units selected according to the Bernoulli(p) RCT
    '''
    design = bernoulliRD(n,p)
    return np.where(design == 1)[0]

def select_clusters_complete(n, k):
    '''
    Complete(n,k) randomized control trial

    n: number of units
    k: number of units to select

    returns the labels/indices of units selected according to the Bernoulli(p) RCT
    '''
    design = completeRD(n, k)
    return np.where(design == 1)[0]

def sequential_treatment_probs(beta, p):
    '''
    beta : degree of the potential outcomes model
    p : treatment budget / marginal treatment probability
    '''
    P = [(i)*p/(beta) for i in range(beta+1)]
    return np.array(P)


def staggered_Bernoulli(n, P, r):
    '''
    Returns a Txnxr tensor specifying the treatment selections of staggered rollout Bernoulli trials.
    n: size of population
    P (1d np array): probabilities for each rollout timestep
    r: number of trials (repetitions)
    '''
    # T = P.size
    # U = np.random.rand(T, n, r) # initialise random tensor
    # P_broad = P.reshape(T, 1, 1) # broadcast P
    # design = (U < P_broad).astype(int) # compare each rxT slice with the p's given in U
    # design = np.maximum.accumulate(design, axis=0) # ensures individuals stay treated along T axis

    T = P.size
    U = np.random.rand(n,r) # one uniform value used for all time steps
    design = np.greater.outer(P,U) + 0

    return design


def clustered_staggered_Bernoulli(n, P, r, selected):
    design = staggered_Bernoulli(n, P, r)
    # masking
    mask = np.zeros_like(design, dtype=bool)
    mask[:, selected, :] = True # for the individuals dim, set each i in selected to 1
    return np.where(mask, design, 0)
