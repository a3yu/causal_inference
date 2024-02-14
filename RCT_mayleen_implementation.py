import numpy as np

def select_clusters_bernoulli(numOfClusters, q):
    '''Chooses clusters according to simple Bernoulli(q) randomized design
    
    Parameters
    ------------
    numOfClusters : int
        number of clusters
    q : float
        fraction of clusters you wish to select (in expectation)

    Returns
    --------
    selected : numpy array
        array of the labels of the randomly selected clusters
    '''

    design = (np.random.rand(numOfClusters) < q) + 0
    selected = np.where(design == 1)[0]
    return selected

def select_clusters_complete(numOfClusters, K):
    '''Selects clusters according to complete(K/numOfClusters) RD
    
    Parameters
    ------------
    numOfClusters : int
        number of clusters
    K : int
        number of clusters you wish to select

    Returns
    --------
    selected : numpy array
        array of the labels of the randomly selected clusters

    '''
    design = np.zeros(shape=(numOfClusters))
    design[0:K] = np.ones(shape=(K))
    rng = np.random.default_rng()
    rng.shuffle(design)
    selected = np.where(design==1)[0]
    
    return selected

def zU_to_z(z_U, U, z_U_prime, Uprime, n):
    '''
    Let U be the set of individuals whose clusters were chosen to be randomized to treatment or control.
    Let Uprime be the set of in-neighbors of nodes in U who are not themselves in U (i.e. the boundary of U)
    This function takes the treatment assignment vector of U and Uprime and returns
    the treatment assignment vector for the whole population of size n.

    Parameters
    -----------
    z_U : array
        treatment assignment vector for nodes in U
    U : list
        list of the nodes in U
    z_U_prime : array
        treatment assignment vector for nodes in Uprime
    U : list
        list of the nodes in Uprime   
    n : int
        size of the popluation
    '''
    # Get the indices from [n_U] and [n_{Uprime}] of treated units
    treated_U = np.nonzero(z_U)[0]
    treated_U_prime = np.nonzero(z_U_prime)[0]

    # Get their corresponded indices in [N]
    treated_U = list(map(U.__getitem__, treated_U))
    treated_U_prime = list(map(Uprime.__getitem__, treated_U_prime))
    treated = treated_U + treated_U_prime
    
    # Create the treatment assignment vector of the whole population
    z = np.zeros(n)
    np.put(z,treated,1)

    return z

def staggered_rollout_bern(n, P):
  '''
  Returns Treatment Samples from Bernoulli Staggered Rollout

  beta (int): degree of potential outcomes model
  n (int): size of population
  P (numpy array): treatment probabilities for each time step
  '''

  ### Initialize ###
  Z = np.zeros(shape=(P.size,n))   # for each treatment sample z_t
  U = np.random.rand(n)

  ### staggered rollout experiment ###
  for t in range(P.size):
    ## sample treatment vector ##
    Z[t,:] = (U < P[t])+0

  return Z

def staggered_rollout_bern_clusters(n, selected, P, bndry, P_prime):
  '''
  Returns Treatment Samples from Bernoulli Staggered Rollout with clustering

  n (int): size of population
  selected (list): list of the nodes who were selected to be in the staggered rollout experiment
  P (numpy array): treatment probabilities for each time step for the selected group
  bndry (list): boundary of selected (neighbors of nodes in selected who are not themselves selected)
  P_prime (numpy array): treatment probabilities for each time step for the boundary group
  '''

  ### Initialize ###
  T = len(P)
  Z = np.zeros(shape=(T,n))   # for each treatment sample z_t
  W = np.random.rand(len(selected))
  W_prime = np.random.rand(len(bndry))

  ### staggered rollout experiment ###
  for t in range(T):
    ## sample treatment vector ##
    z_U = (W < P[t])+0
    z_U_prime = (W_prime < P_prime[t])+0
    Z[t,:] = zU_to_z(z_U, selected, z_U_prime, bndry, n)

  return Z

def seq_treatment_probs(beta, p):
  '''
  Returns list of treatment probabilities for Bernoulli staggered rollout

  beta (int): degree of the polynomial; order of interactions of the potential outcomes model
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05

  returns: P such that P[t] is the treatment probability for the t-th time step of the rollout; equally spaced over [0,p]
  '''
  fun = lambda i: (i)*p/(beta)
  P = np.fromfunction(fun, shape=(beta+1,))
  return P
