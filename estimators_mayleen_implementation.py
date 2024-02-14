import numpy as np
import scipy.sparse

def PI(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    if n > 0:
        return (1/n)*H.dot(sums)
    else:
        return 0
    
def outcome_sums(n, Y, Z, selected):
    '''
    Returns the sums of the outcomes Y(z_t) for each timestep t

    Y (function): potential outcomes model
    Z (numpy array): treatment vectors z_t for each timestep t
    - each row should correspond to a timestep, i.e. Z should be beta+1 by n
    selected (list): indices of units in the population selected to be part of the experiment (i.e in U)
    '''
    if len(selected) == n: # if we selected all nodes, sums = sums_U
        sums = np.zeros(Z.shape[0])
        for t in range(Z.shape[0]):
            outcomes = Y(Z[t,:])
            sums[t] = np.sum(outcomes)
        return sums, sums
    else: 
        sums, sums_U = np.zeros(Z.shape[0]), np.zeros(Z.shape[0]) 
        for t in range(Z.shape[0]):
            outcomes = Y(Z[t,:])
            sums[t] = np.sum(outcomes)
            sums_U[t] = np.sum(outcomes[selected])
    return sums, sums_U

def poly_LS_prop(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

def poly_LS_num(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  count = 1
  treated_neighb = np.array(A.sum(axis=1)).flatten()-1
  for i in range(beta):
      X[:,count] = np.power(treated_neighb,i)
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  return TTE_hat

def DM_naive(y, z):
    '''
    Returns an estimate of the TTE using difference in means
    (mean outcome of individuals in treatment) - (mean outcome of individuals in control)

    y (numpy array): observed outcomes
    z (numpy array): treatment vector
    '''
    treated = np.sum(z)
    untreated = np.sum(1-z)
    est = 0
    if treated > 0:
        est = est + y.dot(z)/treated
    if untreated > 0:
        est = est - y.dot(1-z)/untreated
    return est

def DM_fraction(n, y, A, z, tol):
    '''
    Returns an estimate of the TTE using weighted difference in means where 
    we only count neighborhoods with at least tol fraction of the neighborhood being
    assigned to treatment or control

    n (int): number of individuals
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    tol (float): neighborhood fraction treatment/control "threshhold"
    '''
    z = np.reshape(z,(n,1))
    treated = 1*(A.dot(z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    treated = np.multiply(treated,z).flatten()
    control = 1*(A.dot(1-z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    control = np.multiply(control,1-z).flatten()

    est = 0
    if np.sum(treated) > 0:
        est = est + y.dot(treated)/np.sum(treated)
    if np.sum(control) > 0:
        est = est - y.dot(control)/np.sum(control)
    return est

#######################################
# Estimators - Horvitz-Thomson & Hajek
#######################################

def horvitz_thompson_old(n, nc, y, A, z, q, p):
    '''Computes the Horvitz-Thompson estimate of the TTE under Bernoulli design or Cluster-Bernoulli design.
    
    Parameters
    ----------
    n : int
        the size of the population/network
    nc : int
        the number of clusters (equals n if simple Bernoulli design with no clustering)
    y : numpy array
        the outcomes of each unit in the population
    A : scipy sparse array
        adjacency matrix of the network such that A[i,j]=1 indicates that unit j is an in-neighbor of i
    z : numpy array
        the treatment assignment of each unit in the population
    q : float
        probability that a cluster is indepdently chosen for treatment (should equal 1 under simple Bernoulli design with no clustering)
    p : float
        the treatment probability for chosen clusters in the staggered rollout
    '''
    neighborhoods = [list(row.nonzero()[1]) for row in A] # list of neighbors of each unit
    neighborhood_sizes = A.sum(axis=1).tolist() # size of each unit's neighborhood
    neighbor_treatments = [list(z[neighborhood]) for neighborhood in neighborhoods] # list of treatment assignments in each neighborhood

    A = A.multiply(scipy.sparse.csr_array(np.tile(np.repeat(np.arange(1,nc+1),n//nc), (n,1)))) # modifies the adjancecy matrix so that if there's an edge from j to i, A[i,j]=cluster(j)
    cluster_neighborhoods = [np.unique(row.data,return_counts=True) for row in A] # for each i, cluster_neighborhoods[i] = [a list of clusters i's neighbors belong to, a list of how many neighbors are in each of these clusters]
    cluster_neighborhood_sizes = [len(x[0]) for x in cluster_neighborhoods] # size of each unit's cluster neighborhood
    
    # Probabilities of each person's neighborhood being entirely treated or entirely untreated
    all_treated_prob = np.multiply(np.power(p, neighborhood_sizes), np.power(q, cluster_neighborhood_sizes))
    none_treated_prob = [np.prod((1-q) + np.power(1-p, x[1])*q) for x in cluster_neighborhoods]
    
    # Indicators of each person's neighborhood being entirely treated or entirely untreated
    all_treated = [np.prod(treatments) for treatments in neighbor_treatments]
    none_treated = [all(z == 0 for z in treatments)+0 for treatments in neighbor_treatments]

    zz = np.nan_to_num(np.divide(all_treated,all_treated_prob) - np.divide(none_treated,none_treated_prob))

    return 1/n * y.dot(zz)

def horvitz_thompson(n, nc, y, A, z, q, p):    
    AA = A.toarray()

    cluster = []
    for i in range(1,nc+1):
        cluster.extend([i]*(n//nc))

    cluster_neighborhoods = np.apply_along_axis(lambda x: np.bincount(x*cluster, minlength=nc+1), axis=1, arr=AA)[:,1:]
    
    degree = np.sum(cluster_neighborhoods, axis=1)
    cluster_degree = np.count_nonzero(cluster_neighborhoods, axis=1)

    # Probabilities of each person's neighborhood being entirely treated or entirely untreated
    all_treated_prob = np.power(p, degree) * np.power(q, cluster_degree)
    none_treated_prob = np.prod(np.where(cluster_neighborhoods>0,(1-q)+np.power(1-p,cluster_neighborhoods)*q,1),axis=1)

    # Indicators of each person's neighborhood being entirely treated or entirely untreated
    all_treated = np.prod(np.where(AA>0,z,1),axis=1)
    none_treated = np.prod(np.where(AA>0,1-z,1),axis=1)

    zz = np.nan_to_num(np.divide(all_treated,all_treated_prob) - np.divide(none_treated,none_treated_prob))

    return 1/n * y.dot(zz)
"""
def hajek(n, p, y, A, z, clusters=np.array([])): 
    '''
    TODO
    '''
    if clusters.size == 0:
        zz_T = np.prod(np.tile(z/p,(n,1)), axis=1, where=A==1)
        zz_C = np.prod(np.tile((1-z)/(1-p),(n,1)), axis=1, where=A==1)
    else:
        deg = np.sum(clusters,axis=1)
        wt_T = np.power(p,deg)
        wt_C = np.power(1-p,deg)
        zz_T = np.multiply(np.prod(A*z,axis=1),wt_T) 
        zz_C = np.multiply(np.prod(A*(1-z),axis=1),wt_C)
    all_ones = np.ones(n)
    est_T = 0
    est_C=0
    if all_ones.dot(zz_T) > 0:
        est_T = y.dot(zz_T) / all_ones.dot(zz_T)
    if all_ones.dot(zz_C) > 0:
        est_C = y.dot(zz_C) / all_ones.dot(zz_C)
    return est_T - est_C
"""