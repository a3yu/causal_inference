import numpy as np
import scipy.sparse

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





########################################
# Functions to generate network weights
########################################

def simpleWeights(A, diag=5, offdiag=5, rand_diag=np.array([]), rand_offdiag=np.array([])):
    '''
    Returns weights generated from model described in Experiments Section

    A (numpy array): n by n adjacency matrix of the network
    diag (float): maximum norm of direct effects
    offdiag (float): maximum norm of the indirect effects
    rand_diag (numpy array): array of n numbers governing direct effects of each node
    rand_offdiag (numpy arry): array of n numbers governing indirect effects of each node
    '''
    n = A.shape[0]

    if rand_offdiag.size == 0:
        rand_offdiag = np.random.rand(n)
    C_offdiag = offdiag*rand_offdiag

    in_deg = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(),0)  # array of the in-degree of each node
    C = in_deg.dot(A - scipy.sparse.eye(n))
    col_sum = np.array(C.sum(axis=0)).flatten()
    col_sum[col_sum==0] = 1
    temp = scipy.sparse.diags(C_offdiag/col_sum)
    C = C.dot(temp)

    if rand_diag.size == 0:
        rand_diag = np.random.rand(n)
    C_diag = diag*rand_diag
    C.setdiag(C_diag)

    return C

########################################
# Potential Outcomes Models
########################################
linear_pom = lambda C,alpha, z : C.dot(z) + alpha

# Scale the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1      # for quadratic effects
a3 = 1      # for cubic effects
a4 = 1      # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z # should be equivalent to linear_pom
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
  '''
  Returns k-degree polynomial potential outcomes (POM) function
  
  beta (int): degree of POM 
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''
  g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()

  if beta == 0:
      return lambda z: alpha + a1*z
  elif beta == 1:
      f = f_linear
  elif beta == 2:
      f = f_quadratic
  elif beta == 3:
      f = f_cubic
  elif beta == 4:
      f = f_quadratic
  else:
      print("ERROR: invalid degree")

  return lambda z: f(alpha, C.dot(z), g(z)) 