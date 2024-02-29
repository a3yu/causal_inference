import numpy as np
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
