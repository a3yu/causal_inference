import numpy as np

def polynomial_estimate(Z, Y, P):
    H = berns_coeff(P)
    
    #time_sums = [np.sum(Y[step]) for step in Z]
    #return (1/np.size(Z,1))*H.dot(time_sums)

    return 1/np.size(Z,1) * H @ np.sum(Y,axis=1)    

def berns_coeff(P):
    H = np.zeros(P.size)
    for t in range(P.size):
        one_minusP = 1 - P          
        pt_minusP = P[t] - P
        minusP = -1*P
        one_minusP[t] = 1
        pt_minusP[t] = 1
        minusP[t] = 1
        fraction1 = one_minusP/pt_minusP
        fraction2 = minusP/pt_minusP
        H[t] = np.prod(fraction1) - np.prod(fraction2)
    return H
    
# def __init__(): 
#     berns_coeff(np.array([1,2,3,4,5,6]))
#     berns_coeff_old(np.array([1,2,3,4,5,6]))

def clustered_polynomial_estimate(Z, Y, Q, p):
    H = berns_coeff(Q)

    #time_sums = [np.sum(Y[step]) for step in Z]
    #return (1/np.size(Z,1))*H.dot(time_sums)

    return 1/np.size(Z,1) * Q[-1]/p * H @ np.sum(Y,axis=1)    