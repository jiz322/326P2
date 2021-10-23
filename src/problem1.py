'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances


def linear_kernel(X1, X2, sigma=1):
    
    """
    Compute linear kernel between two set of feature vectors.
    The constant 1 is not appended to the x's.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    
    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel evaluated on column i from X1 and column j from X2.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    K = np.zeros((X1.shape[1],X2.shape[1]))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i][j] = X1.T[i].dot(X2.T[j])
    return K
    #########################################



def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute Gaussian kernel between two set of feature vectors.
    
    The constant 1 is not appended to the x's.
    
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)

    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel evaluated on column i from X1 and column j from X2

    """
    #########################################
    ## INSERT YOUR CODE HERE
    K = np.zeros((X1.shape[1],X2.shape[1]))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            #K[i][j] = np.exp(-euclidean_distances([X1.T[i]], [X2.T[j]])**2/sigma**2) 
            #Normalized using number of features???
            #K[i][j] = np.exp((-euclidean_distances([X1.T[i]], [X2.T[j]])**2/sigma**2)/X1.shape[0]) 
            K[i][j] = np.exp((-euclidean_distances([X1.T[i]], [X2.T[j]])**2/sigma**2)/2) #see example 3.11
    return K
    #########################################


def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may be calculated using a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    ## INSERT YOUR CODE HERE
    l = np.zeros((1, len(z)))
    for i in range(l.shape[1]):
        l[0][i] = np.maximum(0, 1-z[i]*y[0][i])
    return l
    #########################################
