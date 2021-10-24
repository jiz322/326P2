# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to the objective function of Eq. (47).
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    #implement Eqn. 52
    m = alpha.shape[1]
    #Compute the sum of alpha value
    sum_alpha = 0
    for i in range(m):
        sum_alpha = sum_alpha + alpha[0][i]
    #Compute second term
    # L_alpha = sum_alpha - 0.5* sencond_term
    sencond_term = 0
    for i in range(m):
        for j in range(m):
            xi = np.array([train_X.T[i]]).T
            xj = np.array([train_X.T[j]]).T
            if kernel_function.__name__ == "linear_kernel":
                kxx = kernel_function(xi, xj)
            else:
                kxx = kernel_function(xi, xj, sigma)
            sencond_term = sencond_term + alpha[0][i]*alpha[0][j]*train_y[0][i]*train_y[0][j]*kxx
    
    return sum_alpha - 0.5 * sencond_term
    ## INSERT YOUR CODE HERE
    #########################################


# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: you need to use kernel trick when come to Gaussian kernel. Refer to the derivation of the dual objective function Eq. (47) to check how to find
            1/2 ||w||^2 and the decision_function with kernel trick.
    """
    #########################################
    m = alpha.shape[1]
    n = train_X.shape[0]
    if kernel_function.__name__ == "linear_kernel":
        #compute w for linear Kernel
        w = np.zeros((n,1))   #??? n?????
        for i in range(m):
           # print("\n\n",alpha[0][i]*train_y[0][i]*train_X.T[i:i+1].T)
            w = w + alpha[0][i]*train_y[0][i]*train_X.T[i:i+1].T
        z = np.dot(w.T[0], train_X) + b
        xi = hinge_loss(z, train_y)
        sum_xi = 0
        for i in xi:
            sum_xi = sum_xi + i
        return 0.5 *  w.T.dot(w)[0][0] + C * sum_xi[0]
    
    #Guassian Kernel
    w_sq = 0
    for i in range(m):
        for j in range(m):
            xi = np.array([train_X.T[i]]).T
            xj = np.array([train_X.T[j]]).T
            kxx = kernel_function(xi, xj, sigma)[0][0]
            w_sq = w_sq + alpha[0][i]*alpha[0][j]*train_y[0][i]*train_y[0][j]*kxx
    z = np.zeros(m) 
    for i in range(m):
        xi = np.array([train_X.T[i]]).T
        z = z + alpha[0][i] * train_y[0][i] * kernel_function(xi, train_X, sigma)[0]
    z  = z + b
    xi = hinge_loss(z, train_y)
    sum_xi = 0
    for i in xi[0]:
        sum_xi = sum_xi + i
    first_term = 0.5 * w_sq


    # print("fs : ", first_term)
    # print("z : ", z)
    # print("xi : ", xi)
    # print("sum_xi : ", sum_xi)
    return first_term + C * sum_xi
    ## INSERT YOUR CODE HERE
    #########################################

#using train_X as target of X, ignore test_X here
def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    ## INSERT YOUR CODE HERE
    m = train_X.shape[1]
    n = train_X.shape[0]
    m2 = test_X.shape[1]
    if kernel_function.__name__ == "linear_kernel":
        w = np.zeros((n,1))
        for i in range(m):
            w = w + alpha[0][i]*train_y[0][i]*train_X.T[i:i+1].T
            #print("www: ", w)
        #print("wwww:v ",w.shape)
        if w.shape != ():
            #print(w.T)
            #print("test_X:", test_X)
            z = np.dot(w.T, test_X) + b
            
            #print(z)
        else:
            z = np.dot(w, test_X.T) + b
        return z
    
    #Guassian Kernel
    z = np.zeros((1,m2)) 
    # for i in range(m):
    #     for j in range(m):
    #         xi = np.array([train_X.T[i]]).T
    #         xj = np.array([train_X.T[j]]).T
    #         z[i] = z[i] + alpha[0][i] * train_y[0][i] * kernel_function(xj, xi, sigma)[0][0]
    #         print("\n      ",z[i])
    for i in range(m):
        xi = np.array([train_X.T[i]]).T
        
        #print("test x: ", test_X)
        #print("x_i: ", xi)
        z = z + alpha[0][i] * train_y[0][i] * kernel_function(xi, test_X, sigma)
        print("kernel: ", kernel_function(xi, test_X, sigma))
        print("add to z: ", alpha[0][i] * train_y[0][i] * kernel_function(xi, test_X, sigma))
        #print("shape of Z in P2: ", z)
        # print("\nalpha :", alpha[0][i])
        # print("\ntrain_y[0][i] :", train_y[0][i])
        # print("\nkernel_function :", kernel_function(xi, train_X, sigma)[0])
        # print("\nzzz :", alpha[0][i])
    z  = z + b



    return z
    #########################################
