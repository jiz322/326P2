# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np
import random as rng
import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            train_y: 1 x m labels (-1 or 1) of training data.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #For same index, duals is less than primals
    print("maxpass: \n", max_passes)
    iter_num = []
    duals = []
    primals = []
    models = []
    for t in range(max_iters):
        #print("\n\n\n\n\n\nt: ", t)
        num_passes = 0
        # counter = 0
        while num_passes < max_passes:
            # counter = counter + 1
            # if counter == 10:
            #     break
            num_changes = 0
            for i in range(model.m):
                #print("\nstarting alpha\n", model.alpha)
                print("\ni: \n", i)
                if violate_KKT(model, i, tol):
                    j = int(rng.uniform(0, model.m))
                    while j == i: #j has to be different than i
                        j = int(rng.uniform(0, model.m))

                    #Due to the box constraint 0 ≤ α2 ≤ C and C − ρ > −ρ, the updated
                    # αnew must be upper-bounded by ≤ H = min{C, C − ρ} = min{C, C − (αold − αold)}, where 212
                    # αold are the values before the update. For the same reasons, the lower bound of αnew will be i2
                    # L = max{0, −ρ} = max{0, −(αold − αold )}
                    #Compute the lower bound L and upper bound H for new_alpha_j
                    L_j = 0
                    H_j = 0
                    if model.train_y[0][i]*model.train_y[0][j] == -1:
                        H_j = np.minimum(model.C, model.C - (model.alpha[0][i] - model.alpha[0][j]))
                        L_j = np.maximum(0, -(model.alpha[0][i] - model.alpha[0][j]))
                    elif model.train_y[0][i]*model.train_y[0][j] == 1:
                        H_j = np.minimum(model.C, model.alpha[0][i] + model.alpha[0][j])
                        L_j = np.maximum(0, model.alpha[0][i] + model.alpha[0][j] - model.C)
                    else:
                        print("ERORR: product of y should be 1 or -1!!!!!")
                    
                    #Compute K11, K22, K12
                    K = np.zeros((2,2))
                    xi = np.array([model.train_X.T[i]]).T
                    xj = np.array([model.train_X.T[j]]).T
                    for iter1 in range(2):
                        for iter2 in range(2):
                            K[iter1][iter2] = model.kernel_func(xi, xj, model.sigma)[0][0]
                    #To compute g1 g[0], g2 g[1]
                    g = [0, 0]
                    for iter1 in range(model.m):  
                        x_iter = np.array([model.train_X.T[iter1]]).T
                        Kim = model.kernel_func(xi, x_iter, model.sigma)[0][0]
                        Kjm = model.kernel_func(xj, x_iter, model.sigma)[0][0]
                        g[0] = g[0] + model.alpha[0][iter1] * model.train_y[0][iter1] * Kim
                        g[1] = g[1] + model.alpha[0][iter1] * model.train_y[0][iter1] * Kjm
                    g[0] = g[0] + model.b
                    g[1] = g[1] + model.b
                    #Using L and H, compute new_alpha_j
                    nominator = model.train_y[0][j] * (g[0] - model.train_y[0][i] - (g[1] - model.train_y[0][j]))
                    denominator = K[0][0] + K[1][1] - 2 * K[0][1] + tol
                    new_alpha_j = model.alpha[0][j] + nominator / denominator
                    #clip
                    if new_alpha_j > H_j:
                        new_alpha_j = H_j
                    if new_alpha_j < L_j:
                        new_alpha_j = L_j
                    #if does not change much
                    if np.absolute(new_alpha_j - model.alpha[0][j]) <= tol:
                        #print("here")
                        continue
                    
                    #compute the new value of model.alpha[0][i] given new_alpha_j
                    new_alpha_i = model.alpha[0][i] + model.train_y[0][i] * model.train_y[0][j] * (model.alpha[0][j] - new_alpha_j)

                    #E1, E2, alpha diff before update alpha
                    E1 = g[0] - model.train_y[0][i]
                    E2 = g[1] - model.train_y[0][j]
                    alpha_diff_i = model.alpha[0][i] - new_alpha_i
                    alpha_diff_j = model.alpha[0][j] - new_alpha_j
                    #Commit two new values to the actual dual variables
                    model.alpha[0][i] = new_alpha_i
                    model.alpha[0][j] = new_alpha_j
                    #update the bias b
                    b1 = model.b - E1 - model.train_y[0][i] * alpha_diff_i * K[0][0] - model.train_y[0][j] * alpha_diff_j * K[0][1]
                    b2 = model.b - E2 - model.train_y[0][i] * alpha_diff_i * K[0][1] - model.train_y[0][j] * alpha_diff_j * K[1][1]
                    can_b1 = (new_alpha_i > 0 and new_alpha_i < model.C)
                    can_b2 = (new_alpha_j > 0 and new_alpha_j < model.C)
                    if can_b2:
                        model.b = b2
                    if can_b1:
                        model.b = b1
                    if not (can_b1 or can_b2):
                        model.b = 0.5 * (b1 + b2)
                    #Increase the num_changes by  1
                    num_changes = num_changes + 1
            #One pass without changing any parameters
            print("\nnum_change: ",num_changes)
            if num_changes <= 2:
                num_passes = num_passes + 1
            #At least one pair of alpha's are changed
            else:
                num_passes = 0
        #Record dual and primal objective function values and the modell parameters (all dual variables and b) every "record every" iterations
        iter_num.append(t)
        duals.append(dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.sigma))
        primals.append(primal_objective_function(model.alpha, model.train_y, model.train_X, model.b, model.C, model.kernel_func, model.sigma))
        models.append(copy.deepcopy(model))
    #return the recoded objective function values and modell parameters over the course of training
    #iter_num, duals, primals, models with list type
    return iter_num, duals, primals, models
    #########################################

#If model.alpha[0][i] violate KTT, return True
def violate_KKT(model, i, tol):
    z = decision_function(model.alpha, model.train_y, model.train_X, model.b, model.kernel_func, model.sigma, model.train_X)
    xi = hinge_loss(z, model.train_y)

    #possible error, so I print out values here
    #print("\n (xi[i], alpha_i) : ", (xi[0][i], model.alpha[0][i]))
    #print("\n z : ", z)

    if model.alpha[0][i] >= 0 - tol and model.alpha[0][i] < model.C + tol:
        if xi[0][i] >= 0 - tol and xi[0][i] <= 0 + tol:
            return False
    if model.alpha[0][i] >= model.C - tol and model.alpha[0][i] <= model.C + tol:
        #print("KKT(67): ", z[i]*model.train_y[0][i] - 1 + xi[0][i])
        if xi[0][i] >= 0 - tol:
            return False
    #print("True")
    return True

def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    ## INSERT YOUR CODE HERE
    z = decision_function(model.alpha, model.train_y, model.train_X, model.b, model.kernel_func, model.sigma, test_X)
    print("Z: ", z)
    if (len(z.shape)) == 1:
        z = np.array(z[np.newaxis,:])
    for i in range(z.shape[1]):
        if z[0][i] >= 0:
            z[0][i] = 1
        else:
            z[0][i] = -1
    return z
    #########################################
