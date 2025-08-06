import math
import numpy as np

def dp_newton (S_n, lambda_0, theta, rho, w_0, T, SOI):
    """
    Implementtation of the differentially private Newton's (second order update) method for 
    logistic regression from Faster Differentially Private Convex Optimization
    via Second-Order Methods
    Parameters:
        S_n (np.ndarray): training set, must be preproccessed to a 2D NumPy array, last column
                          contains labels, other columns contain features
        lambda_0 (float): minimum eigenvalue threshold
        theta (float): privacy budget allocation between noise injection steps, (0, 1)
        rho (float): total privacy budget specified in zero-concentrated differential privacy
        w_0 (np.ndarray): initial weight vector, 1d array
        T (int): number of iterations
        SOI (str): method for modifying Hessian, (clip, add)
    Returns:
        w_T (np.ndarray): final weight vector after optimization
    """
    
    n = S_n.shape[0]
    d = S_n.shape[1] - 1
    I_d = np.identity(d)
    w = w_0
    
    sigma_1 = math.sqrt(T) / (n * math.sqrt(2 * rho * (1 - theta)))
    
    if SOI == "add":
        sigma_2 = math.sqrt(T) / ((4 * n * lambda_0 * lambda_0 + lambda_0) * math.sqrt(2 * rho * theta))
    elif SOI == "clip":
        sigma_2 = math.sqrt(T) / ((4 * n * lambda_0 * lambda_0 - lambda_0) * math.sqrt(2 * rho * theta))
    else:
        print("SOI must be set to add or clip")
        return
    
    for t in range (T):
        G_t_init = gradient(w, S_n)
        H_t_init = hessian(w, S_n, d)

        if SOI == "add":
            H_t = H_t_init + lambda_0 * I_d
        elif SOI == "clip":
            eigenvals, eigenvectors = np.linalg.eigh(H_t_init)
            eigenvals_new = np.maximum(eigenvals, lambda_0)
            eigen_mat = np.diag(eigenvals_new)
            H_t = eigenvectors @ eigen_mat @ eigenvectors.T
        
        G_t = G_t_init + gaussian_noise(0, sigma_1 * sigma_1, d)

        w = w - inverse(H_t) @ G_t + gaussian_noise(0, (norm(G_t) ** 2) * (sigma_2 ** 2), d)

    return w

def gradient(w, S_n):
    sum = np.zeros_like(w)
    n = S_n.shape[0]
    for i in range (n):
        x_i = S_n[i, :-1]
        y_i = S_n[i, -1]
        sum += (-1 * x_i * y_i) / (1 + np.exp(y_i * np.dot(w, x_i)))
    return 1/n * sum

def hessian(w, S_n, d):
    sum = np.zeros((d, d))
    n = S_n.shape[0]
    for i in range (n):
        x_i = S_n[i, :-1]
        y_i = S_n[i, -1]
        sum += (np.outer(x_i, x_i)) / ((np.exp(-1 * np.dot(w, x_i) / 2) + np.exp(np.dot(w, x_i) / 2)) ** 2)
    return 1/n * sum

def gaussian_noise(mean, var, d):
    return np.random.normal(mean, math.sqrt(var), d)

def inverse(mat):
    return np.linalg.inv(mat)

def norm(vec):
    return np.linalg.norm(vec)