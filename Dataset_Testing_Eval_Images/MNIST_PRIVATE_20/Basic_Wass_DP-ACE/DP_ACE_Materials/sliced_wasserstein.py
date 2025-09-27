import numpy as np
from scipy.stats import wasserstein_distance


def sliced_wasserstein_distance(X, Y, n_projections=1000, p=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_features = X.shape[1]

    projections = np.random.randn(n_projections, n_features)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    wasserstein_distances = []
    
    for i in range(n_projections):
        theta = projections[i]
        X_proj = X @ theta
        Y_proj = Y @ theta
        wd = wasserstein_distance(X_proj, Y_proj)
        wasserstein_distances.append(wd ** p)
    
    sliced_wd = (np.mean(wasserstein_distances)) ** (1.0 / p)
    
    return sliced_wd
