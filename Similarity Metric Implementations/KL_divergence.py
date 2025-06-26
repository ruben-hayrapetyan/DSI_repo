import numpy as np
import torch

def KL_torch(a, b):
    if torch.is_tensor(a):
        a = a.float()
    else:
        a = torch.tensor(a, dtype=torch.float32)
    
    if torch.is_tensor(b):
        b = b.float()
    else:
        b = torch.tensor(b, dtype=torch.float32)
    
    eps = 1e-8
    b = b + eps
    return torch.sum(torch.where(a != 0, a * torch.log(a / b), torch.tensor(0.0)))

def KL_numpy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    epsilon = 1e-10
    b = np.where(b == 0, epsilon, b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))