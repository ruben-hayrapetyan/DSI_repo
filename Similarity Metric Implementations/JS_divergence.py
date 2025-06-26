import numpy as np
from torchvision.datasets import CIFAR10, MNIST, USPS, FashionMNIST
import torch
import torch.nn.functional as F
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.clustering import *
from ignite.metrics.regression import *
from ignite.utils import *
from KL_divergence import KL_numpy

device = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR LOAD
cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  

# MNIST LOAD
mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
data_np_mnist = mnist_tensor[:50000].numpy()

# USPS LOAD
usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data.astype(np.float32) / 255.0
data_np_usps = data_np_usps[:, None, :, :]  
usps_tensor = torch.from_numpy(data_np_usps)
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)
data_np_usps = usps_tensor.numpy()

# FashionMNIST LOAD
fm = FashionMNIST(root='data', train=True, download=True)
data_np_fm = fm.data.numpy().astype(np.float32) / 255.0
data_np_fm = data_np_fm[:, None, :, :]  
fm_tensor = torch.from_numpy(data_np_fm)
fm_tensor = F.interpolate(fm_tensor, size=(32, 32), mode='bilinear', align_corners=False)
fm_tensor = fm_tensor.repeat(1, 3, 1, 1)
data_np_fm = fm_tensor[:50000].numpy()

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

# Store datasets in a dictionary for easy iteration
datasets = {
    'CIFAR': data_np_cifar,
    'MNIST': data_np_mnist,
    'USPS': data_np_usps,
    'FashionMNIST': data_np_fm
}

# Convert all datasets to probability distributions
prob_datasets = {}
avg_datasets = {}

for name, data in datasets.items():
    prob_datasets[name] = to_prob_dist(data)
    avg_datasets[name] = prob_datasets[name].mean(axis=0)

# Compute JS divergence for all pairs
print("\n" + "="*60)
print("Jensen-Shannon Divergence between dataset pairs:")
print("="*60)

for name1 in datasets.keys():
    for name2 in datasets.keys():
        if name1 != name2:
            # Compute JS divergence between average distributions
            m = (avg_datasets[name1] + avg_datasets[name2]) * 0.5
            js = 0.5 * KL_numpy(avg_datasets[name1], m) + 0.5 * KL_numpy(avg_datasets[name2], m)
            print(f"{name1} vs {name2}: {js:.6f}")