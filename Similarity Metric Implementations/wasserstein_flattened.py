import numpy as np
from scipy.stats import wasserstein_distance_nd
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn.functional as F

cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)
data_np_cifar = data_np_cifar.reshape(50000, -1)

# Load MNIST (only 50000 to match CIFAR)
mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:50000].numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)

# Interpolate MNIST to (32, 32) and 3 channels
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()
data_np_mnist = data_np_mnist.reshape(50000, -1)

print("Wasserstein Distance:", wasserstein_distance_nd(data_np_mnist, data_np_cifar))