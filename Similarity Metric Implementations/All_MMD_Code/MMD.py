from scipy.stats import laplace, norm
import numpy as np
import matplotlib.pyplot as plt

from discrepancies import MaximumMeanDiscrepancy
from kernels import (
    PolynomialKernel,
    GaussianKernel,
    LaplacianKernel,
    InverseMultiQuadraticKernel,
)

# MNIST Dataset Example
# The MMD can be used on data for which the data generating distribution is unknown. As an example, we can use the MNIST dataset to compare images from the same digit as if they are "samples" from the same underlying distribution for that digit. 
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784")
x = np.array(mnist.data)
y = mnist.target

device = "cuda" if torch.cuda.is_available() else "cpu"

#CIFAR LOAD
cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  

#MNIST LOAD
mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  

#MNIST MANIPULATION
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
data_np_mnist = mnist_tensor[:5000].cpu().numpy()

# plt.imshow(
#     np.stack(
#         [x[y == str(i)][0, :].reshape((28, 28)) for i in range(10)], axis=1
#     ).reshape((28, 28 * 10))
# )
# plt.show()

# We can visualise a heatmap of the MMDs for samples from different digits. 
mmd = MaximumMeanDiscrepancy(kernel=GaussianKernel(1e-3))
data_np_mnist_flat = data_np_mnist.reshape((data_np_mnist.shape[0], -1))
mmd_gaus = mmd.compute(data_np_mnist_flat, data_np_mnist_flat)
print(f"MMD between CIFAR-10 and MNIST: {mmd_gaus}")

# mnist_mmd_gauss = np.zeros((10, 10))
# n_samples = 100
# sigma = 1e-7
# mmd = MaximumMeanDiscrepancy(kernel=GaussianKernel(sigma))
# for i in range(10):
#     a = x[y == str(i)][:n_samples]
#     for j in range(10):
#         if i == j:
#             b = x[y == str(j)][n_samples : 2 * n_samples]
#         else:
#             b = x[y == str(j)][:n_samples]
#             mnist_mmd_gauss[i, j] = mmd.compute(a, b)

# fig, ax = plt.subplots(1, 1)
# plt.imshow(mnist_mmd_gauss, cmap="gray")
# ax.set_title(f"MNIST MMD Digit Comparison")
# ax.set_xlabel("First Digit that was Sampled")
# ax.set_ylabel("Second Digit that was Sampled")
# ax.set_xticks(np.arange(10))
# ax.set_yticks(np.arange(10))
# plt.colorbar()
# plt.show()
