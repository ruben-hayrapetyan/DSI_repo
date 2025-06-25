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
np.random.seed(0)
# Maximum Mean Discrepanc

# # Polynomial Kernels
# n_dimensions = 5
# n_samples = 25
# n_trials = 5
# sigma = 0.1
# m, n = 30, 30
# n_polynomials = 4
# covariance_scales = np.arange(1, 20)

# mu_x = np.zeros(n_dimensions)
# covariance_x = np.eye(n_dimensions)
# mu_y = np.zeros(n_dimensions)

# fig, axs = plt.subplots(1, n_polynomials)
# fig.set_figheight(5)
# fig.set_figwidth(20)

# for i, p in enumerate(np.arange(1, n_polynomials + 1)):
#     X, MMDs = [], []
#     mmd = MaximumMeanDiscrepancy(kernel=PolynomialKernel(p))
#     for covariance_scale in covariance_scales:
#         covariance_y = covariance_scale * np.eye(n_dimensions)
#         X.extend([covariance_scale] * n_trials)
#         MMDs.extend(
#             [
#                 mmd.compute(
#                     x=np.random.multivariate_normal(mu_x, covariance_x, n_samples),
#                     y=np.random.multivariate_normal(mu_y, covariance_y, n_samples),
#                 )
#                 for _ in range(n_trials)
#             ]
#         )
#     axs[i].scatter(X, MMDs)
#     axs[i].set_title(f"MMD Covariance Comparison for p={p}")
#     axs[i].set_xlabel("Covariance of 2nd Distribution")
#     axs[i].set_ylabel("MMD")
# n_dimensions = 5
# n_samples = 25
# n_trials = 5
# sigma = 0.1
# m, n = 30, 30
# n_polynomials = 4
# mu_shifts = np.arange(0, 20)

# mu_x = np.zeros(n_dimensions)
# covariance_x = np.eye(n_dimensions)
# covariance_y = np.eye(n_dimensions)

# fig, axs = plt.subplots(1, n_polynomials)
# fig.set_figwidth(20)
# fig.set_figheight(5)
# for i, p in enumerate(np.arange(1, n_polynomials + 1)):
#     X, MMDs = [], []
#     mmd = MaximumMeanDiscrepancy(kernel=PolynomialKernel(p))
#     for mu_shift in mu_shifts:
#         mu_y = np.zeros(n_dimensions) + mu_shift
#         X.extend([mu_shift] * n_trials)
#         MMDs.extend(
#             [
#                 mmd.compute(
#                     x=np.random.multivariate_normal(mu_x, covariance_x, n_samples),
#                     y=np.random.multivariate_normal(mu_y, covariance_y, n_samples),
#                 )
#                 for _ in range(n_trials)
#             ]
#         )
#     axs[i].scatter(X, MMDs)
#     axs[i].set_title(f"MMD Mean Test with p={p}")
#     axs[i].set_xlabel("Mean of 2nd Distribution")
#     axs[i].set_ylabel("MMD")
# # Convergence of $\hat{MMD}$
# n_dimensions = 3
sigma = 1e-2
# mu = np.zeros(n_dimensions)
# n_trials = 20
# X = []
# MMDs = []
# max_n_sample = 50
mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(sigma))
# for n_sample in range(2, max_n_sample):
#     X.extend([n_sample] * n_trials)
#     MMDs.extend(
#         [
#             mmd.compute(
#                 x=np.random.multivariate_normal(
#                     mu, np.eye(n_dimensions, n_dimensions), n_sample
#                 ),
#                 y=np.random.multivariate_normal(
#                     mu, np.eye(n_dimensions, n_dimensions), n_sample
#                 ),
#             )
#             for _ in range(n_trials)
#         ]
#     )

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# fig.set_figwidth(20)
# fig.set_figheight(10)

# bound_x = np.arange(2, max_n_sample)
# ax.plot(
#     bound_x,
#     np.std(MMDs[:n_trials]) * np.sqrt(2) / np.sqrt(bound_x),
#     "r",
#     label="O(n^{-0.5})",
# )
# plt.scatter(X, MMDs)

# ax.set_title(f"MMD Convergence")
# ax.set_xlabel("Number of Samples")
# ax.set_ylabel("|Emperical MMD - MMD|")
# ax.legend()
# plt.show()
# MNIST Dataset Example
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")
x = np.array(mnist.data)
y = mnist.target
y.shape
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
data_np_mnist = mnist_tensor[:50000].numpy()
# plt.imshow(
#     np.stack(
#         [x[y == str(i)][0, :].reshape((28, 28)) for i in range(10)], axis=1
#     ).reshape((28, 28 * 10))
# )
# plt.show()
print(mmd.compute(data_np_cifar, data_np_mnist))
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