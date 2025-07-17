import numpy as np
from sklearn import datasets as sk_datasets
# from scipy.stats import wasserstein_distance_nd
from sliced_wasserstein import sliced_wasserstein_distance
from torchvision import datasets as tv_datasets
import torch
import torch.nn.functional as F

cifar = tv_datasets.CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)
data_np_cifar = data_np_cifar.reshape(50000, -1)

# Load MNIST (only 50000 to match CIFAR)
mnist = tv_datasets.MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:50000].numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()
data_np_mnist = data_np_mnist.reshape(50000, -1)

fashion_mnist = tv_datasets.FashionMNIST(root='data', train=False, download=True)
data_np_fashion = fashion_mnist.data.numpy().astype(np.float32) / 255.0  # (10000, 28, 28)
data_np_fashion = data_np_fashion[:, None, :, :]  # (10000, 1, 28, 28)
fashion_tensor = torch.from_numpy(data_np_fashion).float()
fashion_tensor = F.interpolate(fashion_tensor, size=(32, 32), mode='bilinear', align_corners=False)
fashion_tensor = fashion_tensor.repeat(1, 3, 1, 1)  # (10000, 3, 32, 32)
data_np_fashion = fashion_tensor.numpy()  # (10000, 3, 32, 32)
data_np_fashion = np.tile(data_np_fashion, (5, 1, 1, 1))  # (50000, 3, 32, 32)
data_np_fashion = data_np_fashion.reshape(50000, -1)  # Now shape is (50000, 3072)

print("S.Wasserstein Distance between MNIST and CIFAR:", sliced_wasserstein_distance(data_np_mnist, data_np_cifar))
print("S.Wasserstein Distance between MNIST and FASHION:", sliced_wasserstein_distance(data_np_mnist, data_np_fashion))
print("S.Wasserstein Distance between CIFAR and FASHION:", sliced_wasserstein_distance(data_np_cifar, data_np_fashion))
