import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn.functional as F

def cosine_similarity(np_array1, np_array2):
    # print("FIRST ARRAY FLATTENED", np_array1.flatten())
    # print("SECOND ARRAY FLATTENED", np_array2.flatten())
    # print(f"Shape of First Array/First Array Flattened: {np_array1.shape}/{np_array1.flatten().shape}")
    # print(f"Shape of Second Array/Second Array Flattened: {np_array2.shape}/{np_array2.flatten().shape}")
    # print("Unique MNIST elements FLATTENED:", np.unique(np_array1))
    # print("Unique CIFAR elemetns FLATTENED:", np.unique(np_array2))
    numerator = np.dot(np_array1.flatten(), np_array2.flatten())
    sum_1 = np.sum(np_array1 ** 2)
    sum_2 = np.sum(np_array2 ** 2)
    denominator = np.sqrt(sum_1)*np.sqrt(sum_2)
    # print("DENOMENATOR:", denominator)
    # print("NUMERATOR:", numerator)
    cosine_similarity = (numerator*1.0)/denominator
    return cosine_similarity

# device = "cuda" if torch.cuda.is_available() else "cpu"
# #CIFAR LOAD
# cifar = CIFAR10(root='data', train=True, download=True)
# data_np_cifar = cifar.data.astype(np.float32) / 255.0
# data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  
# #MNIST LOAD
# mnist = MNIST(root='data', train=True, download=True)
# data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0
# print("Before interpolation — max:", data_np_mnist.max(), "min:", data_np_mnist.min())
# data_np_mnist = data_np_mnist[:, None, :, :]  
# #MNIST MANIPULATION
# mnist_tensor = torch.from_numpy(data_np_mnist)
# mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
# mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
# print("After interpolation — max:", mnist_tensor.max(), "min:", mnist_tensor.min())
# data_np_mnist = mnist_tensor[:50000].detach().cpu().numpy()
# print("Final numpy version — max:", data_np_mnist.max(), "min:", data_np_mnist.min())
# cifar = CIFAR10(root='data', train=True, download=True)
# data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
# data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)
cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)

# Load MNIST (only 50000 to match CIFAR)
mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:50000].numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)

# Interpolate MNIST to (32, 32) and 3 channels
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()

# print("MNIST shape:", data_np_mnist.shape)
# print("CIFAR shape:", data_np_cifar.shape)
# print("MNIST mean:", data_np_mnist.mean(), "max:", data_np_mnist.max())
# print("CIFAR mean:", data_np_cifar.mean(), "max:", data_np_cifar.max())
# print("Unique MNIST elements:", np.unique(data_np_mnist))
# print("Unique CIFAR elemetns:", np.unique(data_np_cifar))

print("Cosine Similarity (only cosine):", cosine_similarity(data_np_mnist, data_np_cifar))
print("Cosine Similarity Score:", np.arccos(cosine_similarity(data_np_mnist, data_np_cifar)))

