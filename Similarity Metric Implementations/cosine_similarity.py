import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn.functional as F

def cosine_similarity(np_array1, np_array2):
    print("FIRST ARRAY FLATTENED:", np_array1.flatten())
    print("SECOND ARRAY FLATTENED:", np_array2.flatten())
    print(f"Shape of First Array/First Array Flattened: {np_array1.shape}/{np_array1.flatten().shape}")
    print(f"Shape of Second Array/Second Array Flattened: {np_array2.shape}/{np_array2.flatten().shape}")
    numerator = np.dot(np_array1.flatten(), np_array2.flatten())
    print(numerator)
    sum_1 = 0
    sum_2 = 0
    for element in np_array1:
        sum_1+=element**2
    for element in np_array2:
        sum_2+=element**2
    denominator = np.sqrt(sum_1)*np.sqrt(sum_2)
    cosine_similarity = (numerator*1.0)/denominator
    return cosine_similarity

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
data_np_mnist = mnist_tensor[:50000].detach().cpu().numpy()

print("Cosine Similarity (only cosine):", cosine_similarity(data_np_mnist, data_np_cifar))
print("Cosine Similarity Score:", np.arccos(cosine_similarity(data_np_mnist, data_np_cifar)))

