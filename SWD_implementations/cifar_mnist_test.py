import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torch
from SWD_implementation import swd
import torch.nn.functional as F

def main():
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
    mnist_tensor = mnist_tensor[:50000]
    x1 = mnist_tensor.float().to(device)
    x2 = torch.from_numpy(data_np_cifar[:50000]).float().to(device)
    print(x1.shape, x2.shape)
    out = swd(x1, x2, device=device)
    print(out)

if __name__ == "__main__":
    main()