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

cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)
cifar_tensor = torch.from_numpy(data_np_cifar).float()

usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data.astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_usps = data_np_usps[:, None, :, :]  # (50000, 1, 28, 28)

usps_tensor = torch.from_numpy(data_np_usps).float()
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_usps = usps_tensor.detach().cpu().numpy()

fm = FashionMNIST(root='data', train=True, download=True)
data_np_fm = fm.data.numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_fm = data_np_fm[:, None, :, :]  # (50000, 1, 28, 28)

fm_tensor = torch.from_numpy(data_np_fm).float()
fm_tensor = F.interpolate(fm_tensor, size=(32, 32), mode='bilinear', align_corners=False)
fm_tensor = fm_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_fm = fm_tensor.detach().cpu().numpy()

# Load MNIST (only 50000 to match CIFAR)
mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)

# Interpolate MNIST to (32, 32) and 3 channels
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

metric = JSDivergence()
metric.attach(default_evaluator, 'js-div')
state = default_evaluator.run([[mnist_tensor, fm_tensor]])
print("JS Divergence: ", state.metrics['js-div'])