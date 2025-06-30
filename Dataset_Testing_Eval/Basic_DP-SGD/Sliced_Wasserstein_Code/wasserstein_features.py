import numpy as np
from scipy.stats import wasserstein_distance_nd
from torchvision import transforms
from torchvision import datasets as tv_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sliced_wasserstein import sliced_wasserstein_distance
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)               
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)       
        self.fc2 = nn.Linear(128, 10)             

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   
        x = self.pool(F.relu(self.conv2(x)))      
        x = x.view(x.size(0), -1)               
        x = F.relu(self.fc1(x))                
        return x

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])

mnist = tv_datasets.MNIST(root='data', train=True, transform=transform, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=1000, shuffle=False)
print("loaded mnnist")

cifar = tv_datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
cifar_loader = torch.utils.data.DataLoader(cifar, batch_size=1000, shuffle=False)
print("loader cifar")

fashion_mnist = tv_datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
fashion_loader = DataLoader(dataset=fashion_mnist, batch_size=1000, shuffle=False)


cnn = CNN()

def feature_extractor(loader):
    features = []
    with torch.no_grad():
        for x, _ in loader:
            feature = cnn(x)
            features.append(feature.numpy())
    return np.concatenate(features, axis=0)

mnist_features = feature_extractor(mnist_loader)
print("extracted mnist features")

cifar_features = feature_extractor(cifar_loader)
print("extracted cifar features")

fashion_features = feature_extractor(fashion_loader)

print("S.Wasserstein Distance between MNIST and CIFAR:", sliced_wasserstein_distance(mnist_features, cifar_features))
print("S.Wasserstein Distance between MNIST and FASHION:", sliced_wasserstein_distance(mnist_features, fashion_features))
print("S.Wasserstein Distance between CIFAR and FASHION:", sliced_wasserstein_distance(cifar_features, fashion_features))
