import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, USPS
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import gc
import numpy as np
import torch.nn.functional as F

PRINTOUT = True

transform = transforms.Compose([
    transforms.ToTensor(),  
])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

usps_dataset  = datasets.USPS(root='./data', train=True, transform=transform, download=True)
usps_loader  = DataLoader(dataset=usps_dataset, batch_size=1000, shuffle=False)

max_mnist_contribution = len(mnist_dataset) 
max_usps_contribution = len(usps_dataset) 
total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
total_if_usps_maxed = int(len(usps_dataset) / 0.96)
total_desired_size = min(total_if_mnist_maxed, total_if_usps_maxed)
mnist_size = int(0.04 * total_desired_size)
usps_size = int(0.96 * total_desired_size)
np.random.seed(42)
mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
usps_indices = np.random.choice(len(usps_dataset), size=usps_size, replace=False)
mnist_subset = Subset(mnist_dataset, mnist_indices)
usps_subset = Subset(usps_dataset, usps_indices)
train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=usps_subset, batch_size=64, shuffle=True)


test_dataset  = datasets.USPS(root='./data', train=False, transform=transform, download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
if PRINTOUT:
    print(f"\nDataset Information:")
    print(f"Original MNIST training set: {len(mnist_dataset)} samples")
    print(f"Original USPS training set: {len(usps_dataset)} samples")
    print(f"Modified MNIST loader: {len(mnist_subset)} samples ({len(mnist_subset)/(len(mnist_subset)+len(usps_subset))*100:.1f}%)")
    print(f"Modified USPS loader: {len(usps_subset)} samples ({len(usps_subset)/(len(mnist_subset)+len(usps_subset))*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"\nData Loader Batch Information:")
    for batch_idx, (data, target) in enumerate(train_loader_public):
        print(f"MNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        if batch_idx == 0:  
            break

    for batch_idx, (data, target) in enumerate(train_loader_private):
        print(f"USPS batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        if batch_idx == 0:
            break
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, stride=1, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.fc1 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = x.view(-1, 120)  # Flatten the tensor
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x

model = mnist()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

privacy_engine = PrivacyEngine()


gc.collect()






#SLICED WASSERSTEIN
print("\n"*5)
from sliced_wasserstein import sliced_wasserstein_distance

def feature_extractor(loader):
    features = []
    with torch.no_grad():
        for x, _ in loader:
            feature = model(x)
            features.append(feature.numpy())
    return np.concatenate(features, axis=0)

MNIST_features = feature_extractor(train_loader_public)
print("extracted mnist features")
 
USPS_features = feature_extractor(train_loader_private)
print("extracted USPS features")

print("S.Wasserstein Distance between USPS and MNIST:", sliced_wasserstein_distance(USPS_features, MNIST_features))



#HELLINGER
import numpy as np
from PIL import Image
from torchvision import datasets

def hellinger_np(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

usps_data = np.array(usps_dataset.data)
mnist_data = np.array(mnist_dataset.data)

usps_data_resized = []
for img in usps_data:
    pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
    resized_img = pil_img.resize((28, 28), Image.BILINEAR)
    usps_data_resized.append(np.array(resized_img))

usps_data_resized = np.array(usps_data_resized)  

mnist_data = np.array(mnist_dataset.data)

usps_flat = usps_data_resized.reshape(len(usps_data_resized), -1)
mnist_flat = mnist_data.reshape(len(mnist_data), -1)

usps_mean = usps_flat.mean(axis=0)
mnist_mean = mnist_flat.mean(axis=0)

usps_prob = usps_mean / np.sum(usps_mean)
mnist_prob = mnist_mean / np.sum(mnist_mean)

# Compute Hellinger distance
print("\n"*5)
print("Hellinger Distance between USPS and MNIST:", hellinger_np(usps_prob, mnist_prob))










#JS_DIVERGENCE
def KL_numpy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    epsilon = 1e-10
    b = np.where(b == 0, epsilon, b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
data_np_mnist = mnist_tensor[:50000].numpy()

usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data.astype(np.float32) / 255.0
data_np_usps = data_np_usps[:, None, :, :]  
usps_tensor = torch.from_numpy(data_np_usps)
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)
data_np_usps = usps_tensor.numpy()

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

datasets = {
    'MNIST': data_np_mnist,
    'USPS': data_np_usps
}

prob_datasets = {}
avg_datasets = {}

for name, data in datasets.items():
    prob_datasets[name] = to_prob_dist(data)
    avg_datasets[name] = prob_datasets[name].mean(axis=0)

print("\n"*5)
print("Jensen-Shannon Divergence between dataset pairs:")

for name1 in datasets.keys():
    for name2 in datasets.keys():
        if name1 != name2:
            m = (avg_datasets[name1] + avg_datasets[name2]) * 0.5
            js = 0.5 * KL_numpy(avg_datasets[name1], m) + 0.5 * KL_numpy(avg_datasets[name2], m)
            print(f"{name1} vs {name2}: {js:.6f}")









#L2 Norm + Cosine Similarity
from torchvision.models import wide_resnet50_2
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

transform = transforms.Compose([
    transforms.Resize((32, 32)),            
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor()
])

mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
data_np_mnist = mnist_tensor[:50000].numpy()

usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data.astype(np.float32) / 255.0
data_np_usps = data_np_usps[:, None, :, :]  
usps_tensor = torch.from_numpy(data_np_usps)
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)
data_np_usps = usps_tensor.numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = wide_resnet50_2(weights=True).to(device)
usps_loader = torch.utils.data.DataLoader(
    USPS(root='data', train=True, download=True, transform=transform),
    batch_size=1, shuffle=True, num_workers=0
)
mnist_loader = torch.utils.data.DataLoader(
    MNIST(root='data', train=True, download=True, transform=transform),
    batch_size=1, shuffle=True, num_workers=0
)

def get_gradient_vector(model, dataloader):
    model.eval()
    model.zero_grad()
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))  # flatten
    grad_vector = torch.cat(grads)
    return grad_vector

grad_usps = get_gradient_vector(model, usps_loader)
grad_mnist = get_gradient_vector(model, mnist_loader)

l2_diff = torch.norm(grad_usps - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_usps.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()
print("\n"*5)
print(f"L2 norm of gradient difference: {1/l2_diff:.4f}")
print(f"Cosine similarity between gradients: {cos_sim:.4f}")







#MMD
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

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:5000].numpy().astype(np.float32) / 255.0  # (5000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (5000, 1, 28, 28)
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (5000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()

usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data[:5000].astype(np.float32) / 255.0  
data_np_usps = data_np_usps[:, None, :, :]  
usps_tensor = torch.from_numpy(data_np_usps)
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)
data_np_usps = usps_tensor.numpy()
mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(1e-2))
print("\n"*5)
print("MMD between MNIST and USPS:", mmd.compute(data_np_usps, data_np_mnist))








#FRECHET INCEPTON DISTANCE
print("\n"*5)
import torch
from torcheval.metrics import FrechetInceptionDistance

def prepare_images(x):
    x = x.float().div(255.0)
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return x

batch_size = 64
fid = FrechetInceptionDistance()

def update_fid(fid, data_tensor, is_real):
    n = data_tensor.shape[0]
    for i in range(0, n, batch_size):
        batch = data_tensor[i:i+batch_size]
        batch = prepare_images(batch)
        fid.update(batch, is_real=is_real)

usps_tensor = torch.from_numpy(usps_data).unsqueeze(1).float()
mnist_tensor = torch.from_numpy(mnist_data).unsqueeze(1).float()

update_fid(fid, usps_tensor, is_real=True)
print("Updated with USPS.")
update_fid(fid, mnist_tensor, is_real=False)
print("Updated with MNIST.")

print("FID:", fid.compute().item())