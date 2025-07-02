import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, SEMEION
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

semeion_dataset = datasets.SEMEION(root='./data', transform=transform, download=True)
semeion_loader = DataLoader(dataset=semeion_dataset, batch_size=1000, shuffle=False)

max_mnist_contribution = len(mnist_dataset) 
max_semeion_contribution = len(semeion_dataset) 
total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
total_if_semeion_maxed = int(len(semeion_dataset) / 0.96)
total_desired_size = min(total_if_mnist_maxed, total_if_semeion_maxed)
mnist_size = int(0.04 * total_desired_size)
semeion_size = int(0.96 * total_desired_size)
np.random.seed(42)
mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
semeion_indices = np.random.choice(len(semeion_dataset), size=semeion_size, replace=False)
mnist_subset = Subset(mnist_dataset, mnist_indices)
semeion_subset = Subset(semeion_dataset, semeion_indices)
train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=semeion_subset, batch_size=64, shuffle=True)

# Note: SEMEION doesn't have a separate test set, so we'll use a portion of the full dataset
# Split the semeion dataset for testing
test_size = int(0.2 * len(semeion_dataset))
test_indices = np.random.choice(len(semeion_dataset), size=test_size, replace=False)
test_subset = Subset(semeion_dataset, test_indices)
test_loader = DataLoader(dataset=test_subset, batch_size=1000, shuffle=False)

if PRINTOUT:
    print(f"\nDataset Information:")
    print(f"Original MNIST training set: {len(mnist_dataset)} samples")
    print(f"Original Semeion dataset: {len(semeion_dataset)} samples")
    print(f"Modified MNIST loader: {len(mnist_subset)} samples ({len(mnist_subset)/(len(mnist_subset)+len(semeion_subset))*100:.1f}%)")
    print(f"Modified Semeion loader: {len(semeion_subset)} samples ({len(semeion_subset)/(len(mnist_subset)+len(semeion_subset))*100:.1f}%)")
    print(f"Test set: {len(test_subset)} samples")
    print(f"\nData Loader Batch Information:")
    for batch_idx, (data, target) in enumerate(train_loader_public):
        print(f"MNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        if batch_idx == 0:  
            break

    for batch_idx, (data, target) in enumerate(train_loader_private):
        print(f"Semeion batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        if batch_idx == 0:
            break

# Modified model to handle different input sizes
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
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # Resize input to 28x28 if it's 16x16 (Semeion)
        if x.shape[-1] == 16:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = self.adaptive_pool(x)  # Use adaptive pooling
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
 
Semeion_features = feature_extractor(train_loader_private)
print("extracted Semeion features")

print("S.Wasserstein Distance between Semeion and MNIST:", sliced_wasserstein_distance(Semeion_features, MNIST_features))

#HELLINGER
import numpy as np

def hellinger_np(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

# Extract raw data from datasets
semeion_data = []
for i in range(len(semeion_dataset)):
    img, _ = semeion_dataset[i]
    semeion_data.append(img.squeeze().numpy())
semeion_data = np.array(semeion_data)

mnist_data = np.array(mnist_dataset.data.numpy()) / 255.0  # Normalize to 0-1

# Resize Semeion to match MNIST for comparison
semeion_resized = np.array([F.interpolate(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), 
                                        size=(28, 28), mode='bilinear', align_corners=False).squeeze().numpy() 
                           for img in semeion_data])

semeion_flat = semeion_resized.reshape(len(semeion_resized), -1)
mnist_flat = mnist_data.reshape(len(mnist_data), -1)

semeion_mean = semeion_flat.mean(axis=0)
mnist_mean = mnist_flat.mean(axis=0)

semeion_prob = semeion_mean / np.sum(semeion_mean)
mnist_prob = mnist_mean / np.sum(mnist_mean)

# Compute Hellinger distance
print("\n"*5)
print("Hellinger Distance between Semeion and MNIST:", hellinger_np(semeion_prob, mnist_prob))

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

transform = transforms.Compose([
    transforms.ToTensor()
])

semeion = SEMEION(root='data', download=True, transform=transform)
semeion_data_raw = []
for i in range(len(semeion)):
    img, _ = semeion[i]
    semeion_data_raw.append(img.squeeze().numpy())

data_np_semeion = np.array(semeion_data_raw)
data_np_semeion = data_np_semeion[:, None, :, :]
semeion_tensor = torch.from_numpy(data_np_semeion)
semeion_tensor = F.interpolate(semeion_tensor, size=(32, 32), mode='bilinear', align_corners=False)
semeion_tensor = semeion_tensor.repeat(1, 3, 1, 1)
data_np_semeion = semeion_tensor[:len(semeion)].numpy()

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

datasets_dict = {
    'MNIST': data_np_mnist,
    'Semeion': data_np_semeion
}

prob_datasets = {}
avg_datasets = {}

for name, data in datasets_dict.items():
    prob_datasets[name] = to_prob_dist(data)
    avg_datasets[name] = prob_datasets[name].mean(axis=0)

print("\n"*5)
print("Jensen-Shannon Divergence between dataset pairs:")

for name1 in datasets_dict.keys():
    for name2 in datasets_dict.keys():
        if name1 != name2:
            m = (avg_datasets[name1] + avg_datasets[name2]) * 0.5
            js = 0.5 * KL_numpy(avg_datasets[name1], m) + 0.5 * KL_numpy(avg_datasets[name2], m)
            print(f"{name1} vs {name2}: {js:.12f}")

#L2 Norm + Cosine Similarity
from torchvision.models import wide_resnet50_2
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np

transform_for_resnet = transforms.Compose([
    transforms.ToTensor(),                    # Convert PIL Image to tensor first
    transforms.Resize((224, 224)),            
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"
model_resnet = wide_resnet50_2(weights=True).to(device)

# Create single sample loaders for gradient computation
semeion_single = SEMEION(root='./data', transform=transform_for_resnet, download=True)
mnist_single = MNIST(root='./data', train=True, transform=transform_for_resnet, download=True)

semeion_single_loader = DataLoader(semeion_single, batch_size=1, shuffle=True)
mnist_single_loader = DataLoader(mnist_single, batch_size=1, shuffle=True)

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

grad_semeion = get_gradient_vector(model_resnet, semeion_single_loader)
grad_mnist = get_gradient_vector(model_resnet, mnist_single_loader)

l2_diff = torch.norm(grad_semeion - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_semeion.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()
print("\n"*5)
print(f"L2 norm of gradient difference: {1/l2_diff:.12f}")
print(f"Cosine similarity between gradients: {cos_sim:.12f}")

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

semeion = SEMEION(root='data', download=True, transform=transform)
semeion_data_for_mmd = []
for i in range(min(5000, len(semeion))):
    img, _ = semeion[i]
    semeion_data_for_mmd.append(img.squeeze().numpy())
data_np_semeion = np.array(semeion_data_for_mmd)
data_np_semeion = data_np_semeion[:, None, :, :]  # (N, 1, 16, 16)
semeion_tensor = torch.from_numpy(data_np_semeion)
semeion_tensor = F.interpolate(semeion_tensor, size=(32, 32), mode='bilinear', align_corners=False)
semeion_tensor = semeion_tensor.repeat(1, 3, 1, 1)
data_np_semeion = semeion_tensor.numpy()

mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(1e-2))
print("\n"*5)
print("MMD between MNIST and Semeion:", mmd.compute(data_np_semeion, data_np_mnist))





from torcheval.metrics import FrechetInceptionDistance
print("\n"*5)

def prepare_images(x):
    x = x.float()
    if x.max() > 1.0: 
        x = x / 255.0
    
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    
    return x

batch_size = 64
fid = FrechetInceptionDistance()

def update_fid(fid, dataset, is_real):
    transform = transforms.ToTensor()
    
    data_list = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        data_list.append(img.unsqueeze(0))
    
    data_tensor = torch.cat(data_list, dim=0)
    
    n = data_tensor.shape[0]
    for i in range(0, n, batch_size):
        batch = data_tensor[i:i+batch_size]
        batch = prepare_images(batch)
        fid.update(batch, is_real=is_real)

semeion_fid = SEMEION(root='./data', download=True)
mnist_fid = MNIST(root='./data', train=True, download=True)

print("Updating FID with Semeion dataset...")
update_fid(fid, semeion_fid, is_real=True)
print("Updated with Semeion.")

print("Updating FID with MNIST dataset...")
update_fid(fid, mnist_fid, is_real=False)
print("Updated with MNIST.")

fid_score = fid.compute().item()
print(f"FID Score: {fid_score:.6f}")