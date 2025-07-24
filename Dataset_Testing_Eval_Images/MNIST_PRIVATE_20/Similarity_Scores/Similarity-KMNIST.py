import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, KMNIST
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import gc
import numpy as np
import torch.nn.functional as F

PRINTOUT = True  

torch.manual_seed(42)
np.random.seed(42)

common_transform = transforms.Compose([
    transforms.ToTensor(),  
])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=common_transform, download=True)
kmnist_dataset = KMNIST(root='./data', train=True, transform=common_transform, download=True)

max_mnist_contribution = len(mnist_dataset) 
max_kmnist_contribution = len(kmnist_dataset) 
total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
total_if_kmnist_maxed = int(len(kmnist_dataset) / 0.96)
total_desired_size = min(total_if_mnist_maxed, total_if_kmnist_maxed)
mnist_size = int(0.04 * total_desired_size)
kmnist_size = int(0.96 * total_desired_size)

mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
kmnist_indices = np.random.choice(len(kmnist_dataset), size=kmnist_size, replace=False)
mnist_subset = Subset(mnist_dataset, mnist_indices)
kmnist_subset = Subset(kmnist_dataset, kmnist_indices)

train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=kmnist_subset, batch_size=64, shuffle=True)

test_dataset = datasets.KMNIST(root='./data', train=False, transform=common_transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

if PRINTOUT:
    print(f"\nDataset Information:")
    print(f"Original MNIST training set: {len(mnist_dataset)} samples")
    print(f"Original KMNIST training set: {len(kmnist_dataset)} samples")
    print(f"Modified MNIST loader: {len(mnist_subset)} samples ({len(mnist_subset)/(len(mnist_subset)+len(kmnist_subset))*100:.1f}%)")
    print(f"Modified KMNIST loader: {len(kmnist_subset)} samples ({len(kmnist_subset)/(len(mnist_subset)+len(kmnist_subset))*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} samples")

    print(f"\nData Loader Batch Information:")
    for batch_idx, (data, target) in enumerate(train_loader_public):
        print(f"MNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        break

    for batch_idx, (data, target) in enumerate(train_loader_private):
        print(f"KMNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        break

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
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

model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

privacy_engine = PrivacyEngine()
gc.collect()




# SLICED WASSERSTEIN DISTANCE
print("\n"*5)
print("SLICED WASSERSTEIN DISTANCE")

from sliced_wasserstein import sliced_wasserstein_distance

def feature_extractor(loader, max_samples=1000):
    """Extract features using the model"""
    features = []
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for x, _ in loader:
            if sample_count >= max_samples:
                break
            
            feature = model(x)
            features.append(feature.numpy())
            sample_count += x.shape[0]
    
    return np.concatenate(features, axis=0)

mnist_features = feature_extractor(train_loader_public)
print("Extracted MNIST features")
 
kmnist_features = feature_extractor(train_loader_private)
print("Extracted K-MNIST features")

print("Sliced Wasserstein Distance between K-MNIST and MNIST:", 
      sliced_wasserstein_distance(kmnist_features, mnist_features))






# HELLINGER DISTANCE
print("\n"*5)
print("HELLINGER DISTANCE")

def hellinger_np(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

kmnist_data = np.array(kmnist_dataset.data)
mnist_data = np.array(mnist_dataset.data)

kmnist_flat = kmnist_data.reshape(len(kmnist_data), -1)
mnist_flat = mnist_data.reshape(len(mnist_data), -1)

kmnist_mean = kmnist_flat.mean(axis=0)
mnist_mean = mnist_flat.mean(axis=0)

kmnist_prob = kmnist_mean / np.sum(kmnist_mean)
mnist_prob = mnist_mean / np.sum(mnist_mean)

print("Hellinger Distance between KMNIST and MNIST:", hellinger_np(kmnist_prob, mnist_prob))





# JENSEN-SHANNON DIVERGENCE
print("\n"*5)
print("JENSEN-SHANNON DIVERGENCE")

def KL_numpy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    epsilon = 1e-10
    b = np.where(b == 0, epsilon, b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

mnist_fresh = MNIST(root='data', train=True, download=True)
kmnist_fresh = KMNIST(root='data', train=True, download=True)

data_np_mnist = mnist_fresh.data.numpy().astype(np.float32) / 255.0
data_np_kmnist = kmnist_fresh.data.numpy().astype(np.float32) / 255.0

data_np_mnist = data_np_mnist[:, None, :, :]  # (60000, 1, 28, 28)
data_np_kmnist = data_np_kmnist[:, None, :, :]

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

datasets = {
    'MNIST': data_np_mnist,
    'KMNIST': data_np_kmnist
}

prob_datasets = {}
avg_datasets = {}

for name, data in datasets.items():
    prob_datasets[name] = to_prob_dist(data)
    avg_datasets[name] = prob_datasets[name].mean(axis=0)

print("Jensen-Shannon Divergence between dataset pairs:")
for name1 in datasets.keys():
    for name2 in datasets.keys():
        if name1 != name2:
            m = (avg_datasets[name1] + avg_datasets[name2]) * 0.5
            js = 0.5 * KL_numpy(avg_datasets[name1], m) + 0.5 * KL_numpy(avg_datasets[name2], m)
            print(f"{name1} vs {name2}: {js:.16f}")






# L2 NORM + COSINE SIMILARITY
print("\n"*5)
print("GRADIENT-BASED COMPARISON")

from torchvision.models import wide_resnet50_2

gradient_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"
gradient_model = wide_resnet50_2(weights=True).to(device)

kmnist_single = DataLoader(
    KMNIST(root='data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True
)

mnist_single = DataLoader(
    MNIST(root='data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True
)

def get_gradient_vector(model, dataloader):
    model.eval()
    model.zero_grad()
    x, y = next(iter(dataloader))
    
    x = x.repeat(1, 3, 1, 1)  
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize(x)
    
    x, y = x.to(device), y.to(device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grad_vector = torch.cat(grads)
    return grad_vector

grad_kmnist = get_gradient_vector(gradient_model, kmnist_single)
grad_mnist = get_gradient_vector(gradient_model, mnist_single)

l2_diff = torch.norm(grad_kmnist - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_kmnist.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()

print(f"L2 norm of gradient difference: {l2_diff:.16f}")
print(f"Cosine similarity between gradients: {cos_sim:.16f}")



# MAXIMUM MEAN DISCREPANCY (MMD)
print("\n"*5)
print("MAXIMUM MEAN DISCREPANCY")

from discrepancies import MaximumMeanDiscrepancy
from kernels import LaplacianKernel

mnist_mmd = MNIST(root='data', train=True, download=True)
kmnist_mmd = KMNIST(root='data', train=True, download=True)

data_np_mnist_mmd = mnist_mmd.data[:5000].numpy().astype(np.float32) / 255.0
data_np_mnist_mmd = data_np_mnist_mmd[:, None, :, :]  

data_np_kmnist_mmd = kmnist_mmd.data[:5000].numpy().astype(np.float32) / 255.0
data_np_kmnist_mmd = data_np_kmnist_mmd[:, None, :, :]


mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(1e-2))
print("MMD between KMNIST and MNIST:", mmd.compute(data_np_kmnist_mmd, data_np_mnist_mmd))





# FRECHET INCEPTION DISTANCE (FID)
print("\n"*5)
print("FRECHET INCEPTION DISTANCE")

from torcheval.metrics import FrechetInceptionDistance

def prepare_images_for_fid(x):
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.0:  
        x = x.div(255.0)
    
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    
    return x

batch_size = 32
fid = FrechetInceptionDistance()

def update_fid_safe(fid, data_tensor, is_real, batch_size=32):
    n = data_tensor.shape[0]
    for i in range(0, min(n, 1000), batch_size): #speed
        batch = data_tensor[i:i+batch_size]
        batch = prepare_images_for_fid(batch)
        fid.update(batch, is_real=is_real)

kmnist_tensor = torch.from_numpy(kmnist_data).unsqueeze(1).float()
mnist_tensor = torch.from_numpy(mnist_data).unsqueeze(1).float()

update_fid_safe(fid, kmnist_tensor, is_real=True)
print("Updated FID with KMNIST.")
update_fid_safe(fid, mnist_tensor, is_real=False)
print("Updated FID with MNIST.")

print("FID between KMNIST and MNIST:", fid.compute().item())