import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, SVHN
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import gc
import numpy as np
import torch.nn.functional as F

PRINTOUT = True 

torch.manual_seed(42)
np.random.seed(42)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),  
])

svhn_transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)
svhn_dataset = SVHN(root='./data', split="train", transform=svhn_transform, download=True)


max_mnist_contribution = len(mnist_dataset) 
max_svhn_contribution = len(svhn_dataset) 
total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
total_if_svhn_maxed = int(len(svhn_dataset) / 0.96)
total_desired_size = min(total_if_mnist_maxed, total_if_svhn_maxed)
mnist_size = int(0.04 * total_desired_size)
svhn_size = int(0.96 * total_desired_size)

mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
svhn_indices = np.random.choice(len(svhn_dataset), size=svhn_size, replace=False)
mnist_subset = Subset(mnist_dataset, mnist_indices)
svhn_subset = Subset(svhn_dataset, svhn_indices)

train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=svhn_subset, batch_size=64, shuffle=True)

test_dataset = datasets.SVHN(root='./data', split="test", transform=svhn_transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

if PRINTOUT:
    print(f"\nDataset Information:")
    print(f"Original MNIST training set: {len(mnist_dataset)} samples")
    print(f"Original SVHN training set: {len(svhn_dataset)} samples")
    print(f"Modified MNIST loader: {len(mnist_subset)} samples ({len(mnist_subset)/(len(mnist_subset)+len(svhn_subset))*100:.1f}%)")
    print(f"Modified SVHN loader: {len(svhn_subset)} samples ({len(svhn_subset)/(len(mnist_subset)+len(svhn_subset))*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} samples")

    print(f"\nData Loader Batch Information:")
    for batch_idx, (data, target) in enumerate(train_loader_public):
        print(f"MNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        break

    for batch_idx, (data, target) in enumerate(train_loader_private):
        print(f"SVHN batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
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
print("\n" *5)
print("S. WASSERSTEIN DISTANCE")

# Note: Uncomment if you have the sliced_wasserstein library
from sliced_wasserstein import sliced_wasserstein_distance

def feature_extractor(loader, max_samples=1000):
    features = []
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for x, _ in loader:
            if sample_count >= max_samples:
                break
            
            if x.shape[1] == 3:
                x = transforms.functional.rgb_to_grayscale(x)
            
            if x.shape[-1] != 28:
                x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            
            feature = model(x)
            features.append(feature.numpy())
            sample_count += x.shape[0]
    
    return np.concatenate(features, axis=0)

mnist_features = feature_extractor(train_loader_public)
print("Extracted MNIST features")
 
svhn_features = feature_extractor(train_loader_private)
print("Extracted SVHN features")

print("Sliced Wasserstein Distance between SVHN and MNIST:", sliced_wasserstein_distance(svhn_features, mnist_features))



# HELLINGER DISTANCE
print("\n" *5)
print("HELLINGER DISTANCE")

def hellinger_np(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

svhn_data = np.array(svhn_dataset.data)
mnist_data = np.array(mnist_dataset.data)
if PRINTOUT:
    print("SVHN data shape:", svhn_data.shape)
    print("MNIST data shape:", mnist_data.shape)

svhn_gray = np.mean(svhn_data, axis=3)
svhn_resized = np.array([
    np.array(transforms.functional.resize(
        transforms.functional.to_pil_image(img.astype(np.uint8)), 
        (28, 28)
    )) for img in svhn_gray
])

svhn_flat = svhn_resized.reshape(len(svhn_resized), -1)
mnist_flat = mnist_data.reshape(len(mnist_data), -1)
if PRINTOUT:
    print("SVHN flat shape:", svhn_flat.shape)
    print("MNIST flat shape:", mnist_flat.shape)

svhn_mean = svhn_flat.mean(axis=0)
mnist_mean = mnist_flat.mean(axis=0)

svhn_prob = svhn_mean / np.sum(svhn_mean)
mnist_prob = mnist_mean / np.sum(mnist_mean)

print("Hellinger Distance between SVHN and MNIST:", hellinger_np(svhn_prob, mnist_prob))




# JENSEN-SHANNON DIVERGENCE
print("\n" *5)
print("JENSEN-SHANNON DIVERGENCE")

def KL_numpy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    epsilon = 1e-10
    b = np.where(b == 0, epsilon, b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

mnist_dataset_new = MNIST(root='data', train=True, download=True)
svhn_dataset_new = SVHN(root='./data', split="train", download=True)


data_np_mnist = mnist_dataset_new.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  # Add channel dimension
if PRINTOUT:
    print("MNIST data shap after adding channel dimention:", data_np_mnist.shape)
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
data_np_mnist = mnist_tensor[:50000].numpy()

data_np_svhn = svhn_dataset_new.data.astype(np.float32) / 255.0
data_np_svhn = data_np_svhn.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
data_np_svhn = data_np_svhn[:50000]

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

datasets = {
    'MNIST': data_np_mnist,
    'SVHN': data_np_svhn
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
            print(f"{name1} vs {name2}: {js:.6f}")



# L2 NORM + COSINE SIMILARITY
print("\n"*5)
print("L2 NORM + COSINE SIMILARITY")

from torchvision.models import wide_resnet50_2

gradient_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resnet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mnist_to_rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"
gradient_model = wide_resnet50_2(weights=True).to(device)

svhn_single = DataLoader(
    SVHN(root='./data', split="train", download=True, transform=gradient_transform),
    batch_size=1, shuffle=True
)

mnist_single = DataLoader(
    MNIST(root='data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True
)

def get_gradient_vector(model, dataloader, is_mnist=False):
    model.eval()
    model.zero_grad()
    x, y = next(iter(dataloader))
    
    if is_mnist:
        x = x.repeat(1, 3, 1, 1)  # Grayscale to RGB
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

grad_svhn = get_gradient_vector(gradient_model, svhn_single)
grad_mnist = get_gradient_vector(gradient_model, mnist_single, is_mnist=True)

l2_diff = torch.norm(grad_svhn - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_svhn.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()

print(f"L2 norm of gradient difference: {l2_diff:.4f}")
print(f"Cosine similarity between gradients: {cos_sim:.4f}")



# MAXIMUM MEAN DISCREPANCY (MMD)

print("\n"*5)
print("MAXIMUM MEAN DISCREPANCY")

from discrepancies import MaximumMeanDiscrepancy
from kernels import LaplacianKernel

mnist_mmd = MNIST(root='data', train=True, download=True)
svhn_mmd = SVHN(root='./data', split="train", download=True)

data_np_mnist_mmd = mnist_mmd.data[:5000].numpy().astype(np.float32) / 255.0 
data_np_mnist_mmd = data_np_mnist_mmd[:, None, :, :]
mnist_tensor_mmd = torch.from_numpy(data_np_mnist_mmd).float()
mnist_tensor_mmd = F.interpolate(mnist_tensor_mmd, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor_mmd = mnist_tensor_mmd.repeat(1, 3, 1, 1)
data_np_mnist_mmd = mnist_tensor_mmd.detach().cpu().numpy()

data_np_svhn_mmd = svhn_mmd.data[:5000].astype(np.float32) / 255.0  

mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(1e-2))
print("MMD between MNIST and SVHN:", mmd.compute(data_np_svhn_mmd, data_np_mnist_mmd))



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
    for i in range(0, min(n, 1000), batch_size): 
        batch = data_tensor[i:i+batch_size]
        batch = prepare_images_for_fid(batch)
        fid.update(batch, is_real=is_real)

svhn_tensor = torch.from_numpy(svhn_data).float()
# if len(svhn_tensor.shape) == 4:  # (N, H, W, C) -> (N, C, H, W)
#     svhn_tensor = svhn_tensor.permute(0, 3, 1, 2)

mnist_tensor = torch.from_numpy(mnist_data).unsqueeze(1).float()

update_fid_safe(fid, svhn_tensor, is_real=True)
print("Updated FID with SVHN.")
update_fid_safe(fid, mnist_tensor, is_real=False)
print("Updated FID with MNIST.")

print("FID between SVHN and MNIST:", fid.compute().item())

if PRINTOUT:
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)