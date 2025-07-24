import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, STL10
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

stl_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Grayscale(),
])

mnist_dataset = MNIST(root='./data', train=True, transform=mnist_transform, download=True)
stl_dataset  = STL10(root='./data', split="train", transform=stl_transform, download=True)

max_mnist_contribution = len(mnist_dataset) 
max_stl_contribution = len(stl_dataset) 
total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
total_if_stl_maxed = int(len(stl_dataset) / 0.96)
total_desired_size = min(total_if_mnist_maxed, total_if_stl_maxed)
mnist_size = int(0.04 * total_desired_size)
stl_size = int(0.96 * total_desired_size)

mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
stl_indices = np.random.choice(len(stl_dataset), size=stl_size, replace=False)
mnist_subset = Subset(mnist_dataset, mnist_indices)
stl_subset = Subset(stl_dataset, stl_indices)

train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=stl_subset, batch_size=64, shuffle=True)

test_dataset = STL10(root='./data', split="test", transform=stl_transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

if PRINTOUT:
    print(f"\nDataset Information:")
    print(f"Original MNIST training set: {len(mnist_dataset)} samples")
    print(f"Original STL10 training set: {len(stl_dataset)} samples")
    print(f"Modified MNIST loader: {len(mnist_subset)} samples ({len(mnist_subset)/(len(mnist_subset)+len(stl_subset))*100:.1f}%)")
    print(f"Modified STL10 loader: {len(stl_subset)} samples ({len(stl_subset)/(len(mnist_subset)+len(stl_subset))*100:.1f}%)")
    print(f"Test set: {len(test_dataset)} samples")

    print(f"\nData Loader Batch Information:")
    for batch_idx, (data, target) in enumerate(train_loader_public):
        print(f"MNIST batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
        break

    for batch_idx, (data, target) in enumerate(train_loader_private):
        print(f"STL batch {batch_idx + 1}: {data.shape}, labels: {target.shape}")
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



#SLICED WASSERSTEIN DISTANCE (SWD)
print("\n"*5)
print("SLICED WASSERSTEIN DISTANCE")

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

stl_features = feature_extractor(train_loader_private)
print("Extracted STL10 features:", stl_features.shape)

mnist_features = feature_extractor(train_loader_public)
print("Extracted MNIST features:", mnist_features.shape)

print("Sliced Wasserstein Distance between MNIST and STL10:", sliced_wasserstein_distance(mnist_features, stl_features))




# HELLINGER DISTANCE (HD)
print("\n"*5)
print("HELLINGER DISTANCE")

def hellinger_np(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

stl_data = np.array(stl_dataset.data) # (5000, 96, 96, 3)
mnist_data = np.array(mnist_dataset.data) # (60000, 28, 28)

stl_gray = np.mean(stl_data, axis=3)
stl_resized = np.array([
    np.array(transforms.functional.resize(
        transforms.functional.to_pil_image(img.astype(np.uint8)), 
        (28, 28)
    )) for img in stl_gray
])

stl_flat = stl_resized.reshape(len(stl_resized), -1) #(5000, 784)
mnist_flat = mnist_data.reshape(len(mnist_data), -1) #(60000, 784)

stl_mean = stl_flat.mean(axis=0)
mnist_mean = mnist_flat.mean(axis=0)

stl_prtob = stl_mean / np.sum(stl_mean)
mnist_prtob = mnist_mean / np.sum(mnist_mean)

print("Hellinger Distance between MNIST and STL10:", hellinger_np(mnist_prtob, stl_prtob))


# JENSEN-SHANNON DIVERGENCE (JSD)
print("\n"*5)
print("JENSEN-SHANNON DIVERGENCE")

def KL_numpy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    epsilon = 1e-10
    b = np.where(b == 0, epsilon, b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

mnist_fresh = MNIST(root='data', train=True, download=True)
stl_fresh = STL10(root='./data', split="train", download=True)

data_np_mnist = mnist_fresh.data.numpy().astype(np.float32) / 255.0
data_np_mnist = data_np_mnist[:, None, :, :]  # Add channel dimension
mnist_tensor = torch.from_numpy(data_np_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1) 
data_np_mnist = mnist_tensor[:5000].numpy()  

# CHANGED: STL10 doesn't have .data attribute, need to extract manually
stl_data_list = []
for i in range(min(5000, len(stl_fresh))):
    img, _ = stl_fresh[i]
    # Convert tensor to numpy array
    img_np = np.array(img)  # CHANGED: direct conversion instead of permute
    if len(img_np.shape) == 3 and img_np.shape[0] == 3:  # If (C, H, W)
        img_np = img_np.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img_np = (img_np * 255).astype(np.uint8)  # Convert back to 0-255 range
    stl_data_list.append(img_np)
data_np_stl = np.array(stl_data_list).astype(np.float32) / 255.0
data_np_stl = data_np_stl.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

# CHANGED: Resize STL10 to match MNIST dimensions (32x32)
stl_tensor = torch.from_numpy(data_np_stl)
stl_tensor = F.interpolate(stl_tensor, size=(32, 32), mode='bilinear', align_corners=False)
data_np_stl = stl_tensor.numpy()

def to_prob_dist(data):
    data_flat = data.reshape(data.shape[0], -1)
    data_flat = data_flat + 1e-10
    return data_flat / data_flat.sum(axis=1, keepdims=True)

datasets = {
    'MNIST': data_np_mnist,
    'STL10': data_np_stl
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mnist_to_rgb_tranform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"
gradient_model = wide_resnet50_2(weights=True).to(device)

stl_single = DataLoader(
    STL10(root='./data', split="train", transform=gradient_transform, download=True),
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

grad_stl = get_gradient_vector(gradient_model, stl_single, is_mnist=False)
grad_mnist = get_gradient_vector(gradient_model, mnist_single, is_mnist=True)

l2_diff = torch.norm(grad_stl - grad_mnist).item()
cos_sim = torch.nn.functional.cosine_similarity(
    grad_stl.unsqueeze(0), grad_mnist.unsqueeze(0)
).item()

print(f"L2 norm of gradient difference: {l2_diff:.4f}")
print(f"Cosine similarity between gradients: {cos_sim:.4f}")



# MAXIMUM MEAN DISCREPANCY (MMD)
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST, STL10
from discrepancies import MaximumMeanDiscrepancy
from kernels import LaplacianKernel

print("\n" * 5)
print("MAXIMUM MEAN DISCREPANCY (MMD)")

mnist = MNIST(root='data', train=True, download=True)
data_mnist = mnist.data[:2000].numpy().astype(np.float32) / 255.0  # (5000, 28, 28)
data_mnist = data_mnist[:, None, :, :]
mnist_tensor = torch.from_numpy(data_mnist)
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)
mnist_np = mnist_tensor.numpy().reshape(2000, -1)  # Flatten to (5000, 3072)

stl = STL10(root='data', split='train', download=True)
data_stl = stl.data[:2000].astype(np.float32) / 255.0  # (5000, 96, 96, 3)
stl_tensor = torch.from_numpy(data_stl).float()
stl_tensor = F.interpolate(stl_tensor, size=(32, 32), mode='bilinear', align_corners=False)
stl_np = stl_tensor.numpy().reshape(2000, -1)  # Flatten to (5000, 3072)

mmd = MaximumMeanDiscrepancy(kernel=LaplacianKernel(1e-2))
print("MMD between MNIST and STL10:", mmd.compute(stl_np, mnist_np))







# FRECHET INCEPTION DISTANCE (FID)
print("\n"*5)
print("FRECHET INCEPTION DISTANCE (FID)")

from torcheval.metrics import FrechetInceptionDistance

def prepare_images_for_fid(x):
    """Prepare images for FID computation"""
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

stl_tensor = torch.from_numpy(stl_data).float()
# if len(stl_tensor.shape) == 4:  # (N, H, W, C) -> (N, C, H, W)
#     stl_tensor = stl_tensor.permute(0, 3, 1, 2)

mnist_tensor = torch.from_numpy(mnist_data).unsqueeze(1).float()
update_fid_safe(fid, stl_tensor, is_real=True)
print("Updated FID with STL10.")
update_fid_safe(fid, mnist_tensor, is_real=False)
print("Updated FID with MNIST.")

print("FID between STL10 and MNIST:", fid.compute().item())

if PRINTOUT:
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)