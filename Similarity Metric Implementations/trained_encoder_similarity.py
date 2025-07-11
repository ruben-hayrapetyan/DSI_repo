from torchvision.datasets import CIFAR10, MNIST, USPS, FashionMNIST, SVHN, KMNIST, QMNIST, SEMEION, STL10
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32, 32)),            
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),                
])

cifar = CIFAR10(root='data', train=True, download=True)
data_np_cifar = cifar.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
data_np_cifar = np.transpose(data_np_cifar, (0, 3, 1, 2))  # (50000, 3, 32, 32)

mnist = MNIST(root='data', train=True, download=True)
data_np_mnist = mnist.data[:50000].numpy().astype(np.float32) / 255.0  # (50000, 28, 28)
data_np_mnist = data_np_mnist[:, None, :, :]  # (50000, 1, 28, 28)
mnist_tensor = torch.from_numpy(data_np_mnist).float()
mnist_tensor = F.interpolate(mnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
mnist_tensor = mnist_tensor.repeat(1, 3, 1, 1)  # (50000, 3, 32, 32)
data_np_mnist = mnist_tensor.detach().cpu().numpy()

usps = USPS(root='data', train=True, download=True)
data_np_usps = usps.data.astype(np.float32) / 255.0
data_np_usps = data_np_usps[:, None, :, :]  
usps_tensor = torch.from_numpy(data_np_usps)
usps_tensor = F.interpolate(usps_tensor, size=(32, 32), mode='bilinear', align_corners=False)
usps_tensor = usps_tensor.repeat(1, 3, 1, 1)
data_np_usps = usps_tensor.numpy()

fm = FashionMNIST(root='data', train=True, download=True)
data_np_fm = fm.data.numpy().astype(np.float32) / 255.0
data_np_fm = data_np_fm[:, None, :, :]  
fm_tensor = torch.from_numpy(data_np_fm)
fm_tensor = F.interpolate(fm_tensor, size=(32, 32), mode='bilinear', align_corners=False)
fm_tensor = fm_tensor.repeat(1, 3, 1, 1)
data_np_fm = fm_tensor[:50000].numpy()

svhn = SVHN(root='./data', split="train", download=True)
data_np_svhn = svhn.data.astype(np.float32) / 255.0

kmnist = KMNIST(root='./data', train=True, download=True)
data_np_kmnist = kmnist.data.numpy().astype(np.float32) / 255.0
data_np_kmnist = data_np_kmnist[:, None, :, :]  
kmnist_tensor = torch.from_numpy(data_np_kmnist)
kmnist_tensor = F.interpolate(kmnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
kmnist_tensor = kmnist_tensor.repeat(1, 3, 1, 1)
data_np_kmnist = kmnist_tensor[:50000].numpy()

qmnist = QMNIST(root='./data', train=True, download=True)
data_np_qmnist = qmnist.data.numpy().astype(np.float32) / 255.0
data_np_qmnist = data_np_qmnist[:, None, :, :]  
qmnist_tensor = torch.from_numpy(data_np_qmnist)
qmnist_tensor = F.interpolate(qmnist_tensor, size=(32, 32), mode='bilinear', align_corners=False)
qmnist_tensor = qmnist_tensor.repeat(1, 3, 1, 1)
data_np_qmnist = qmnist_tensor[:50000].numpy()

stl10 = STL10(root='./data', split="train", download=True)
data_np_stl10 = stl10.data.astype(np.float32) / 255.0
stl10_tensor = torch.from_numpy(data_np_stl10[:50000]).float()
stl10_tensor = F.interpolate(stl10_tensor, size=(32, 32), mode='bilinear', align_corners=False)
data_np_stl10 = stl10_tensor.detach().cpu().numpy()

semeion  = SEMEION(root='./data', transform=transform, download=True)
data_np_semeion = semeion.data.astype(np.float32).reshape(-1, 1, 16, 16)
semeion_tensor = torch.from_numpy(data_np_semeion)
semeion_tensor = F.interpolate(semeion_tensor, size=(32, 32), mode='bilinear', align_corners=False)
semeion_tensor = semeion_tensor.repeat(1, 3, 1, 1)
data_np_semeion = semeion_tensor[:50000].numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__ (self, latent_dimension):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dimension)
        )

    def forward(self, x):
        return self.encoder(x)
    
class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
    
train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

encoder = Encoder(128).to(device)
model = Classifier(encoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

mnist_in = torch.from_numpy(data_np_mnist).to(device)
mnist_for_usps_in = torch.from_numpy(data_np_mnist[:7291]).to(device)
mnist_for_stl10_in = torch.from_numpy(data_np_mnist[:5000]).to(device)
mnist_for_semeion_in = torch.from_numpy(data_np_mnist[:1593]).to(device)
cifar_in = torch.from_numpy(data_np_cifar).to(device)
fmnist_in = torch.from_numpy(data_np_fm).to(device)
usps_in = torch.from_numpy(data_np_usps).to(device)
svhn_in = torch.from_numpy(data_np_svhn[:50000]).to(device)
kmnist_in = torch.from_numpy(data_np_kmnist).to(device)
qmnist_in = torch.from_numpy(data_np_qmnist).to(device)
stl10_in = torch.from_numpy(data_np_stl10[:50000]).to(device)
semeion_in = torch.from_numpy(data_np_semeion[:50000]).to(device)

with torch.no_grad():
    mnist_rep = encoder(mnist_in)
    mnist_for_usps_rep = encoder(mnist_for_usps_in)
    mnist_for_stl10_rep = encoder(mnist_for_stl10_in)
    mnist_for_semeion_rep = encoder(mnist_for_semeion_in)
    cifar_rep = encoder(cifar_in)
    fmnist_rep = encoder(fmnist_in)
    usps_rep = encoder(usps_in)
    svhn_rep = encoder(svhn_in)
    kmnist_rep = encoder(kmnist_in)
    qmnist_rep = encoder(qmnist_in)
    stl10_rep = encoder(stl10_in)
    semeion_rep = encoder(semeion_in)

mnist_rep_t = mnist_rep.T
mnist_for_usps_rep_t = mnist_for_usps_rep.T
mnist_for_stl10_rep_t = mnist_for_stl10_rep.T
mnist_for_semeion_rep_t = mnist_for_semeion_rep.T
cifar_rep_t = cifar_rep.T
fmnist_rep_t = fmnist_rep.T
usps_rep_t = usps_rep.T
svhn_rep_t = svhn_rep.T
kmnist_rep_t = kmnist_rep.T
qmnist_rep_t = qmnist_rep.T
stl10_rep_t = stl10_rep.T
semeion_rep_t = semeion_rep.T

def similarity_score(d_1, d_2):
    d_2 = torch.linalg.pinv(d_2)
    T = d_1 @ d_2
    I = torch.eye(T.shape[0], device=device)
    #score = torch.norm(T - I).item() #frobenius norm
    #print(T)
    #return normalized
    #trace = torch.trace(T) # trace
    #return trace/T.shape[0]
    score = torch.linalg.matrix_norm(T - I, ord=-1)
    normalized = 1 / (1 + score)
    return normalized

print("Similarity MNIST-MNIST:", similarity_score(mnist_rep_t, mnist_rep_t))
print("Similarity MNIST-USPS:", similarity_score(usps_rep_t, mnist_for_usps_rep_t))
print("Similarity MNIST-FashionMNIST:", similarity_score(fmnist_rep_t, mnist_rep_t))
print("Similarity MNIST-SVHN:", similarity_score(svhn_rep_t, mnist_rep_t))
print("Similarity MNIST-CIFAR-10:", similarity_score(cifar_rep_t, mnist_rep_t))
print("Similarity MNIST-STL-10:", similarity_score(stl10_rep_t, mnist_for_stl10_rep_t))
print("Similarity MNIST-QMNIST:", similarity_score(qmnist_rep_t, mnist_rep_t))
print("Similarity MNIST-KMNIST:", similarity_score(kmnist_rep_t, mnist_rep_t))
print("Similarity MNIST-SEMEION:", similarity_score(semeion_rep_t, mnist_for_semeion_rep_t))