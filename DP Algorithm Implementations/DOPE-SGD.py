import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import *
import numpy as np

def DOPESGD(model, D_loader, D_s_loader, lr, sigma, C, l, T):
    """
    Parameters:
        D_loader: private training data loader
        n: private batch size
        D_s_loader: public training data loader
        n_s: public batch size
        lr: learning rate
        sigma: noise scale
        C: gradient norm clip
        l: loss function
        T: number of training iterations
    """
    
    optim = torch.optim.SGD(model.parameters(), lr)
    model.train()
    pub_iter = iter(D_s_loader)

    for epoch in range(T):
        
        total_loss = 0.0
        correct = 0
        total = 0

        for priv_inputs, priv_labels in D_loader:
            
            try:
                pub_inputs, pub_labels = next(pub_iter)
            except StopIteration:
                pub_iter = iter(D_s_loader)
                pub_inputs, pub_labels = next(pub_iter)
            
            optim.zero_grad()
            pub_out = model(pub_inputs)
            pub_loss = l(pub_out, pub_labels)
            pub_loss.backward()
            pub_grad = []
            for p in model.parameters():
                pub_grad.append(p.grad.detach().clone())
            pub_grad = [g.clone() for g in pub_grad]

            grads = [torch.zeros_like(p) for p in model.parameters()]

            for x, y in zip(priv_inputs, priv_labels):
                optim.zero_grad()
                output = model(x.unsqueeze(0))
                loss = l(output, y.unsqueeze(0))
                loss.backward()

                for i, p in enumerate(model.parameters()):
                    if p.requires_grad and p.grad is not None:
                        grads[i] += pub_grad[i] + ((p.grad - pub_grad[i]) * C) / (torch.max(torch.tensor(C), torch.norm(p.grad - pub_grad[i])))

                for i in range(len(grads)):    
                    grads[i] += torch.normal(0, sigma * C, grads[i].shape)

            for p, g in zip(model.parameters(), grads):
                if p.requires_grad:
                    p.grad = g / len(priv_inputs)
            optim.step()

            total_loss += pub_loss.item() * priv_inputs.size(0)
            with torch.no_grad():
                priv_out = model(priv_inputs)
                _, predicted = torch.max(priv_out.data, 1)
                total += priv_labels.size(0)
                correct += (predicted == priv_labels).sum().item()
        
        epoch_loss = total_loss / total
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{T}] ... Loss: {epoch_loss:.4f} ... Training Accuracy: {epoch_accuracy:.2f}%")

transform = transforms.Compose([
    transforms.ToTensor(),  
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_size_private = int(0.96 * len(train_dataset))
train_size_public = len(train_dataset) - train_size_private
train_dataset_public, train_dataset_private = random_split(train_dataset, [train_size_public, train_size_private])
train_loader_public = DataLoader(dataset=train_dataset_public, batch_size=64, shuffle=True)
train_loader_private = DataLoader(dataset=train_dataset_private, batch_size=64, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        x = x.view(-1, 120)
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN()
loss = nn.CrossEntropyLoss()

DOPESGD(
    model=model,
    D_loader=train_loader_private,
    D_s_loader=train_loader_public,
    lr=0.01,           
    sigma=1.0, 
    C=1.0, 
    l=loss,
    T=10,
)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

evaluate(model, test_loader)