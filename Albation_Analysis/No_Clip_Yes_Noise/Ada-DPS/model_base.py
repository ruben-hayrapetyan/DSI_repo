import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import copy
from generate_data import generate_mnist


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
        self.fc1 = nn.Linear(480, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = x.view(-1, 480)
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x


class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)

        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((32, 32)),               
            transforms.ToTensor(),                       
        ])

        cifar_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        self.train_loader = DataLoader(dataset=cifar_dataset, batch_size=64, shuffle=True)

        cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        self.test_loader = DataLoader(dataset=cifar_test_dataset, batch_size=1000, shuffle=False)

        if self.dataset == "USPS":
            public_dataset = datasets.USPS(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "FashionMNIST":
            public_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "SVHN":
            public_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "MNIST":
            public_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "STL-10":
            public_dataset = datasets.STL10(root='./data', split='train', transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "QMNIST":
            public_dataset = datasets.QMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "KMNIST":
            public_dataset = datasets.KMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.dataset == "SEMEION":
            public_dataset = datasets.SEMEION(root='./data', transform=transform, download=True)

            self.public_loader = DataLoader(dataset=public_dataset, batch_size=64, shuffle=True)

        self.model = CNN()  # hard-coding a bit
        #self.model = nn.Linear(784, 10)
        print(self.model)
        self.loss = nn.CrossEntropyLoss()
        self.loss_flat = nn.CrossEntropyLoss(reduction='none')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def estimate_preconditioner(self):
        
        tmp_mean = dict()
        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = torch.zeros_like(p)
            tmp_mean[p_name] = torch.zeros_like(p)


        for i, (x_pub, labels_pub) in enumerate(self.public_loader):
            x_pub = x_pub.to(self.device)
            labels_pub = labels_pub.to(self.device)
            predicted = self.model(x_pub)
            l = self.loss(predicted, labels_pub)
            self.model.zero_grad()
            l.backward()
            for p_name, p in self.model.named_parameters():
                self.preconditioner[p_name] += p.grad ** 2
                tmp_mean[p_name] += p.grad

        # self.mean is optional
        for p_name, p in self.model.named_parameters():  # some approximation (take the last iterate)
            self.mean[p_name] = 0.9 * self.mean[p_name] + 0.1 * (tmp_mean[p_name] / (i+1))

        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = self.preconditioner[p_name]/(i+1)



    def get_pub_gradient(self):
        
        pub_g = dict()

        for p_name, p in self.model.named_parameters():
            pub_g[p_name] = torch.zeros_like(p)

        x_pub = self.public_x.to(self.device)
        labels_pub = self.public_y.to(self.device)
        predicted = self.model(x_pub)
        l = self.loss(predicted, labels_pub)
        self.model.zero_grad()
        l.backward()
        for p_name, p in self.model.named_parameters():
            pub_g[p_name] = copy.deepcopy(p.grad)
        return pub_g

    def loss_flat_reg(self, predicted, labels):
        loss_vector = self.loss_flat(predicted, labels)
        l2_reg = None
        for p in self.model.parameters():
            if l2_reg is None:
                l2_reg = 0.5 * p.norm(2) ** 2
            else:
                l2_reg = l2_reg + 0.5 * p.norm(2) ** 2
        return loss_vector + l2_reg

    def get_bow_frequency(self):
        freq = np.zeros(10000)
        idx = 0

        train_data = np.load('data/imdb_10000d_train.npz')
        for sample in train_data['x']:  # sample: a vector containing word indices
            for word_i in sample:
                freq[word_i] += 1
            idx += 1
        freq = (freq+0.1) / idx
        return freq + 1e-10

    def get_tf_idf_value(self):
        tf_idf = np.zeros(10000)
        train_data = np.load('data/imdb_10000d_train_tf-idf.npz')
        idx = 0
        for sample in train_data['x']:
            tf_idf += sample
            idx += 1
        return tf_idf + 1e-10

    def get_loss_and_gradients(self, input, labels):

        predicted = self.model(input)
        l = self.loss(predicted, labels)
        self.optimizer.zero_grad()
        l.backward()
        g = []
        for x in self.model.parameters():
            g.append(x.grad)
        return l.item(), g

    def apply_gradients(self, grads):

        for i, x in enumerate(self.model.parameters()):
            x.grad.data = grads[i]
        self.optimizer.step()

    def get_gradient_norm(self):

        total_norm = 0
        for p in self.model.parameters():
            total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_weight_norm(self):

        total_norm = 0

        for p in self.model.parameters():
            total_norm += np.linalg.norm(p.cpu().detach().numpy(), 2) ** 2

        total_norm = total_norm ** 0.5
        return total_norm

    def get_test_accuracy(self):

        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, labels in self.test_loader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().data
        return correct * 1.0 / total

    def get_train_accuracy_and_loss(self):

        self.model.eval()
        with torch.no_grad():  # not training
            correct = 0
            loss = 0
            total = 0
            for i, (x, labels) in enumerate(self.train_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().data
                l = self.loss(outputs, labels)
                loss += l.item()
        return correct * 1.0 / total, loss / i