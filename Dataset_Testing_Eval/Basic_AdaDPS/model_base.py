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


class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)

        """train_data = np.load('data/imdb_10000d_train.npz')
        test_data = np.load('data/imdb_10000d_test.npz')
        train_data = np.load('data/imdb_10000d_train_bow.npz')
        test_data = np.load('data/imdb_10000d_test_bow.npz')
        train_data = dict(train_data)
        test_data = dict(test_data)
        x_train, y_train = train_data['x'], train_data['y']
        perm = np.random.permutation(len(y_train))
        x_train, y_train = x_train[perm], y_train[perm]  # randomly shuffle
        x_public, y_public = x_train[:250], y_train[:250]
        x_train, y_train = x_train[250:], y_train[250:]
        x_test, y_test = test_data['x'], test_data['y']
        print('x train: ', x_train.shape)  # (25000, 10000)
        print('x test: ', x_test.shape)"""

        transform = transforms.Compose([
            transforms.ToTensor(),  
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_size_public = int(0.04 * len(train_dataset))
        train_public_subset = Subset(train_dataset, range(train_size_public))
        the_train_loader_public = DataLoader(dataset=train_public_subset, batch_size=self.public_bs, shuffle=True)
        self.public_loader = the_train_loader_public

        (train_x_full, train_y_full), (x_test, y_test) = generate_mnist()
        x_public, y_public = train_x_full[:train_size_public], train_y_full[:train_size_public]
        self.public_x = torch.FloatTensor(x_public)
        self.public_y = torch.LongTensor(y_public)

        if self.dataset == "USPS":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])
            train_dataset_2  = datasets.USPS(root='./data', train=True, transform=transform, download=True)
            train_size_private = int(0.96 * len(train_dataset_2))
            train_private_subset = Subset(train_dataset_2, range(train_size_private))
            train_loader_private  = DataLoader(dataset=train_private_subset, batch_size=self.batch_size, shuffle=False)
            test_dataset  = datasets.USPS(root='./data', train=False, transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.dataset == "FashionMNIST":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])
            train_dataset_2  = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
            train_size_private = int(0.96 * len(train_dataset_2))
            train_private_subset = Subset(train_dataset_2, range(train_size_private))
            train_loader_private  = DataLoader(dataset=train_private_subset, batch_size=self.batch_size, shuffle=False)
            test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.dataset == "SVHN":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Grayscale()
            ])
            train_dataset_2  = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
            train_size_private = int(0.96 * len(train_dataset_2))
            train_private_subset = Subset(train_dataset_2, range(train_size_private))
            train_loader_private  = DataLoader(dataset=train_private_subset, batch_size=self.batch_size, shuffle=False)
            test_dataset  = datasets.SVHN(root='./data', split='test', transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.dataset == "CIFAR10":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Grayscale()
            ])
            train_dataset_2  = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
            train_size_private = int(0.96 * len(train_dataset_2))
            train_private_subset = Subset(train_dataset_2, range(train_size_private))
            train_loader_private  = DataLoader(dataset=train_private_subset, batch_size=self.batch_size, shuffle=False)
            test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.dataset == "STL10":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Grayscale()
            ])
            train_dataset_2  = datasets.STL10(root='./data', split='train', transform=transform, download=True)
            train_size_private = int(0.96 * len(train_dataset_2))
            train_private_subset = Subset(train_dataset_2, range(train_size_private))
            train_loader_private  = DataLoader(dataset=train_private_subset, batch_size=self.batch_size, shuffle=False)
            test_dataset  = datasets.STL10(root='./data', split='test', transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.train_loader = train_loader_private
            self.test_loader = test_loader

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