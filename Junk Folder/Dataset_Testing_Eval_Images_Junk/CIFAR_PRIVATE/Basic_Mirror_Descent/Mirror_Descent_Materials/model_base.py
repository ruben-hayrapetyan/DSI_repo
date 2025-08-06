import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import copy
from torchvision import datasets, transforms

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.shortcut = nn.Identity() if self.equalInOut else nn.Conv2d(
            in_planes, out_planes, 1, stride=stride, bias=False)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        return out + self.shortcut(x)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes,
                                stride if i == 0 else 1, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropRate=0.0, in_ch=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(in_ch, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

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

        if self.public_dataset == "USPS":
            public_dataset = datasets.USPS(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "FashionMNIST":
            public_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "SVHN":
            public_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "MNIST":
            public_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "STL-10":
            public_dataset = datasets.STL10(root='./data', split='train', transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "QMNIST":
            public_dataset = datasets.QMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "KMNIST":
            public_dataset = datasets.KMNIST(root='./data', train=True, transform=transform, download=True)

            np.random.seed(42)
            public_indices = np.random.choice(len(public_dataset), size=2000, replace=False)
            public_subset = Subset(public_dataset, public_indices)

            self.public_loader = DataLoader(dataset=public_subset, batch_size=64, shuffle=True)
        if self.public_dataset == "SEMEION":
            public_dataset = datasets.SEMEION(root='./data', transform=transform, download=True)

            self.public_loader = DataLoader(dataset=public_dataset, batch_size=64, shuffle=True)

        # #train_data = np.load('data/imdb_10000d_train.npz')
        # #test_data = np.load('data/imdb_10000d_test.npz')
        # train_data = np.load('data/imdb_10000d_train_bow.npz')
        # test_data = np.load('data/imdb_10000d_test_bow.npz')
        # train_data = dict(train_data)
        # test_data = dict(test_data)
        # x_train, y_train = train_data['x'], train_data['y']
        # perm = np.random.permutation(len(y_train))
        # x_train, y_train = x_train[perm], y_train[perm]  # randomly shuffle
        # x_public, y_public = x_train[:250], y_train[:250]
        # x_train, y_train = x_train[250:], y_train[250:]
        # x_test, y_test = test_data['x'], test_data['y']
        # print('x train: ', x_train.shape)  # (25000, 10000)
        # print('x test: ', x_test.shape)

        # self.public_dataset = TensorDataset(torch.FloatTensor(x_public), torch.LongTensor(y_public))
        # self.train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
        # self.test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))

        # self.public_x = torch.FloatTensor(x_public)
        # self.public_y = torch.LongTensor(y_public)

        # self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset,
        #                                                  batch_size=self.public_bs,
        #                                                  num_workers=4,
        #                                                  shuffle=True)

        # self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
        #                                                 batch_size=self.batch_size,
        #                                                 num_workers=4,
        #                                                 shuffle=True)

        # self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
        #                                                batch_size=self.batch_size,
        #                                                num_workers=4,
        #                                                shuffle=False)

        #self.model = LSTM(max_words=10000, emb_size=64, hid_size=64)  # hard-coding a bit
        self.model = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3, in_ch=1)
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

        # x_pub = self.public_x.to(self.device)
        # labels_pub = self.public_y.to(self.device)
        x_pub, labels_pub = next(iter(self.public_loader))
        x_pub = x_pub.to(self.device)
        labels_pub = labels_pub.to(self.device)
        
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