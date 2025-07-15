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



class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
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
        x = x.view(-1, 120)
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x

class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)


        """transform = transforms.Compose([
            transforms.ToTensor(),  
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_size_public = int(0.04 * len(train_dataset))
        train_public_subset = Subset(train_dataset, range(train_size_public))
        train_loader_public = DataLoader(dataset=train_public_subset, batch_size=64, shuffle=True)
        self.public_loader = train_loader_public"""

        if self.public_dataset == "USPS":
            transform_mnist = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_usps = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])

            usps_dataset  = datasets.USPS(root='./data', train=True, transform=transform_usps, download=True)
            usps_loader  = DataLoader(dataset=usps_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_usps_contribution = len(usps_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_usps_maxed = int(len(usps_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_usps_maxed)
            mnist_size = int(0.96 * total_desired_size)
            usps_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            usps_indices = np.random.choice(len(usps_dataset), size=usps_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            usps_subset = Subset(usps_dataset, usps_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=usps_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader 
        if self.public_dataset == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            # mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=self.public_bs, shuffle=True)

            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])

            mnist2_dataset  = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            # mnist2_loader  = DataLoader(dataset=mnist2_dataset, batch_size=1000, shuffle=False)

            # max_mnist_contribution = len(mnist_dataset) 
            # max_mnist2_contribution = len(mnist2_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.04)
            total_if_mnist2_maxed = int(len(mnist2_dataset) / 0.96)
            total_desired_size = min(total_if_mnist_maxed, total_if_mnist2_maxed)
            mnist_size = int(0.04 * total_desired_size)
            mnist2_size = int(0.96 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            mnist2_indices = np.random.choice(len(mnist2_dataset), size=mnist2_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            mnist2_subset = Subset(mnist2_dataset, mnist2_indices)
            train_loader_public = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_private = DataLoader(dataset=mnist2_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader 
        if self.public_dataset == "FashionMNIST":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_fashion = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((28, 28)),               
                transforms.ToTensor(),                       
            ])


            fm_dataset  = datasets.FashionMNIST(root='./data', train=True, transform=transform_fashion, download=True)
            fm_loader  = DataLoader(dataset=fm_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_fm_contribution = len(fm_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_fm_maxed = int(len(fm_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_fm_maxed)
            mnist_size = int(0.96 * total_desired_size)
            fm_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            fm_indices = np.random.choice(len(fm_dataset), size=fm_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            fm_subset = Subset(fm_dataset, fm_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=fm_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.public_dataset == "SVHN":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_svhn = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ])

            svhn_dataset  = datasets.SVHN(root='./data', split="train", transform=transform_svhn, download=True)
            svhn_loader  = DataLoader(dataset=svhn_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_svhn_contribution = len(svhn_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_svhn_maxed = int(len(svhn_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_svhn_maxed)
            mnist_size = int(0.96 * total_desired_size)
            svhn_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            svhn_indices = np.random.choice(len(svhn_dataset), size=svhn_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            svhn_subset = Subset(svhn_dataset, svhn_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=svhn_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.public_dataset == "CIFAR10":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_cifar = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((28, 28)),               
                transforms.ToTensor(),                       
            ])


            CIFAR_dataset  = datasets.CIFAR10(root='./data', train=True, transform=transform_cifar, download=True)
            CIFAR_loader  = DataLoader(dataset=CIFAR_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_CIFAR_contribution = len(CIFAR_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_CIFAR_maxed = int(len(CIFAR_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_CIFAR_maxed)
            mnist_size = int(0.96 * total_desired_size)
            CIFAR_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            CIFAR_indices = np.random.choice(len(CIFAR_dataset), size=CIFAR_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            CIFAR_subset = Subset(CIFAR_dataset, CIFAR_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=CIFAR_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader 
        if self.public_dataset == "STL10":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_stl10 = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ])

            stl_dataset  = datasets.STL10(root='./data', split="train", transform=transform_stl10, download=True)
            stl_loader  = DataLoader(dataset=stl_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_stl_contribution = len(stl_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_stl_maxed = int(len(stl_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_stl_maxed)
            mnist_size = int(0.96 * total_desired_size)
            stl_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            stl_indices = np.random.choice(len(stl_dataset), size=stl_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            stl_subset = Subset(stl_dataset, stl_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=stl_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.public_dataset == "QMNIST":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_qmnist = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((28, 28)),               
                transforms.ToTensor(),                       
            ])


            qmnist_dataset  = datasets.QMNIST(root='./data', train=True, transform=transform_qmnist, download=True)
            qmnist_loader  = DataLoader(dataset=qmnist_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_qmnist_contribution = len(qmnist_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_qmnist_maxed = int(len(qmnist_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_qmnist_maxed)
            mnist_size = int(0.96 * total_desired_size)
            qmnist_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            qmnist_indices = np.random.choice(len(qmnist_dataset), size=qmnist_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            qmnist_subset = Subset(qmnist_dataset, qmnist_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=qmnist_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.public_dataset == "KMNIST":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
            mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

            transform_kmnist = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((28, 28)),               
                transforms.ToTensor(),                       
            ])


            kmnist_dataset  = datasets.KMNIST(root='./data', train=True, transform=transform_kmnist, download=True)
            kmnist_loader  = DataLoader(dataset=kmnist_dataset, batch_size=64, shuffle=False)

            max_mnist_contribution = len(mnist_dataset) 
            max_kmnist_contribution = len(kmnist_dataset) 
            total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            total_if_kmnist_maxed = int(len(kmnist_dataset) / 0.04)
            total_desired_size = min(total_if_mnist_maxed, total_if_kmnist_maxed)
            mnist_size = int(0.96 * total_desired_size)
            kmnist_size = int(0.04 * total_desired_size)
            np.random.seed(42)
            mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            kmnist_indices = np.random.choice(len(kmnist_dataset), size=kmnist_size, replace=False)
            mnist_subset = Subset(mnist_dataset, mnist_indices)
            kmnist_subset = Subset(kmnist_dataset, kmnist_indices)
            train_loader_private = DataLoader(dataset=mnist_subset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=kmnist_subset, batch_size=64, shuffle=True)


            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        if self.public_dataset == "SEMEION":
            transform_mnist = transforms.Compose([
                transforms.ToTensor(),  
            ])

            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)

            transform_semion = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((28, 28)),               
                transforms.ToTensor(),                       
            ])

            full_dataset_2  = datasets.SEMEION(root='./data', transform=transform_semion, download=True)

            # max_mnist_contribution = len(mnist_dataset) 
            # max_semeion_contribution = len(train_dataset_2) 
            # total_if_mnist_maxed = int(len(mnist_dataset) / 0.96)
            # total_if_semeion_maxed = int(len(train_dataset_2) / 0.04)
            # total_desired_size = min(total_if_mnist_maxed, total_if_semeion_maxed)
            # mnist_size = int(0.96 * total_desired_size)
            # semeion_size = int(0.04 * total_desired_size)
            # np.random.seed(42)
            # mnist_indices = np.random.choice(len(mnist_dataset), size=mnist_size, replace=False)
            # semeion_indices = np.random.choice(len(train_dataset_2), size=semeion_size, replace=False)
            # mnist_subset = Subset(mnist_dataset, mnist_indices)
            # semeion_subset = Subset(train_dataset_2, semeion_indices)
            train_loader_private = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)
            train_loader_public = DataLoader(dataset=full_dataset_2, batch_size=64, shuffle=True)

            test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
            test_loader  = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            self.public_loader = train_loader_public
            self.train_loader = train_loader_private
            self.test_loader = test_loader
        
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
        self.model = mnist()
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