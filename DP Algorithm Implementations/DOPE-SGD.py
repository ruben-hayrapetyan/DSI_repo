import torch
from torchvision import datasets, transforms
from torch.utils.data import *
import numpy as np

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

def DOPESGD(model, D_loader, D_s_loader, lr, sigma, C, l, T):
    """
    Parameters:
        D_loader: private training data loader
        n: private batch size
        D_s_loader: public training data loader
        n_s: public batch size
        lr (float): learning rate
        sigma (float): noise scale
        C: gradient norm clip
        l: loss function
        T (int): number of training iterations
    Returns:
        theta: trained model parameters
    """
    
    #initiate theta randomly
    optim = torch.optim.SGD(model.parameters(), lr)

    for epoch in range(T):

        model.train()
        total_loss = 0
        pub_iter = iter(D_s_loader)

        for priv_inputs, priv_labels in D_loader:
            
            pub_inputs, pub_labels = next(pub_iter)
            optim.zero_grad()
            pub_out = model(pub_inputs)
            pub_loss = l(pub_out, pub_labels)
            pub_loss.backward()





    #for t in T
        #B_t is the sample of n instances from D
        #B_s is the sample of n_s instances from D_s
        #∇sθ ← ∇L(B_s) (Instantiate gˆ in Algorithm 1 with Bs)
        #∇Gθ[t] ← 0 vector
        #for all (x, y) in B_t
            #and more..

    #return theta
    return 0