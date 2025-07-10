import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import copy
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import re
import string
from collections import Counter

class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)

        imdb = load_dataset("imdb")

        def clean_text(text):
           text = text.lower()
           text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
           text = re.sub(r"\s+", " ", text).strip()
           return text


        def tokenize(text):
           return clean_text(text).split()


        counter = Counter()
        for example in imdb["train"]:
           counter.update(tokenize(example["text"]))


        min_freq = 3
        vocab = {"<pad>": 0, "<unk>": 1}
        for word, freq in counter.items():
           if freq >= min_freq:
               vocab[word] = len(vocab)
        inv_vocab = {v: k for k, v in vocab.items()}
        PAD_IDX = vocab["<pad>"]
        UNK_IDX = vocab["<unk>"]


        def text_pipeline(text, max_len=256):
           tokens = tokenize(text)
           ids = [vocab.get(token, UNK_IDX) for token in tokens[:max_len]]
           return torch.tensor(ids, dtype=torch.long)

        class Model(nn.Module):
           def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=2):
               super().__init__()
               self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
               self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
               self.fc = nn.Linear(hidden_dim, num_classes)


           def forward(self, x):
               x = self.embedding(x)
               _, (h_n, _) = self.lstm(x)
               return self.fc(h_n[-1])
           
        class TextDataset(Dataset):
            def __init__(self, hf_dataset, label_transform, text_field, max_len=256):
                self.data = hf_dataset
                self.label_transform = label_transform
                self.max_len = max_len
                self.text_field = text_field

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                tokens = text_pipeline(item[self.text_field], max_len=self.max_len)
                label = torch.tensor(self.label_transform(item["label"]), dtype=torch.long)
                return tokens, label

        def collate_batch(batch):
            texts, labels = zip(*batch)
            padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
            return padded, torch.stack(labels)

        if (self.dataset == "CornellMovie"):
            cornellmovie = load_dataset("cornell-movie-review-data/rotten_tomatoes")

            max_imdb_contribution = len(imdb["train"])
            max_cornellmovie_contribution = len(cornellmovie["train"])

            total_if_imdb_maxed = int(len(imdb["train"]) / 0.04)
            total_if_cornellmovie_maxed = int(len(cornellmovie["train"]) / 0.96)
            total_desired_size = min(total_if_imdb_maxed, total_if_cornellmovie_maxed)
            imdb_size = int(0.04 * total_desired_size)
            cornellmovie_size = int(0.96 * total_desired_size)

            np.random.seed(42)
            imdb_indices = np.random.choice(len(imdb["train"]), size=imdb_size, replace=False)
            cornellmovie_indices = np.random.choice(len(cornellmovie["train"]), size=cornellmovie_size, replace=False)

            imdb_subset = imdb["train"].select(imdb_indices)
            cornellmovie_subset = cornellmovie["train"].select(cornellmovie_indices)

            self.public_loader = DataLoader(
                TextDataset(imdb_subset, label_transform=lambda x: x, text_field="text"),
                batch_size=64, shuffle=True, collate_fn=collate_batch
            )

            self.train_loader = DataLoader(
                TextDataset(cornellmovie_subset, label_transform=lambda x: x, text_field="text"),
                batch_size=64, shuffle=True, collate_fn=collate_batch
            )

            self.test_loader = DataLoader(
                TextDataset(cornellmovie["test"], label_transform=lambda x: x, text_field="text"),
                batch_size=64, shuffle=False, collate_fn=collate_batch
            )
        
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
        self.model = Model(len(vocab), num_classes=2)
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

