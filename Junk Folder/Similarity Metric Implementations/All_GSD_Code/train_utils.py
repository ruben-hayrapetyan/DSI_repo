import os
import math
import random
from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
from urllib import request
import zipfile
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset, Dataset

from resnet import resnet20
from densenet import DenseNet121


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        # return x
        return F.log_softmax(x, dim=1)


class Simple2ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=5, kernel_size=8, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, stride=3, kernel_size=4, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=5, kernel_size=8, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, stride=3, kernel_size=4, padding=1)         

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        # self.fc = nn.Linear(16 * 3 * 3, 10)
        self.fc = nn.Linear(16 * 1 * 1, 10)

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = self.pool(x)

        # Flatten the output of the convolutional layers
        # x = x.view(-1, 16 * 3 * 3)
        x = x.view(x.size(0), -1)   # now [B, 16*1*1]


        # Apply fully connected layer
        x = self.fc(x)

        return x


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Imagenet(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)
        self.label = torch.randint(high=10, size=(2000,))
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        target = self.label[index]

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class MNIST_M(Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train
        self.transform = transform
        if train:
            self.image_dir = os.path.join(root, 'mnist_m/mnist_m/mnist_m_train')
            labels_file = os.path.join(root, "mnist_m/mnist_m/mnist_m_train_labels.txt")
        else:
            self.image_dir = os.path.join(root, 'mnist_m/mnist_m/mnist_m_test')
            labels_file = os.path.join(root, "mnist_m/mnist_m/mnist_m_test_labels.txt")

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image, label = self.mapping[idx]
        image = os.path.join(self.image_dir, image)
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


# from https://github.com/zoogzog/chexnet
class ChestXRay(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # ---- Open file, get image paths and labels

        fileDescriptor = open(pathDatasetFile, "r")

        # ---- get into the loop
        line = True

        while line:

            line = fileDescriptor.readline()

            # --- if not empty
            if line:
                lineItems = line.split()

                imageNameList = lineItems[0].split('/')
                imageNameList.append(imageNameList[1])
                imageNameList[1] = "images"
                imageName = '/'.join(imageNameList)
                imagePath = os.path.join(pathImageDirectory, imageName)
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        # imageLabel = torch.argmax(imageLabel)

        if self.transform is not None:
            imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.listImagePaths)


# from https://github.com/kamenbliznashki/chexpert
class ChexpertSmall(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # select only the competition labels
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, mode='train', view='frontal', transform=None, data_filter=None, mini_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ['train', 'valid', 'test', 'vis']
        assert view in ['Frontal', 'Lateral', 'All']
        self.mode = mode
        self.view = view

        # if mode is test; root is path to csv file (in test mode), construct dataset from this csv;
        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.
        if mode == 'test':
            self.data = pd.read_csv(self.root, keep_default_na=True)
            self.root = '.'  # base path; to be joined to filename in csv file in __getitem__
            self.data[self.attr_names] = pd.DataFrame(
                np.zeros((len(self.data), len(self.attr_names))))  # attr is vector of 0s under test
        else:
            self._maybe_download_and_extract()
            self._maybe_process(data_filter)

            data_file = os.path.join(self.root, self.dir_name, 'valid.pt' if mode in ['valid', 'vis'] else 'train.pt')
            self.data = torch.load(data_file)

            if self.view != 'All':
                self.data = self.data[self.data['Frontal/Lateral'] == view]
            if mini_data is not None:
                # truncate data to only a subset for debugging
                self.data = self.data[:mini_data]

            if mode == 'vis':
                # select a subset of the data to visualize:
                #   3 examples from each condition category + no finding category + multiple conditions category

                # 1. get data idxs with a/ only 1 condition, b/ no findings, c/ 2 conditions, d/ >2 conditions; return list of lists
                idxs = []
                data = self.data
                for attr in self.attr_names:  # 1 only condition category
                    idxs.append(self.data.loc[(self.data[attr] == 1) & (
                            self.data[self.attr_names].sum(1) == 1), self.attr_names].head(3).index.tolist())
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1) == 0, self.attr_names].head(
                    3).index.tolist())  # no findings category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1) == 2, self.attr_names].head(
                    3).index.tolist())  # 2 conditions category
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1) > 2, self.attr_names].head(
                    3).index.tolist())  # >2 conditions category
                # save labels to visualize with a list of list of the idxs corresponding to each attribute
                self.vis_attrs = self.attr_names + ['No findings', '2 conditions', 'Multiple conditions']
                self.vis_idxs = idxs

                # 2. select only subset
                idxs_flatten = torch.tensor([i for sublist in idxs for i in sublist])
                self.data = self.data.iloc[idxs_flatten]

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[
            idx]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
        # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return img, attr

    def __len__(self):
        return len(self.data)

    def _maybe_download_and_extract(self):
        fpath = os.path.join(self.root, os.path.basename(self.url))
        # if data dir does not exist, download file to root and unzip into dir_name
        if not os.path.exists(os.path.join(self.root, self.dir_name)):
            # check if zip file already downloaded
            if not os.path.exists(os.path.join(self.root, os.path.basename(self.url))):
                print('Downloading ' + self.url + ' to ' + fpath)

                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
                                                                     float(count * block_size) / float(
                                                                         total_size) * 100.0))
                    sys.stdout.flush()

                request.urlretrieve(self.url, fpath, _progress)
                print()
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
                if os.path.exists(os.path.join(self.root, self.dir_name, '__MACOSX')):
                    os.rmdir(os.path.join(self.root, self.dir_name, '__MACOSX'))
            os.unlink(fpath)
            print('Dataset extracted.')

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, self.dir_name, 'train.pt')
        valid_file = os.path.join(self.root, self.dir_name, 'valid.pt')
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'), keep_default_na=True)
            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, self.dir_name, 'train.csv'),
                                                               data_filter)

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1, 1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k] == v]

            with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
                json.dump(data_filter, f)

        return train_df


# from: https://www.kaggle.com/code/xinruizhuang/skin-lesion-classification-acc-90-pytorch
class HAM10000(Dataset):
    """
    7 classes
    """

    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        df = pd.read_csv(os.path.join(data_dir, 'HAM10000_withtrainval.csv'))
        if train:
            self.df = df[df['train_or_val'] == 'train'].reset_index(drop=True)
        else:
            self.df = df[df['train_or_val'] == 'val'].reset_index(drop=True)

        '''
        Preprocess
        all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'dermatofibroma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
        df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
        df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
        df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

        # this will tell us how many images are associated with each lesion_id
        df_undup = df_original.groupby('lesion_id').count()
        # now we filter out lesion_id's that have only one image associated with it
        df_undup = df_undup[df_undup['image_id'] == 1]
        df_undup.reset_index(inplace=True)

        def get_duplicates(x):
            unique_list = list(df_undup['lesion_id'])
            if x in unique_list:
                return 'unduplicated'
            else:
                return 'duplicated'

        # create a new colum that is a copy of the lesion_id column
        df_original['duplicates'] = df_original['lesion_id']
        # apply the function to this new column
        df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

        # now we filter out images that don't have duplicates
        df_undup = df_original[df_original['duplicates'] == 'unduplicated']

        # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
        y = df_undup['cell_type_idx']
        _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

        # This set will be df_original excluding all rows that are in the val set
        # This function identifies if an image is part of the train or val set.
        def get_val_rows(x):
            # create a list of all the lesion_id's in the val set
            val_list = list(df_val['image_id'])
            if str(x) in val_list:
                return 'val'
            else:
                return 'train'

        # identify train and val rows
        # create a new colum that is a copy of the image_id column
        df_original['train_or_val'] = df_original['image_id']
        # apply the function to this new column
        df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
        # filter out train rows
        df_train = df_original[df_original['train_or_val'] == 'train']
        '''

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))
        y = torch.nn.functional.one_hot(y, num_classes=7)
        y = y.float()
        if self.transform:
            X = self.transform(X)

        return X, y


class KagChest(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label = {'NORMAL': 0, 'PNEUMONIA': 1}
        df = pd.read_csv(os.path.join(data_dir, 'kagchest.csv'))
        if train:
            self.df = df[df['train_or_val'] == 'train'].reset_index(drop=True)
        else:
            self.df = df[df['train_or_val'] == 'val'].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(os.path.join(self.data_dir, self.df['image_path'][index])).convert('RGB')
        y = torch.tensor(self.label[self.df['label'][index]])
        if self.transform:
            X = self.transform(X)

        return X, y


class KagSkin(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label = {'benign': 0, 'malignant': 1}
        df = pd.read_csv(os.path.join(data_dir, 'kagskin.csv'))
        if train:
            self.df = df[df['train_or_val'] == 'train'].reset_index(drop=True)
        else:
            self.df = df[df['train_or_val'] == 'val'].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(os.path.join(self.data_dir, self.df['image_path'][index])).convert('RGB')
        y = torch.tensor(self.label[self.df['label'][index]])
        if self.transform:
            X = self.transform(X)

        return X, y


def keep_layer(model):
    keep_list = ['layer3.0.conv', 'layer3.1.conv', 'fc.weight', 'bn', 'downsample.1']
    # keep_list = ['fc.weight', 'bn', 'downsample.1']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for keep in keep_list:
            if keep in name:
                param.requires_grad = True
    return model


def save_model(model, epoch, ckpt_loc):
    state = {
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(state, ckpt_loc)


def load_model(model, ckpt_loc, load=False):
    if not load:
        return False

    try:
        checkpoint = torch.load(ckpt_loc)
        cur_state = model.state_dict()
        state_dict = checkpoint['model']
        # restore_param
        own_state = cur_state
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        return model, checkpoint['epoch']
    except:
        return False


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_data_loader(args):
    """
        Chestx-ray14 is a considerable large dataset, we assume for such dataset, we can collect around 2000 public examples
        HAM10000 is relatively small, we assume we can only use 300 of them as public
        We assume that the first 2000/300 examples in the testset are public
    """
    dataset_slice = list(range(args.num_examples))
    if args.private_dataset == "chestxray":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        private_dataset = ChestXRay(pathImageDirectory='../data/chest/chest',
                                    pathDatasetFile='../data/chest/chest/test_1.txt',
                                    transform=transform_train)
        public_slice = list(range(len(private_dataset)))
        private_dataset = Subset(private_dataset, public_slice[len(dataset_slice):])
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "ham":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        private_dataset = HAM10000('../data/ham10000', train=True, transform=train_transform)
        public_slice = list(range(len(private_dataset)))
        private_dataset = Subset(private_dataset, public_slice[len(dataset_slice):])
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        private_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_train)
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "kagchest":
        transform_train = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomRotation(degrees=(-20, +20)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        private_dataset = KagChest('../data/chest_xray/', train=True, transform=transform_train)
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "kagskin":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        private_dataset = KagSkin('../data/kagskin/', train=True, transform=train_transform)
        private_dataset = Subset(private_dataset, dataset_slice)
    elif args.private_dataset == "mnist":
        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),            # if your model expects 3-channel input
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        private_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=tf
        )

    elif args.private_dataset == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        private_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=tf
        )
    elif args.private_dataset == "fashionmnist":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        private_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=tf
        )
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "svhn":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        private_dataset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=tf
        )
        private_dataset = Subset(private_dataset, dataset_slice)

    elif args.private_dataset == "stl10":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        private_dataset = datasets.STL10(
            root='./data',
            split='train',
            download=True,
            transform=tf
        )
        private_dataset = Subset(private_dataset, dataset_slice)
    elif args.private_dataset == "usps":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        private_dataset = datasets.USPS(
            root='./data',
            train=False,  # Using test split
            download=True,
            transform=tf
        )
        private_dataset = Subset(private_dataset, dataset_slice)

    private_loader = DataLoader(private_dataset, batch_size=args.train_batch_size, shuffle=True,
                                pin_memory=True, num_workers=8)
    

    if args.public_dataset == "chestxray":
        public_set = ChestXRay(pathImageDirectory='../data/chest/chest',
                               pathDatasetFile='../data/chest/chest/test_1.txt',
                               transform=transforms.Compose([
                                   transforms.Resize(size=(256, 256)),
                                   transforms.RandomRotation(degrees=(-20, +20)),
                                   transforms.CenterCrop(size=224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "kagchest":
        transform_train = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomRotation(degrees=(-20, +20)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        public_set = KagChest('../data/chest_xray/', train=True, transform=transform_train)
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        public_set = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_train)
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "ham":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        public_set = HAM10000('../data/ham10000', train=True, transform=train_transform)
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "kagskin":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        public_set = KagSkin('../data/kagskin/', train=True, transform=train_transform)
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "mnist":
        tf = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        public_set = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=tf
        )

    elif args.public_dataset == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        public_set = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=tf
        )
    elif args.public_dataset == "usps":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        public_set = datasets.USPS(
            root='./data',
            train=False,  # Using test split
            download=True,
            transform=tf
        )
        public_set = Subset(public_set, dataset_slice)
    elif args.public_dataset == "fashionmnist":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        public_set = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=tf
        )
        public_set = Subset(public_set, dataset_slice)

    elif args.public_dataset == "svhn":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        public_set = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=tf
        )
        public_set = Subset(public_set, dataset_slice)

    elif args.public_dataset == "stl10":
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        public_set = datasets.STL10(
            root='./data',
            split='train',
            download=True,
            transform=tf
        )
        public_set = Subset(public_set, dataset_slice)

    public_loader = DataLoader(public_set, batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=8)

    return private_loader, public_loader
