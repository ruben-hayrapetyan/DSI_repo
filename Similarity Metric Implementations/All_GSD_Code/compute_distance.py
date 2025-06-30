"""
    For Chestx-Ray
"""

import logging
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import time

from power_iter import compute_all_svd_things, privately_select_topk, privately_select_topk_np, compute_topk_with_input_perturbation

# from opacus.validators import ModuleValidator
from subspace_distance import matrix_l2_norm, projection_metric, projection_metric_dp
from train_utils import save_model, load_model, WarmupLinearSchedule, AverageMeter, \
    set_seed, get_data_loader, Simple2ConvNet

import loralib as lora
from timm import create_model

from resnet import resnet20
from resnet_pytorch import resnet18

# package for computing individual gradients
# from backpack import backpack, extend

import functorch
from functools import partial
from functorch.experimental import replace_all_batch_norm_modules_

from opacus import GradSampleModule

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def keep_layer(model, keep_list):
    """
        Args: keep_list ['fc', 'bn']
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        for keep in keep_list:
            if keep in name and "weight" in name:
                param.requires_grad = True
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def ReLU_inplace_to_False(model):
    for layer in model.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)
    return model


def setup_model(args):
    if args.private_dataset == "chestxray" or args.public_dataset == "chestxray":
        num_classes = 14
    elif args.private_dataset == "ham" or args.public_dataset == "ham":
        num_classes = 7
    elif args.private_dataset == "cifar100":
        num_classes = 100
    elif args.private_dataset == "kagchest":
        num_classes = 2
    elif args.private_dataset == "mnist" or args.public_dataset == "mnist":
        num_classes = 10
    elif args.private_dataset == "cifar10" or args.public_dataset == "cifar10":
        num_classes = 10
    elif args.private_dataset in ["fashionmnist", "svhn", "stl10"] or \
         args.public_dataset in ["usps"]:
        num_classes = 10

    if args.model_type == "wideresnet":
        model = models.wide_resnet101_2(pretrained=True)
        keep_list = ['fc', 'layer3.0.conv1.weight']
        num_ftrs = model.fc.in_features
        # Prepare the model for private ft
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model = keep_layer(model, keep_list)
    elif args.model_type == "resnet152":
        model = models.resnet152(pretrained=True)
        # keep_list = ['fc', 'layer3.32.conv1.weight']
        keep_list = ['fc']
        num_ftrs = model.fc.in_features
        # num_ftrs = 512
        # Prepare the model for private ft
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model = keep_layer(model, keep_list)
    elif args.model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        keep_list = ['fc']
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model = keep_layer(model, keep_list)
    elif args.model_type == "densenet":
        model = models.densenet121(pretrained=True)
        keep_list = ['classifier', 'features.denseblock3.denselayer23.conv1.weight',
                     'features.denseblock3.denselayer24.conv1.weight']
        num_ftrs = model.classifier.in_features
        # Prepare the model for private ft
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model = keep_layer(model, keep_list)
    elif args.model_type == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
        keep_list = ['classifier']
        model = keep_layer(model, keep_list)
    elif args.model_type == "vit":
        model = create_model(
            'vit_base_patch16_224',
            num_classes=num_classes,
            apply_lora=True,
            lora_r=8,
            lora_alpha=8,
            pretrained=True
        )
        lora.mark_only_lora_as_trainable(model)
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())
    elif args.model_type == "vit_tiny":
        model = create_model(
            'vit_tiny_patch16_224',
            num_classes=num_classes,
            apply_lora=True,
            lora_r=1,
            lora_alpha=1,
            pretrained=True
        )
        model.head = nn.Sequential(model.head, nn.Sigmoid())
        lora.mark_only_lora_as_trainable(model)
    elif args.model_type == "probe":
        model = Simple2ConvNet()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

    num_params = count_parameters(model)
    print(f"number of parameters: {num_params}")

    model.to(args.device)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Use cuda or not")
    parser.add_argument("--model_type", choices=["wideresnet", "resnet152", "vit", "densenet", "probe", "resnet18", "vit_tiny"],
                        default="resnet152",
                        help="model")
    parser.add_argument("--private_dataset",
                        choices=["chestxray","ham","cifar100","kagchest","kagskin","mnist","cifar10", "fashionmnist", "svhn", "stl10", "usps"],
                        default="mnist",
                        help="Which downstream task.")
    parser.add_argument("--public_dataset",
                        choices=["chestxray","kagchest","ham","cifar100","kagskin","mnist","cifar10", "fashionmnist", "svhn", "stl10", "usps"],
                        default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--use_pretrain", default=False,
                        help="Which downstream task.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--num_eigenthings', type=int, default=16,
                        help="random seed for initialization")
    parser.add_argument('--num_examples', type=int, default=2000,
                        help="number of public/private examples for subspace comparison")
    parser.add_argument('--epsilon', type=float, default=4.,
                        help="privacy parameter")
    parser.add_argument('--delta', type=float, default=5e-4,
                        help="privacy parameter")
    parser.add_argument('--max_grad_norm', type=float, default=6.,
                        help="Clip norm")

    args = parser.parse_args()

    # Set seed
    set_seed(args)

    # Prepare dataset
    private_loader, public_loader = get_data_loader(args)

    # Prepare model
    model = setup_model(args)

    # Opacus yyds!
    model = GradSampleModule(model)

    if args.private_dataset == "chestxray" \
            or args.private_dataset == "ham" \
            or args.public_dataset == "ham" \
            or args.public_dataset == "chestxray":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    model.train()

    # start to compare subspace distance
    t0 = time.time()
    # computer topk singular vector
    eigenvals_priv, eigenvecs_priv = compute_all_svd_things(args, model, private_loader, criterion)
    print("\n===> half svd things done")  # p x k

    eigenvals_pub, eigenvecs_pub = compute_all_svd_things(args, model, public_loader, criterion)

    t1 = time.time()
    print("\n===> svd things done")
    print('time: %d s' % (t1 - t0))

    print("\n===> computing subspace closeness")

    subspace_closeness = projection_metric(eigenvecs_priv, eigenvecs_pub).item()


    print(f"subspace distance: {subspace_closeness}")
