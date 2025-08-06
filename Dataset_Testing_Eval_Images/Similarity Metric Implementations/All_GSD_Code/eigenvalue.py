"""
    Record of eigen-value spectrum along the training trajectory
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

from power_iter import compute_svd_things, compute_all_svd_things, get_batch_gradients

# from opacus.validators import ModuleValidator
from subspace_distance import matrix_l2_norm, projection_metric
from train_utils import save_model, load_model, WarmupLinearSchedule, AverageMeter, \
    set_seed, get_data_loader

from resnet import resnet20
from resnet_pytorch import resnet18

# package for computing individual gradients
from backpack import backpack, extend

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

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


def setup_model(args):
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 14), nn.Sigmoid())

    keep_list = ['fc']

    model = keep_layer(model, keep_list)
    model.to(args.device)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Use cuda or not")
    parser.add_argument("--private_dataset", choices=["cifar10", "svhn", "mnist", "fmnist", "chestxray"],
                        default="chestxray",
                        help="Which downstream task.")
    parser.add_argument("--public_dataset",
                        choices=["cifar10", "svhn", "cifar100", "imagenet", "mnist",
                                 "texture", "fake", "stl", "sun", "mnist_m", "emnist",
                                 "flower", "chestxray", "chexpert"],
                        default="chexpert",
                        help="Which downstream task.")
    parser.add_argument("--use_pretrain", default=False,
                        help="Which downstream task.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=1000, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=0.05, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epoch", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--num_eigenthings', type=int, default=500,
                        help="random seed for initialization")
    parser.add_argument('--num_examples', type=int, default=2000,
                        help="number of public/private examples for subspace comparison")

    # Privacy params
    parser.add_argument("--use_dp", default=True,
                        help="Whether train with DP or not.")
    parser.add_argument("--max_grad_norm", type=float, default=3.,
                        help="The maximum L2 norm of per-sample gradients before they are aggregated by the averaging "
                             "step.")
    parser.add_argument("--max_physical_batch_size", type=int, default=64,
                        help="Can be used both for simulating large logical batches with limited memory.")
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="Privacy budget.")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Generally, it should be set to be less than the inverse of the size of the training "
                             "dataset.")

    args = parser.parse_args()

    log_loc = os.path.join('./log/eigenvals/', args.public_dataset + '-' + args.private_dataset + '_' + args.name + '.txt')
    if os.path.exists(log_loc):
        os.remove(log_loc)

    # Set seed
    set_seed(args)

    # Prepare dataset
    private_train_loader, private_test_loader, public_loader = get_data_loader(args)

    # Prepare model
    model = setup_model(args)

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    n_epoch = args.num_epoch
    t_total = n_epoch * (len(private_train_loader.dataset) // args.train_batch_size)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    privacy_engine = PrivacyEngine()
    model, optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=private_train_loader,
        epochs=n_epoch,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )
    optimizer.set_k(args.num_eigenthings)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    epoch = 0
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    while True:
        model.train()

        with BatchMemoryManager(
                data_loader=private_train_loader,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer=optimizer
        ) as memory_safe_data_loader:
            epoch_iterator = tqdm(private_train_loader, desc="Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                losses.update(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
            eigenvals = optimizer.eigenvals
            f = open(log_loc, "a+")
            np.savetxt(f, eigenvals.cpu().numpy(), delimiter=',')

            f.write('\t')
            f.close()
            epoch += 1
            if epoch % n_epoch == 0:
                break
