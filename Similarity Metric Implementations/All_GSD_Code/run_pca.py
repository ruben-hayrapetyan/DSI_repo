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
    keep_layer, set_seed, get_data_loader, setup_model

from resnet import resnet20
from resnet_pytorch import resnet18

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            output = model(x)
            logits = model(x)[0]

            eval_loss = loss_fct(output, y)
            eval_losses.update(eval_loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)

        if len(all_preds) == 0:
            all_preds.append(preds)
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds, axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    print("\n===> Acc: ", accuracy)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Use cuda or not")
    parser.add_argument("--private_dataset", choices=["cifar10", "svhn", "mnist", "fmnist", "chestx"],
                        default="chestx",
                        help="Which downstream task.")
    parser.add_argument("--public_dataset",
                        choices=["cifar10", "svhn", "cifar100", "imagenet", "mnist",
                                 "texture", "fake", "stl", "sun", "mnist_m", "emnist",
                                 "flower", "chestx", "covid"],
                        default="covid",
                        help="Which downstream task.")
    parser.add_argument("--use_pretrain", default=False,
                        help="Which downstream task.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=2000, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=0.15, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epoch", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--num_eigenthings', type=int, default=16,
                        help="random seed for initialization")
    parser.add_argument('--num_examples', type=int, default=2000,
                        help="number of public/private examples for subspace comparison")


    # Privacy params
    parser.add_argument("--use_dp", default=True,
                        help="Whether train with DP or not.")
    parser.add_argument("--max_grad_norm", type=float, default=1.1,
                        help="The maximum L2 norm of per-sample gradients before they are aggregated by the averaging "
                             "step.")
    parser.add_argument("--max_physical_batch_size", type=int, default=16,
                        help="Can be used both for simulating large logical batches with limited memory.")
    parser.add_argument("--noise_multiplier", type=float, default=1.0,
                        help="The amount of noise sampled and added to the average of the gradients in a batch.")
    parser.add_argument("--epsilon", type=float, default=6.0,
                        help="Privacy budget.")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Generally, it should be set to be less than the inverse of the size of the training "
                             "dataset.")

    args = parser.parse_args()

    log_loc = os.path.join('./log', args.public_dataset + '-' + args.private_dataset + '_' + args.name + '.txt')
    if os.path.exists(log_loc):
        os.remove(log_loc)

    # Set seed
    set_seed(args)

    # Prepare dataset
    private_train_loader, private_test_loader, public_loader = get_data_loader(args)

    # Prepare model
    model = setup_model(args)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    n_epoch = args.num_epoch
    t_total = n_epoch * (len(private_train_loader.dataset) // args.train_batch_size)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    losses = AverageMeter()
    global_step, best_acc = 0, 0
    epoch = 0
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    while True:
        model.train()
        epoch_iterator = tqdm(private_train_loader, desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        subspace_distance = []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            # start to compare subspace distance
            f = open(log_loc, "a+")

            # computer topk singular vector
            eigenvals_priv, eigenvecs_priv = compute_all_svd_things(model, private_test_loader, criterion,
                                                                    k=args.num_eigenthings,
                                                                    num_examples=args.num_examples,
                                                                    device=args.device)
            print("\n===> half svd things done")  # p x k

            eigenvals_pub, eigenvecs_pub = compute_all_svd_things(model, public_loader, criterion,
                                                                  k=args.num_eigenthings,
                                                                  num_examples=args.num_examples,
                                                                  device=args.device)
            print("\n===> svd things done")

            print("\n===> computing subspace closeness")

            subspace_closeness = projection_metric(eigenvecs_priv, eigenvecs_pub).item()
            subspace_distance.append(subspace_closeness)
            print(f"subspace distance: {subspace_closeness}")

            f.write(str(subspace_closeness))
            f.write(",")
            f.close()

            del eigenvecs_pub, eigenvecs_priv
            model.zero_grad()

            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            scheduler.step()

            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

            if global_step % t_total == 0:
                break

        print(f"average subspace distance: {np.mean(subspace_distance)}")
        global_step = t_total
        accuracy = valid(args, model, private_test_loader)

        if best_acc < accuracy:
            best_acc = accuracy
        model.train()
        losses.reset()
        losses.reset()
        if global_step % t_total == 0:
            break

        epoch += 1
