import numpy as np
from torch import Tensor

import torch

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from torch.nn.functional import normalize

import torch.nn.functional as F

from train_utils import HAM10000, ChestXRay

from distributions import MatrixBingham_EX, MatrixBingham


def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def get_per_sample_gradients(model, data_batch, criterion):
    model.zero_grad()
    inputs, labels = data_batch
    # labels = torch.randint(high=100, size=labels.shape, device=labels.device)
    pred = model(inputs)
    loss = criterion(pred, labels)

    with backpack(BatchGrad()):
        loss.backward()
    cur_batch_grad_list = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        cur_batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
        del p.grad_batch
    return flatten_tensor(cur_batch_grad_list)  # n x p


def get_batch_gradients(model, data_batch, criterion):
    """
        Returns a flatten gradients tensor (p, )
    """
    all_grads = None
    model.zero_grad()

    inputs, labels = data_batch
    # labels = torch.randint(high=100, size=labels.shape, device=labels.device)
    pred = model(inputs)
    loss = criterion(pred, labels)
    loss.backward()

    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if all_grads is None:
            all_grads = p.grad.flatten()
        else:
            all_grads = torch.cat((all_grads, p.grad.flatten()))
    return all_grads


def random_label(args, labels):
    """
        Given dataset (private or public), return corresponding random label
    """
    if args.private_dataset == "chestxray" or args.public_dataset == "chestxray":
        labels = torch.randint(high=14, size=(labels.shape[0],), device=labels.device)
        labels = F.one_hot(labels, num_classes=14)
        labels = labels.float()
    elif args.private_dataset == "ham" or args.public_dataset == "ham":
        labels = torch.randint(high=7, size=(labels.shape[0],), device=labels.device)
        labels = F.one_hot(labels, num_classes=7)
        labels = labels.float()
    else:
        labels = torch.randint(high=2, size=labels.shape, device=labels.device)
    return labels


def get_all_dataset_gradients(args, model, dataloader, criterion):
    """
        Backpack, it works, but...
    """
    all_flatten_tensor = None
    cur_num_examples = 0

    for step, batch in enumerate(dataloader):
        model.zero_grad()
        cur_batch_grad_list = []
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = batch
        labels = random_label(labels, dataloader)
        pred = model(inputs)
        loss = criterion(pred, labels)

        with backpack(BatchGrad()):
            loss.backward()

        cur_num_examples += len(labels)
        del batch, inputs, labels

        for p in filter(lambda p: p.requires_grad, model.parameters()):
            cur_batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
            del p.grad_batch, p.grad

        if all_flatten_tensor is None:
            all_flatten_tensor = flatten_tensor(cur_batch_grad_list)
        else:
            all_flatten_tensor = torch.vstack((all_flatten_tensor, flatten_tensor(cur_batch_grad_list)))

        if cur_num_examples > args.num_examples:
            break
    return all_flatten_tensor[:args.num_examples, :]  # n x p


def get_all_dataset_gradients_opacus(args, model, dataloader, criterion):
    """
        Opacus yyds!
    """
    all_flatten_tensor = None
    cur_num_examples = 0

    for step, batch in enumerate(dataloader):
        model.zero_grad()
        cur_batch_grad_list = []
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = batch
        labels = random_label(args, labels)
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()

        cur_num_examples += len(labels)

        for p in filter(lambda p: p.requires_grad, model.parameters()):
            cur_batch_grad_list.append(p.grad_sample.reshape(p.grad_sample.shape[0], -1))
            del p.grad_sample, p.grad

        if all_flatten_tensor is None:
            all_flatten_tensor = flatten_tensor(cur_batch_grad_list)
        else:
            all_flatten_tensor = torch.vstack((all_flatten_tensor, flatten_tensor(cur_batch_grad_list)))

        if cur_num_examples > args.num_examples:
            break
    return all_flatten_tensor[:args.num_examples, :]  # n x p


def get_private_dataset_gradients_opacus(args, model, dataloader, criterion):
    all_flatten_tensor = None
    cur_num_examples = 0

    for step, batch in enumerate(dataloader):
        model.zero_grad()
        cur_batch_grad_list = []
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = batch
        labels = random_label(args, labels)
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()

        cur_num_examples += len(labels)

        for p in filter(lambda p: p.requires_grad, model.parameters()):
            cur_batch_grad_list.append(p.grad_sample.reshape(p.grad_sample.shape[0], -1))
            del p.grad_sample, p.grad

        if all_flatten_tensor is None:
            all_flatten_tensor = flatten_tensor(cur_batch_grad_list)
        else:
            all_flatten_tensor = torch.vstack((all_flatten_tensor, flatten_tensor(cur_batch_grad_list)))

        if cur_num_examples > args.num_examples:
            break

    all_flatten_tensor = all_flatten_tensor[:args.num_examples, :]
    # clipping step
    per_sample_norms = torch.linalg.vector_norm(all_flatten_tensor, dim=1)
    per_sample_clip_factor = (args.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
        max=1.0
    )
    all_flatten_tensor = torch.einsum("i,ij->ij", per_sample_clip_factor, all_flatten_tensor)

    return all_flatten_tensor  # n x p


def get_all_dataset_gradients_functorch(args, model, dataloader, criterion, params, buffers, ft_compute_sample_grad):
    """
        Not working
    """
    all_flatten_tensor = None
    cur_num_examples = 0

    for step, batch in enumerate(dataloader):
        model.zero_grad()
        cur_batch_grad_list = []
        batch = tuple(t.to(args.device) for t in batch)
        inputs, labels = batch
        labels = random_label(labels, dataloader)

        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, labels)

        cur_num_examples += len(labels)

        for p in filter(lambda grads: grads.requires_grad, ft_per_sample_grads):
            cur_batch_grad_list.append(p.reshape(p.shape[0], -1))

        if all_flatten_tensor is None:
            all_flatten_tensor = flatten_tensor(cur_batch_grad_list)
        else:
            all_flatten_tensor = torch.vstack((all_flatten_tensor, flatten_tensor(cur_batch_grad_list)))

        if cur_num_examples > args.num_examples:
            break
    return all_flatten_tensor[:args.num_examples, :]  # n x p


def get_all_dataset_gradients_v2(model, dataloader, criterion, device='cuda'):
    """
        Use gradient accumulation and no backpack
        This is experimental (or say, shit)
    """
    model.zero_grad()
    all_flatten_tensor = None

    for step, batch in enumerate(dataloader):
        cur_batch_grad_list = []
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        labels = torch.randint(high=10, size=labels.shape, device=labels.device)
        pred = model(inputs)
        loss = criterion(pred, labels)

        with backpack(BatchGrad()):
            loss.backward()

        del batch, inputs, labels
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            cur_batch_grad_list.append(p.grad.reshape(1, -1))
            del p.grad

        if all_flatten_tensor is None:
            all_flatten_tensor = flatten_tensor(cur_batch_grad_list)
        else:
            all_flatten_tensor = torch.vstack((all_flatten_tensor, flatten_tensor(cur_batch_grad_list)))

    return all_flatten_tensor  # n x p


def compute_grad(model, sample, target, loss_fn):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(filter(lambda p: p.requires_grad, model.parameters())))


def compute_sample_grads(model, data_batch, criterion):
    """not working, require too much memory"""
    """ manually process each sample with per sample gradient """
    inputs, labels = data_batch
    labels = torch.randint(high=10, size=labels.shape, device=labels.device)
    batch_size = len(labels)
    sample_grads = [compute_grad(model, inputs[i], labels[i], criterion) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def compute_svd_things(model, data_batch, criterion, k=6):
    A = get_per_sample_gradients(model, data_batch, criterion)
    # A = compute_sample_grads(model, data_batch, criterion)
    A = normalize(A)
    # u, s, v = torch.svd_lowrank(A, q=k)
    u, s, vh = torch.linalg.svd(A, full_matrices=False)

    v = vh.conj().transpose(-2, -1)
    v = v[:, :k]
    return s[:k], v


def compute_all_svd_things(args, model, dataloader, criterion):
    k = args.num_eigenthings
    A = get_all_dataset_gradients_opacus(args, model, dataloader, criterion)

    A = normalize(A)
    u, s, vh = torch.linalg.svd(A, full_matrices=False)
    del A
    v = vh.conj().transpose(-2, -1)
    v = v[:, :k]
    return s[:k], v


def privately_select_topk(args, model, dataloader, criterion):
    k = args.num_eigenthings
    A = get_private_dataset_gradients_opacus(args, model, dataloader, criterion)
    A = (1 / A.shape[1]) * A.T @ A
    A = (A.shape[1] * args.epsilon / (2 * args.max_grad_norm * args.max_grad_norm)) * A
    v = MatrixBingham_EX(A, k=k).sample(n_iter=400)
    del A
    return v.conj().transpose(-2, -1)


def privately_select_topk_np(args, model, dataloader, criterion):
    k = args.num_eigenthings
    A = get_private_dataset_gradients_opacus(args, model, dataloader, criterion)
    A = (1 / A.shape[1]) * A.T @ A

    A = (A.shape[1] * args.epsilon / (2 * args.max_grad_norm * args.max_grad_norm)) * A
    A = A.cpu().numpy()
    v = MatrixBingham(A, k=k).sample(n_iter=4000)
    del A
    return v.conj().transpose(-2, -1)


def compute_topk_with_input_perturbation(args, model, dataloader, criterion):
    k = args.num_eigenthings
    c = args.max_grad_norm
    A = get_private_dataset_gradients_opacus(args, model, dataloader, criterion)
    m = A.shape[0]
    p = A.shape[1]
    A = (1 / m) * A.T @ A

    beta = ((((p + 1) * c) / (m * args.epsilon)) * np.sqrt(2 * np.log((p ** 2 + p) / (2 * args.delta * np.sqrt(2 * np.pi))))) + (c**2 / (m * np.sqrt(args.epsilon)))
    noise = torch.normal(0, beta ** 2, (p, p), device=A.device)
    noisy_A = A + noise
    eigenvals, eigenvectors = torch.linalg.eigh(noisy_A)
    return eigenvals[p - k:], eigenvectors[:, p - k:]
