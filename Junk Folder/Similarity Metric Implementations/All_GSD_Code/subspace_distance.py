import torch
from torch.nn.functional import normalize
from scipy.sparse.linalg import LinearOperator

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import svds

from warnings import warn


def lnorm_subspace_closeness(eigenvecs_priv, eigenvecs_pub):
    """
        eigenvecs_priv: p x k numpy matrix
    """
    l1_norm = 0
    for i in range(eigenvecs_priv.shape[0]):
        column_sum = 0
        for j in range(eigenvecs_priv.shape[0]):
            subspace_priv = 0
            subspace_pub = 0
            for k in range(eigenvecs_priv.shape[1]):
                subspace_priv += eigenvecs_priv[i][k] * eigenvecs_priv[j][k]
                subspace_pub += eigenvecs_pub[i][k] * eigenvecs_pub[j][k]
            column_sum += torch.abs(subspace_priv - subspace_pub)
        if column_sum > l1_norm:
            l1_norm = column_sum
    return l1_norm


def gr_subspace_distance(eigenvecs_priv, eigenvecs_pub):
    """
        eigenvecs_priv: k x p torch matrix
        eigenvecs_pub: k x p torch matrix
    """
    eigenvecs_priv = normalize(eigenvecs_priv)
    eigenvecs_pub = normalize(eigenvecs_pub)
    cos_angle = torch.einsum("ij, ij->i", eigenvecs_priv, eigenvecs_pub)
    return torch.sqrt(torch.sum(torch.arccos(cos_angle) ** 2))


def projection_subspace_distance(eigenvecs_priv, eigenvecs_pub):
    """
        eigenvecs_priv: k x p torch matrix
        eigenvecs_pub: k x p torch matrix
    """
    k = eigenvecs_priv.shape[0]
    eigenvecs_priv = normalize(eigenvecs_priv)
    eigenvecs_pub = normalize(eigenvecs_pub)
    principle_angles = torch.einsum("ik,jk->ij", eigenvecs_priv, eigenvecs_pub)
    principle_angles = principle_angles.flatten()
    principle_angles, indices = torch.sort(principle_angles)
    principle_angles = principle_angles[-k:]

    return torch.sqrt(k - torch.sum(principle_angles ** 2)) / np.sqrt(k)


def projection_subspace_distance_svd(eigenvecs_priv, eigenvecs_pub):
    """
        eigenvecs_priv: k x p torch matrix
        eigenvecs_pub: k x p torch matrix
    """
    k = eigenvecs_priv.shape[0]
    eigenvecs_priv = normalize(eigenvecs_priv, dim=0)
    eigenvecs_pub = normalize(eigenvecs_pub, dim=0)

    eigenvecs_space = eigenvecs_priv @ eigenvecs_pub.T
    U, principle_angles, Vh = torch.linalg.svd(eigenvecs_space)
    print(principle_angles)
    return torch.sqrt(k - torch.sum(principle_angles ** 2)) / np.sqrt(k)


def l2_subspace_distance(eigenvecs_priv, eigenvecs_pub):
    """
        eigenvecs_priv: p x k torch matrix
        eigenvecs_pub: p x k torch matrix
    """
    p = eigenvecs_priv.shape[0]

    def _matvec(x):
        return eigenvecs_priv @ (eigenvecs_priv.T @ x) - eigenvecs_pub @ (eigenvecs_pub.T @ x)

    def _rmatvec(x):
        return eigenvecs_priv @ (eigenvecs_priv.T @ x) - eigenvecs_pub @ (eigenvecs_pub.T @ x)

    A = LinearOperator((p, p), matvec=_matvec, rmatvec=_rmatvec)
    u2, s2, vT2 = svds(A, k=1)
    return s2


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """

    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)


class MMtOperator(Operator):
    """
    Linear operator for matrix * matrix.T - vector product
    """

    def __init__(self, matrix1, matrix2):
        """
        Args:
            matrix1: p x k
        """
        size = matrix1.shape[0]
        super(MMtOperator, self).__init__(size)
        self.matrix1 = matrix1
        self.matrix2 = matrix2

    def apply(self, vec):
        """
        Args:
            vec: p * 1
        """
        return self.matrix1 @ (self.matrix1.T @ vec) - self.matrix2 @ (self.matrix2.T @ vec)


def matrix_l2_norm(
        matrix1: torch.Tensor,
        matrix2: torch.Tensor,
        num_eigenthings: int = 1,
        num_lanczos_vectors: int = None,
        device: str = "mps"
):
    operator = MMtOperator(matrix1, matrix2)
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = np.float32(x)
        x = torch.from_numpy(x)
        x = x.to(device)
        out = operator.apply(x)
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, matvec=_scipy_apply, rmatvec=_scipy_apply)

    u2, s2, vT2 = svds(scipy_op, k=num_eigenthings)

    return s2


def principle_angle(v1, v2):
    u, s, vh = torch.linalg.svd(v1.conj().transpose(-2, -1) @ v2)
    return s


def projection_metric(v1, v2):
    angles = principle_angle(v1, v2)
    return torch.sqrt(len(angles) - torch.sum(angles ** 2)) / np.sqrt(len(angles))


def projection_metric_dp(v1, v2):
    angles = principle_angle(v1, v2)
    return torch.sqrt(len(angles) - torch.sum(angles ** 2))
