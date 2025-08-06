import torch
import numpy as np
from torch.distributions import Distribution
from scipy.special import hyp1f1
from numpy.linalg import eig
from scipy.linalg import null_space
import linalg
import math


# from https://github.com/byu-aml-lab/optprime/
class ComplexBinghamSampler(object):
    """Sample from a Complex Bingham distribution.
    The pdf is of the form:
        f(z) = c(A)^{-1} exp(z^T A z), z \in CS^{k-1}
    where A is a k*k symmetric matrix, c(A) is a normalizing constant, and
    CS^{k-1} is the unit sphere in C^k.
    Methods based on: Kent, Constable, and Er.  Simulation for the complex
    Bingham distribution.  Statistics and Computing, 2004.  We skip Method 3,
    which is only useful in very limited circumstances (all lambdas equal
    to about 0.5).
    Parameters (either A or lambdas must be specified, but not both):
        A: the parameter matrix of the Bingham distribution
        lambdas: the first k-1 eigenvalues of -A (the smallest is assumed to
            be 0 and is not included in the list).
    """

    def __init__(self, A=None, lambdas=None, eigvecs=None):
        assert A is None or (lambdas is None and eigvecs is None)

        self._sampler = None
        self._eigvecs = eigvecs

        if A is not None:
            eigvals, self._eigvecs = linalg.eigh_swapped(-A)
            smallest_eig = eigvals[-1]
            lambdas = eigvals[:-1] - smallest_eig
        self._lambdas = lambdas

    def dual(self):
        """Return the Bingham(-A) sampler."""
        # Eigenvalues of -A sorted largest to smallest.
        eigvals = np.append(-self._lambdas, 0.0)[::-1]
        smallest_eig = eigvals[-1]
        lambdas = eigvals[:-1] - smallest_eig
        eigvecs = np.array(self._eigvecs[::-1])
        return ComplexBinghamSampler(lambdas=lambdas, eigvecs=eigvecs)

    def _pick_sampler(self):
        if any(l == 0 for l in self._lambdas):
            return self.sample_m2

        k = len(self._lambdas) + 1

        # From Table 1: expected number for M1 with p_T removed.
        m1 = math.log(k - 1)
        for lambda_j in self._lambdas:
            m1 += math.log(1 - math.exp(-lambda_j))

        # From Table 1: expected number for M2 with p_T removed.
        m2 = math.log(k)
        for lambda_j in self._lambdas:
            m2 += math.log(lambda_j)
        m2 -= math.lgamma((k - 1) + 1)

        if m1 < m2:
            return self.sample_m1
        else:
            return self.sample_m2

    def sample(self):
        if self._sampler is None:
            self._sampler = self._pick_sampler()
        return self._sampler()

    def sample_m1(self):
        """Sample using Method 1: Truncation to the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = [np.random.rand() for _ in range(k - 1)]
            s = [-(1 / l_j) * math.log(1 - u_j * (1 - math.exp(-l_j)))
                 for l_j, u_j in zip(self._lambdas, uniforms)]
            if sum(s) < 1:
                return self._convert_s_to_z(s)

    def sample_m2(self):
        """Sample using Method 2: Acceptance-rejection on the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = [np.random.rand() for _ in range(k - 1)]
            uniforms.sort()
            last = 0

            s = []
            for u in uniforms:
                s.append(u - last)
                last = u

            u = math.log(np.random.rand())
            if u < sum((-l_j * s_j) for l_j, s_j in zip(self._lambdas, s)):
                return self._convert_s_to_z(s)

    def _convert_s_to_z(self, s):
        """Convert a list of values on the simplex to values on the sphere."""
        s.append(1 - sum(s))
        s = np.array(s)
        z = s ** 0.5

        if self._eigvecs is not None:
            z = self._eigvecs.dot(z)
        return z


class ComplexBingham(Distribution):
    """Sample from a Complex Bingham distribution.
    The pdf is of the form:
        f(z) = c(A)^{-1} exp(z^T A z), z \in CS^{k-1}
    where A is a k*k symmetric matrix, c(A) is a normalizing constant, and
    CS^{k-1} is the unit sphere in C^k.
    Methods based on: Kent, Constable, and Er.  Simulation for the complex
    Bingham distribution.  Statistics and Computing, 2004.  We skip Method 3,
    which is only useful in very limited circumstances (all lambdas equal
    to about 0.5).
    Parameters (either A or lambdas must be specified, but not both):
        A: the parameter matrix of the Bingham distribution
        lambdas: the first k-1 eigenvalues of -A (the smallest is assumed to
            be 0 and is not included in the list).
    """

    def __init__(self, A=None, lambdas=None, eigvecs=None):
        super().__init__()
        assert A is None or (lambdas is None and eigvecs is None)

        self._sampler = None
        self._eigvecs = eigvecs

        self.device = A.device
        if A is not None:
            eigvals, eigvecs = torch.linalg.eigh(-A)
            argmin = eigvals.argmin()
            eigvals[argmin], eigvals[-1] = eigvals[-1], eigvals[argmin]

            # We can't do this all on one line because numpy slices are views.
            last_eigvec = eigvecs[:, -1]
            smallest_eigvec = eigvecs[:, argmin]
            eigvecs[:, argmin] = last_eigvec
            eigvecs[:, -1] = smallest_eigvec

            self._eigvecs = eigvecs
            smallest_eig = eigvals[-1]
            lambdas = eigvals[:-1] - smallest_eig
        self._lambdas = lambdas

    def dual(self):
        """Return the Bingham(-A) sampler."""
        # Eigenvalues of -A sorted largest to smallest.
        eigvals = np.append(-self._lambdas, 0.0)[::-1]
        smallest_eig = eigvals[-1]
        lambdas = eigvals[:-1] - smallest_eig
        eigvecs = np.array(self._eigvecs[::-1])
        return ComplexBinghamSampler(lambdas=lambdas, eigvecs=eigvecs)

    def _pick_sampler(self):
        if any(l == 0 for l in self._lambdas):
            return self.sample_m2

        k = len(self._lambdas) + 1

        # From Table 1: expected number for M1 with p_T removed.
        m1 = math.log(k - 1)
        for lambda_j in self._lambdas:
            m1 += math.log(1 - math.exp(-lambda_j))

        # From Table 1: expected number for M2 with p_T removed.
        m2 = math.log(k)
        for lambda_j in self._lambdas:
            m2 += math.log(lambda_j)
        m2 -= math.lgamma((k - 1) + 1)

        if m1 < m2:
            return self.sample_m1
        else:
            return self.sample_m2

    def sample(self):
        if self._sampler is None:
            self._sampler = self._pick_sampler()
        return self._sampler()

    def sample_m1(self):
        """Sample using Method 1: Truncation to the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = torch.rand((k - 1,), device=self.device)
            s = -(1 / self._lambdas) * torch.log(1 - uniforms * (1 - torch.exp(-self._lambdas)))
            if s.sum() < 1:
                return self._convert_s_to_z(s)

    def sample_m2(self):
        """Sample using Method 2: Acceptance-rejection on the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = torch.rand((k - 1,), device=self.device)
            uniforms, _ = torch.sort(uniforms)

            last = torch.tensor([0], device=self.device)
            s = uniforms - torch.cat((last, uniforms[:-1]))

            u = torch.log(torch.rand((1,), device=self.device))
            if u < torch.sum(-self._lambdas * s):
                return self._convert_s_to_z(s)

    def _convert_s_to_z(self, s):
        """Convert a list of values on the simplex to values on the sphere."""
        s = torch.cat((s, torch.tensor([1 - s.sum()], device=self.device)))
        z = s ** 0.5
        z = z.reshape(-1, 1)
        if self._eigvecs is not None:
            z = self._eigvecs @ z
        return z


def torch_null_space(A, rcond=None):
    u, s, vh = torch.linalg.svd(A)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(M, N)
    tol = torch.max(s) * rcond
    num = torch.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


class MatrixBingham(Distribution):
    def __init__(self, A, k=2):
        super(MatrixBingham, self).__init__()
        self.A = A
        self.k = k
        self.p = A.shape[0]

    def sample(self, n_iter=2):
        """Gibbs sampling"""
        k = self.k
        p = self.p
        A = self.A
        V = np.ones((k, p))
        for i in range(n_iter):
            for r in range(k):
                N = null_space(np.delete(V, r, 0))
                An = N.T @ A @ N
                vector_bingham = ComplexBinghamSampler(A=An).sample()
                V[r] = (N @ vector_bingham).flatten()
        return V


class MatrixBingham_EX(Distribution):
    def __init__(self, A, k=2):
        super(MatrixBingham_EX, self).__init__()
        self.A = A
        self.k = k
        self.p = A.shape[0]
        self.device = A.device

    def sample(self, n_iter=2):
        """Gibbs sampling"""
        k = self.k
        p = self.p
        A = self.A
        V = torch.ones((k, p), device=self.device)
        for i in range(n_iter):
            for r in range(k):
                N = torch_null_space(delete(V, r, 0))
                An = N.T @ A @ N
                vector_bingham = ComplexBingham(A=An).sample()
                V[r] = (N @ vector_bingham).flatten()
        return V


if __name__ == '__main__':
    g = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    a = g.T @ g

    # v = np.ones((2, 4))
    # v1 = np.delete(v, -1, 0)
    # n = null_space(v1)
    # print(a)

    print(MatrixBingham(a, 2).sample(n_iter=10000))

    gt = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
    at = gt.T @ gt

    print(MatrixBingham_EX(at, 2).sample(n_iter=10000))
