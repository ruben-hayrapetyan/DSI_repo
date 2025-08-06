"""Miscellaneous linear algebra operations.
For numpy arrays vs. matrices, see:
    http://www.scipy.org/NumPy_for_Matlab_Users#head-e9a492daa18afcd86e84e07cd2824a9b1b651935
"""

from __future__ import division, print_function

import math
import random

try:
    import numpy as np
except ImportError:
    import numpy as np


def product_trace(A, B):
    """Find the trace of the matrix product of A and B.
    This is much faster than finding `numpy.linalg.trace(A.dot(B))`.
    See the following for more information about ways to compute this:
    http://stackoverflow.com/questions/18854425/what-is-the-best-way-to-compute-the-trace-of-a-matrix-product-in-numpy
    """
    A = np.asarray(A)
    B = np.asarray(B)
    # return (A * B.T).sum()
    return np.einsum('ij,ji->', A, B)


def eigh_sorted(A):
    """Performs numpy.linalg.eigh and sorts the resulting eigenvalues/vectors.
    Note that sorting is from large to small.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def eigh_swapped(A):
    """Perform numpy.linalg.eigh; swap the smallest eigenvalue with the last.
    Rearrange the eigenvectors accordingly.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    argmin = eigvals.argmin()
    eigvals[argmin], eigvals[-1] = eigvals[-1], eigvals[argmin]

    # We can't do this all on one line because numpy slices are views.
    last_eigvec = np.array(eigvecs[:, -1])
    smallest_eigvec = np.array(eigvecs[:, argmin])
    eigvecs[:, argmin] = last_eigvec
    eigvecs[:, -1] = smallest_eigvec
    return eigvals, eigvecs


def rand_o_matrix(n, rand=None):
    """Creates a random Haar-distributed orthonormal matrix.
    Uses the Stewart algorithm.
    """
    if rand is None:
        rand = random

    # `A` holds the orthonormal matrix that is built inductively (starting
    # from [[1.0]] or [[-1.0]], which are the only 1x1 orthonormal matrices.
    A = np.zeros((n, n))
    A[0, 0] = rand.randrange(-1, 2, 2)

    # Identity matrices can come in handy.
    I = np.eye(n)

    # `x` holds an n-dimensional vector (an nx1 matrix) that will hold normal
    # vectors that will define planes in successively larger subspaces.
    x = np.zeros((n, 1))

    for i in range(1, n):
        # The m-dimensional subspace (in the top-left corner) is the relevant
        # space during the current iteration.
        m = i + 1

        # Extend A to m dimensions by adding a new orthogonal basis vector.
        # The value is either [0, 0, ..., 1] or [0, 0, ..., -1] (flipped).
        # If you don't ever flip, then the algorithm only produces
        # reflection+rotations (never pure rotations).
        A[i, i] = rand.randrange(-1, 2, 2)

        # Numpy "broadcasting" converts a single-axis array into an mx1 array,
        # which then gets saved to the first m slots of x.
        x[:m] = rand_norm_array(i + 1, rand)[:, None]

        # Reflect A across the plane defined by x (this is a Householder
        # transformation).
        householder_mat = I - 2 * np.dot(x, x.T)
        A = np.dot(householder_mat, A)

    return A


def rand_norm_array(n, rand=None):
    """Creates a random normalized array of given size."""
    if rand is None:
        rand = random

    samples = np.empty(n, 'float')
    for i in range(n):
        samples[i] = rand.normalvariate(0, 1)
    normalization = (samples ** 2).sum() ** 0.5
    return samples / normalization


def rand_cliques_matrix(n, m, rand=None):
    """Create an nxn orthogonal matrix with m random orthogonal submatrices.
    Note that the matrix is returned as a numpy array (not a numpy matrix).
    """
    if rand is None:
        rand = random

    result = np.zeros((n, n))
    min_block_size = n // m
    num_big_blocks = n % m

    index = 0
    for block_id in range(m):
        if block_id < num_big_blocks:
            N = min_block_size + 1
        else:
            N = min_block_size

        submatrix = rand_o_matrix(N, rand)
        result[index:index + N, index:index + N] = submatrix

        index += N

    return result


def rand_perm_matrix(n, rand=None):
    """Create a random permutation matrix.
    Note that the matrix is returned as a numpy array (not a numpy matrix).
    """
    if rand is None:
        rand = random

    variables = list(range(n))
    rand.shuffle(variables)
    A = np.zeros((n, n))
    for i, j in enumerate(variables):
        A[i, j] = 1
    return A


def orthogonalize(x, A):
    """Orthogonalize the vector x relative to A, a matrix of column vectors.
    The input is expected to be a numpy array (not a numpy matrix).  The
    array A need not be an orthonormal matrix.
    Based on the Modified Gram-Schmidt algorithm (numerically stable).
    """
    v = x.copy()
    q, _ = np.linalg.qr(A)
    for u in q.T:
        # Subtract out the projection of v onto u.
        v -= np.dot(v, u) * u
    # Normalize.
    v /= np.sqrt(np.dot(v, v))
    return v


def ladd(log_x, log_y):
    """Add two numbers in log space."""

    if log_y > log_x:
        # log_x smaller gives more precision
        log_x, log_y = log_y, log_x
    if math.isinf(log_x):
        return log_x
    return log_x + math.log(1 + math.exp(log_y - log_x))


def chol_update(L, x):
    """Rank 1 update of the lower-triangular Cholesky factor L.
    L is modified in-place.
    """
    x = x.copy()
    for k in range(len(x)):
        r = (L[k, k] ** 2 + x[k] ** 2) ** 0.5
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k + 1:, k] = (L[k + 1:, k] + s * x[k + 1:]) / c
        x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]


def chol_downdate(L, x):
    """Rank 1 downdate of the lower-triangular Cholesky factor L.
    L is modified in-place.
    """
    x = x.copy()
    for k in range(len(x)):
        r_squared = L[k, k] ** 2 - x[k] ** 2
        if r_squared <= 0.0:
            raise NonPosDefError()
        r = r_squared ** 0.5
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k + 1:, k] = (L[k + 1:, k] - s * x[k + 1:]) / c
        x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]


class NonPosDefError(Exception):
    pass


def area(n):
    """Surface area of an n-1 sphere (a sphere in an n-dimensional space)."""
    log_s = math.log(math.pi) * n / 2 - math.lgamma(n / 2)
    return 2 * math.exp(log_s)


def log_area(n):
    """Log of surface area of an n-1 sphere (a sphere in an n-D space)."""
    return math.log(2) + math.log(math.pi) * n / 2 - math.lgamma(n / 2)


def degrees_between(u, v):
    """Return the angle between the input vectors.
    The result is in degrees (0 to 180).
    Note:
    degrees_between_unsigned(u, v) == degrees_between_unsigned(v, u)
    """
    return math.acos(np.dot(u, v)) * 180 / math.pi


def degrees_between_unsigned(u, v):
    """Return the angle between the lines defined by the input vectors.
    The result is in degrees (0 to 90).
    Note:
    degrees_between_unsigned(u, v) == degrees_between_unsigned(v, u)
    degrees_between_unsigned(u, v) == degrees_between_unsigned(u, -v)
    """
    return math.acos(abs(np.dot(u, v))) * 180 / math.pi

# vim: et sw=4 sts=4
