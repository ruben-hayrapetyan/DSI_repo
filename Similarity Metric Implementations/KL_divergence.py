import numpy as np


def entropy(p):
    """Compute the Shannon entropy of a distribution.

    The Shannon entropy is defined as follows
    :math:`\sum_x p(x_i) * \log p(x_i)`.

    Parameters
    ----------
    p : np.ndarray

    Examples
    --------
    >>> distribution = np.array([0.25, 0.25, 0.5])
    >>> entropy(distribution)
    1.5

    """
    assert isinstance(p, np.ndarray), '`p` must be a numpy array'
    assert np.isclose(np.sum(p), 1.), '`p` must be a probability distribution'

    p = p[np.nonzero(p)]

    return np.sum(-p * np.log2(p))


def kl_divergence(p, q):
    """Compute the Kullback-Leibler divergence between two distributions.

    The KL divergence is defined as
    :math:`D_{KL}(p, q) = \sum_x p(x_i) * (\log p(x_i) - \log q(x_i))`
    which can be rewritten as
    :math:`D_{KL}(p, q) = \sum_x p(x_i) * \log \frac{p(x_i)}{q(x_i)}`
    and is computationally more conventient.

    Some interesting properties of the KL divergence:
      - The KL divergence is always non-negative, i.e.
        :math:`D_{KL}(p, q) \geq 0`.

      - The KL divergence is additive for independent distributions, i.e.
        :math:`D_{KL}(P, Q) = D_{KL}(P_1, Q_1) + D_{KL}(P_2, Q_2)`.

    Parameters
    ----------
    p : np.ndarray
    q : np.ndarray

    Examples
    --------
    >>> p = np.array([0.7, 0.2, 0.05, 0.05])
    >>> q = np.array([0.05, 0.05, 0.2, 0.7])
    >>> kl_divergence(p, q)
    2.77478069934

    """
    assert isinstance(p, np.ndarray), '`p` must be a numpy array'
    assert np.isclose(np.sum(p), 1.), '`p` must be a probability distribution'

    assert isinstance(q, np.ndarray), '`q` must be a numpy array'
    assert np.isclose(np.sum(q), 1.), '`q` must be a probability distribution'

    # Define the zero masks for P and Q and ignore them during computation
    q_mask, p_mask = q == 0, p == 0
    # The implication `p => q` is equivalent to `not p or q`
    assert all(~q_mask | p_mask), 'The KL divergence is defined iif Q(x)=0 implies P(x)=0'
    p, q = p[~p_mask], q[~q_mask]

    return np.sum(p * np.log2(p / q))


def symmetric_kl_divergence(p, q):
    """The symmetric Kullback-Leibler divergence.

    Kullback and Leibler themselves defined the symmetric divergence as
    :math:`D_{KL}(p, q) + D_{KL}(q, p)`.

    """
    return kl_divergence(p, q) + kl_divergence(q, p)