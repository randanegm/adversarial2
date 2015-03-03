"""Defines distributions from which to sample conditional data."""

import numpy as np
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.space import VectorSpace
import theano
from theano.tensor.shared_randomstreams import RandomStreams


class Distribution(object):
    def __init__(self, space):
        self.space = space

    def get_space(self):
        return self.space

    def get_total_dimension(self):
        return self.space.get_total_dimension()

    def sample(self, n):
        """
        Parameters
        ----------
        n : integer
            Number of samples to generate

        Returns
        -------
        samples : ndarray of shape (n, num_features)
        """

        raise NotImplementedError("abstract method")


class OneHotDistribution(Distribution):
    """Randomly samples from a distribution of one-hot vectors."""

    def __init__(self, space, rng=None):
        super(OneHotDistribution, self).__init__(space)

        self.dim = space.get_total_dimension()
        self.formatter = OneHotFormatter(self.dim, dtype=space.dtype)

        self.rng = RandomStreams() if rng is None else rng

    def sample(self, n):
        idxs = self.rng.random_integers((n, 1), low=0, high=self.dim - 1)
        return self.formatter.theano_expr(idxs, mode='concatenate')


class KernelDensityEstimateDistribution(Distribution):
    """Randomly samples from a kernel density estimate yielded by a set
    of training points.

    Simple sampling procedure [1]:

    1. With training points $x_1, ... x_n$, sample a point $x_i$
       uniformly
    2. From original KDE, we have a kernel defined at point $x_i$;
       sample randomly from this kernel

    [1]: http://www.stat.cmu.edu/~cshalizi/350/lectures/28/lecture-28.pdf
    """

    def __init__(self, X, bandwidth=1, space=None):
        """
        Parameters
        ----------
        X : ndarray of shape (num_examples, num_features)
            Training examples from which to generate a kernel density
            estimate

        bandwidth : float
            Bandwidth (or h, or sigma) of the generated kernels
        """

        assert X.ndim == 2
        if space is None:
            space = VectorSpace(dim=X.shape[1], dtype=X.dtype)

        super(KernelDensityEstimateDistribution, self).__init__(space)

        self.X = X
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

        self.bandwidth = bandwidth
        self.cov_matrix = bandwidth * np.eye(self.num_features)

    def sample(self, n, out=None):
        if out is None:
            out = np.empty((n, self.num_features))

        # Sample $n$ training examples
        training_samples = self.X[np.random.choice(self.num_examples, n, replace=True)]

        # Sample individually from each selected associated kernel
        for i, sample in enumerate(training_samples):
            out[i] = np.random.multivariate_normal(sample, self.cov_matrix)

        return out
