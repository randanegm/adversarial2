"""Defines distributions from which to sample conditional data."""

import numpy as np
from pylearn2.format.target_format import OneHotFormatter
import theano
from theano.tensor.shared_randomstreams import RandomStreams


class Distribution(object):
    def __init__(self, space):
        self.space = space

    def get_space(self):
        return self.space

    def sample(self, n):
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
