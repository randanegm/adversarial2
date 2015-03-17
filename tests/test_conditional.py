import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.models.mlp import MLP, Linear
from pylearn2.space import VectorSpace
import theano
from theano import tensor as T

from adversarial.conditional import ConditionalGenerator
from adversarial.distributions import OneHotDistribution


class ConditionalGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.noise_dim = 10
        self.num_labels = 10

        self.condition_dtype = 'uint8'
        self.condition_space = VectorSpace(dim=self.num_labels, dtype=self.condition_dtype)
        self.condition_formatter = OneHotFormatter(self.num_labels, dtype=self.condition_dtype)
        self.condition_distribution = OneHotDistribution(self.condition_space)

        # TODO this nvis stuff is dirty. The ConditionalGenerator should handle it
        self.mlp_nvis = self.noise_dim + self.num_labels
        self.mlp_nout = 1

        # Set up model
        self.mlp = MLP(nvis=self.mlp_nvis, layers=[Linear(self.mlp_nout, 'out', irange=0.1)])
        self.G = ConditionalGenerator(input_condition_space=self.condition_space,
                                      condition_distribution=self.condition_distribution,
                                      noise_dim=self.noise_dim,
                                      mlp=self.mlp)

    def test_conditional_generator_input_setup(self):
        """Check that conditional generator correctly sets up composite
        input layer."""

        # Feedforward: We want the net to ignore the noise and simply
        # convert the one-hot vector to a number
        weights = np.concatenate([np.zeros((self.mlp_nout, self.noise_dim)),
                                  np.array(range(self.num_labels)).reshape((1, -1)).repeat(self.mlp_nout, axis=0)],
                                 axis=1).T.astype(theano.config.floatX)
        self.mlp.layers[0].set_weights(weights)

        inp = (T.matrix(), T.matrix(dtype=self.condition_dtype))
        f = theano.function(inp, self.G.mlp.fprop(inp))

        assert_array_equal(
            f(np.random.rand(self.num_labels, self.noise_dim).astype(theano.config.floatX),
              self.condition_formatter.format(np.array(range(self.num_labels)))),
            np.array(range(self.num_labels)).reshape(self.num_labels, 1))

    def test_sample_noise(self):
        """Test barebones noise sampling."""

        n = T.iscalar()
        cond_inp = self.condition_distribution.sample(n)
        sample_and_noise = theano.function([n], self.G.sample_and_noise(cond_inp, all_g_layers=True)[1])

        print sample_and_noise(15)
