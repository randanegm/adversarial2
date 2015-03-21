
from collections import OrderedDict
import functools

import numpy as np
from pylearn2.models.mlp import Layer
from pylearn2.utils import sharedX
import scipy
import theano
import theano.tensor as T


class GenerativeRecurrentLayer(Layer):
    def __init__(self, dim, layer_name, max_steps,
                 irange=0.1, init_bias=0., nonlinearity=T.tanh,
                 *args, **kwargs):
        self.dim = dim
        self.layer_name = layer_name
        self.max_steps = max_steps
        self.irange = irange
        self.init_bias = init_bias
        self.nonlinearity = nonlinearity

        self._scan_updates = OrderedDict()

        super(GenerativeRecurrentLayer, self).__init__(*args, **kwargs)

    @functools.wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self._params = self._init_params()

    def _init_params(self):
        input_dim = self.input_space.dim
        rng = self.mlp.rng

        # W: input -> hidden
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim))

        # U: hidden(n) -> hidden(n+1)
        U = rng.randn(self.dim, self.dim)
        U, _ = scipy.linalg.qr(U)

        # bias
        b = np.zeros((self.dim,)) + self.init_bias

        return [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b, name=(self.layer_name + '_b'))
        ]

    @functools.wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        updates.update(self._scan_updates)

    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)

        W, U, b = self._params

        # First map input -> hidden
        state_start = T.dot(state_below, W) + b

        # Now scan!
        z, updates = theano.scan(fn=self.fprop_step,
                                 outputs_info=state_start,
                                 n_steps=self.max_steps)

        self._scan_updates.update(updates)
        return z

    def fprop_step(self, state_before):
        # TODO stop condition
        _, U, _ = self._params
        return self.nonlinearity(T.dot(state_before, U))
