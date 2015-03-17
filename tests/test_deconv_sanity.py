"""
Simple sanity check for deconvolution layer.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from pylearn2.models.mlp import MLP
from pylearn2.space import Conv2DSpace
import theano

from adversarial.deconv import Deconv


input_space = Conv2DSpace(shape=(2, 1), num_channels=16, axes=('c', 0, 1, 'b'))

deconv = Deconv(layer_name='deconv',
                num_channels=1,
                kernel_shape=(4, 4),
                output_stride=(2, 2),
                irange=0.)

mlp = MLP(input_space=input_space, layers=[deconv])

X = input_space.get_theano_batch()
f = theano.function([X], mlp.fprop(X))

# Construct dummy filters.
# Just use two for simplicity.
filter1 = np.array([[0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0]])
filter2 = np.array([[-1, 0, -1, 0],
                    [0, -1, 0, -1],
                    [-1, 0, -1, 0],
                    [0, -1, 0, -1]])

filters_dest = deconv.transformer._filters
new_filters = np.zeros((16, 4, 4), dtype=filters_dest.dtype)
new_filters[0] = filter1
new_filters[1] = filter2
new_filters = new_filters.reshape(16, 4, 4, 1).swapaxes(0, 3)
deconv.transformer._filters.set_value(new_filters)


def test_deconv_simple():
    # Now try a feedforward
    input = np.zeros((16, 2, 1, 1), dtype=filters_dest.dtype)
    input[0, 0, 0, 0] = 1
    input[1, 0, 0, 0] = -0.5
    input[0, 1, 0, 0] = 2
    input[1, 1, 0, 0] = 1
    deconvolution = f(input).reshape((6, 4))

    # Above deconvolution should be equivalent to overlapping the two below
    # layers (each layer produced from one kernel-wise slice of the input
    # layer)
    out0 = np.concatenate([1 * filter1 - 0.5 * filter2, np.zeros((2, 4))])
    out1 = np.concatenate([np.zeros((2, 4)), 2 * filter1 + 1 * filter2])
    check = out0 + out1

    assert_array_almost_equal(deconvolution, check)
