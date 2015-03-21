
import numpy as np
from pylearn2.models.mlp import MLP
from pylearn2.space import VectorSpace
import theano
import theano.tensor as T

from adversarial.rnn import GenerativeRecurrentLayer


def test_rnn_generator():
    n_steps = 10
    generator = MLP(input_space=VectorSpace(dim=1),
                    layers=[GenerativeRecurrentLayer(layer_name='recurrent',
                                                     max_steps=n_steps,
                                                     dim=1)])

    generator.layers[0].get_params()[0].set_value(np.ones((1, 1)))
    generator.layers[0].get_params()[1].set_value(np.ones((1, 1)))

    inp = T.matrix()
    f = theano.function([inp], generator.fprop(inp))

    expected = []
    x = 1
    for i in range(n_steps):
        x = np.tanh(x)
        expected.append(x)

    np.testing.assert_array_almost_equal(f(np.ones((1, 1))).flatten(),
                                         expected)
