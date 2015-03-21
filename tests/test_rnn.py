
import numpy as np
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.space import VectorSpace
import theano
import theano.tensor as T

from adversarial.rnn import GenerativeRecurrentMLP


def test_rnn_generator():
    # hidden->hidden nested MLP
    hidden_mlp = MLP(layer_name='hidden->hidden',
                     layers=[Tanh(layer_name='hidden->hidden_tanh', irange=0., dim=1)])



    # hidden->output nested MLP
    output_mlp = MLP(layer_name='hidden->output',
                     layers=[Tanh(layer_name='hidden->output_tanh', irange=0., dim=1)])

    n_steps = 10
    generator = GenerativeRecurrentMLP(input_space=VectorSpace(dim=1),
                                       hidden_mlp=hidden_mlp,
                                       output_mlp=output_mlp,
                                       max_steps=n_steps,
                                       irange=0.01,
                                       num_classes=100)

    hidden_mlp.layers[0].get_params()[0].set_value(np.ones((2, 1)))
    output_mlp.layers[0].get_params()[0].set_value(np.ones((1, 1)))

    generator.get_params()[1].set_value(np.ones((1, 1)))

    inp = T.matrix()
    f = theano.function([inp], generator.fprop(inp))

    outputs, hiddens = f(np.ones((1, 1)))

    expected_hiddens = []
    x = np.tanh(2)
    for out in outputs:
        expected_hiddens.append(x)
        x = np.tanh(x + out)

    np.testing.assert_array_almost_equal(hiddens.flatten(),
                                         expected_hiddens)
