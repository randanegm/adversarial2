
from collections import OrderedDict

import numpy as np
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
import theano
import theano.tensor as T

from adversarial.rnn import GenerativeRecurrentMLP


def test_rnn_generator():
    # hidden->hidden nested MLP
    hidden_mlp = MLP(layer_name='hidden->hidden',
                     layers=[Tanh(layer_name='hidden->hidden_tanh', irange=1., dim=1)])



    # hidden->output nested MLP
    output_mlp = MLP(layer_name='hidden->output',
                     layers=[Tanh(layer_name='hidden->output_tanh', irange=1., dim=1)])

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

    outputs, hiddens, idxs = f(np.ones((1, 1)))

    expected_hiddens = []
    x = np.tanh(2)
    for out in outputs:
        expected_hiddens.append(x)
        x = np.tanh(x + out)

    np.testing.assert_array_almost_equal(hiddens.flatten(),
                                         expected_hiddens)


def test_learn_abc():
    """Can we learn the ABCs?

    Make sure the generative recurrent MLP can learn to output a sequence "abc."

    Train with cross-entropy loss at each of the softmax outputs."""

    # hidden->hidden nested MLP
    hidden_mlp = MLP(layer_name='hidden->hidden',
                     layers=[Tanh(layer_name='hidden->hidden_tanh', irange=1., dim=1)])


    # hidden->output nested MLP
    output_mlp = MLP(layer_name='hidden->output',
                     layers=[Tanh(layer_name='hidden->output_tanh', irange=1., dim=1)])

    embedding_n = 3
    n_letters = 3
    n_steps = 3

    generator = GenerativeRecurrentMLP(input_space=VectorSpace(dim=embedding_n),
                                       hidden_mlp=hidden_mlp,
                                       output_mlp=output_mlp,
                                       max_steps=n_steps,
                                       irange=0.1,
                                       num_classes=n_letters)

    emb_idxs = T.ivector()
    correct = T.imatrix()

    def get_cost(inp, correct, learning_rate=0.1):
        outputs, hiddens, softmaxes = generator.fprop(inp)

        def cost_t(softmax_t, correct_t, cost_acc):
            cost = T.nnet.categorical_crossentropy(softmax_t, correct_t).sum()
            updates = {W: W - learning_rate * T.grad(cost, wrt=W, disconnected_inputs='warn')
                       for W in generator.get_params()}

            return cost_acc + cost, updates

        cost_acc = T.zeros((1,))
        cost, updates = theano.scan(cost_t, outputs_info=[cost_acc], sequences=[softmaxes, correct])

        return cost, updates

    cost, updates = get_cost(generator.E[emb_idxs], correct)
    f = theano.function([emb_idxs, correct], cost, updates=updates)

    ## RUN ##

    idxs = np.cast['int32'](np.array([0]))
    correct = np.cast['int32'](np.array([[1]]))

    for x in range(5):
        cost = f(idxs, correct)
        print cost
