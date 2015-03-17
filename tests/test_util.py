import numpy as np
from numpy.testing import assert_array_almost_equal

from pylearn2.models.mlp import MLP
from theano import tensor as T
import theano

from adversarial import util


def test_identity_layer():
    nvis = 10

    mlp = MLP(nvis=nvis, layers=[util.IdentityLayer(layer_name='ident')])

    X = T.matrix()
    f = theano.function([X], mlp.fprop(X))

    for _ in range(5):
        X = np.random.rand(10, nvis).astype(theano.config.floatX)
        yield _test_identity_layer, f, X


def _test_identity_layer(mlp_fun, X):
    assert_array_almost_equal(mlp_fun(X), X)
