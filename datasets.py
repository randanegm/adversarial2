import os
import logging

import numpy
from theano.compat.six.moves import xrange

from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.utils import contains_nan, serial, string_utils


_logger = logging.getLogger(__name__)


class CIFAR10OneHot(CIFAR10):
    def __init__(self, which_set, onehot_dtype='uint8',
                 center=False, rescale=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro=False, preprocessor=None):
        """Modified version of the CIFAR10 constructor which creates Y
        as one-hot vectors rather than simple indexes. This is super
        hacky. Sorry, Guido.."""

        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype = 'uint8'
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest = 10000

        # we also expose the following details:
        self.img_shape = (3, 32, 32)
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 10
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # prepare loading
        fnames = ['data_batch_%i' % i for i in range(1, 6)]
        datasets = {}
        datapath = os.path.join(
            string_utils.preprocess('${PYLEARN2_DATA_PATH}'),
            'cifar10', 'cifar-10-batches-py')
        for name in fnames + ['test_batch']:
            fname = os.path.join(datapath, name)
            if not os.path.exists(fname):
                raise IOError(fname + " was not found. You probably need to "
                              "download the CIFAR-10 dataset by using the "
                              "download script in "
                              "pylearn2/scripts/datasets/download_cifar10.sh "
                              "or manually from "
                              "http://www.cs.utoronto.ca/~kriz/cifar.html")
            datasets[name] = cache.datasetCache.cache_file(fname)

        lenx = numpy.ceil((ntrain + nvalid) / 10000.) * 10000
        x = numpy.zeros((lenx, self.img_size), dtype=dtype)
        y = numpy.zeros((lenx, 1), dtype=dtype)

        # load train data
        nloaded = 0
        for i, fname in enumerate(fnames):
            _logger.info('loading file %s' % datasets[fname])
            data = serial.load(datasets[fname])
            x[i * 10000:(i + 1) * 10000, :] = data['data']
            y[i * 10000:(i + 1) * 10000, 0] = data['labels']
            nloaded += 10000
            if nloaded >= ntrain + nvalid + ntest:
                break

        # load test data
        _logger.info('loading file %s' % datasets['test_batch'])
        data = serial.load(datasets['test_batch'])

        # process this data
        Xs = {'train': x[0:ntrain],
              'test': data['data'][0:ntest]}

        Ys = {'train': y[0:ntrain],
              'test': data['labels'][0:ntest]}

        X = numpy.cast['float32'](Xs[which_set])

        y = Ys[which_set]
        if isinstance(y, list):
            y = numpy.asarray(y).astype(dtype)
        if which_set == 'test':
            assert y.shape[0] == 10000
            y = y.reshape((y.shape[0], 1))

        formatter = OneHotFormatter(self.n_classes, dtype=onehot_dtype)
        y = formatter.format(y, mode='concatenate')

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale

        if toronto_prepro:
            assert not center
            assert not gcn
            X = X / 255.
            if which_set == 'test':
                other = CIFAR10(which_set='train')
                oX = other.X
                oX /= 255.
                X = X - oX.mean(axis=0)
            else:
                X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
            assert X.shape[0] == y.shape[0]

        if which_set == 'test':
            assert X.shape[0] == 10000

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)

        super(CIFAR10, self).__init__(X=X, y=y, view_converter=view_converter,
                                      )#y_labels=self.n_classes)

        assert not contains_nan(self.X)

        if preprocessor:
            preprocessor.apply(self)

        # Another hack: rename 'targets' to match model expectations
        space, (X_source, y_source) = self.data_specs
        self.data_specs = (space, (X_source, 'condition'))
