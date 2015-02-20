
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import OneHotFormatter


class CIFAR10OneHot(CIFAR10):
    def __init__(self, which_set, onehot_dtype='uint8',
                 center=False, rescale=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None):
        super(CIFAR10OneHot, self).__init__(which_set, center=center, rescale=rescale,
                                            gcn=gcn, start=start, stop=stop, axes=axes,
                                            toronto_prepro=toronto_prepro,
                                            preprocessor=preprocessor)

        # OK, mess with the target "Y" matrix and then re-init
        #
        # This is hacky, but the alternative is to copy-paste the entire
        # CIFAR10 constructor and change just a few lines.
        y = self.y

        formatter = OneHotFormatter(self.y_labels, dtype=onehot_dtype)
        y = formatter.format(y, mode='concatenate')

        # So hacky. I am sorry, Guido..
        DenseDesignMatrix.__init__(self, X=self.X, y=y, view_converter=self.view_converter,
                                   y_labels=None)

        # Another hack: rename 'targets' to match model expectations
        space, (X_source, y_source) = self.data_specs
        self.data_specs = (space, (X_source, 'condition'))
