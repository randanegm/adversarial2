
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.format.target_format import OneHotFormatter


class CIFAR10OneHot(CIFAR10):
    def __init__(self, which_set, onehot_dtype='uint8',
                 center=False, rescale=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None):
        super(CIFAR10OneHot, self).__init__(*args, **kwargs)

        # OK, mess with the target "Y" matrix and then re-init
        #
        # This is hacky, but the alternative is to copy-paste the entire
        # CIFAR10 constructor and change just a few lines.
        y = self.y

        formatter = OneHotFormatter(self.y_labels, dtype=onehot_dtype)
        y = formatter.format(y, mode='concatenate')

        # So hacky. I am sorry, Guido..
        kwargs['y'] = y
        super(CIFAR10OneHot, self).__init__(*args, **kwargs)
