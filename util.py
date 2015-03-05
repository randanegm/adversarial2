
import functools

import numpy as np
from PIL import Image
from pylearn2.models.mlp import Layer
from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.utils import serial

from adversarial.conditional import ConditionalAdversaryPair, ConditionalGenerator


class IdentityLayer(Layer):
    def __init__(self, layer_name, *args, **kwargs):
        self.layer_name = layer_name
        super(IdentityLayer, self).__init__(*args, **kwargs)

    @functools.wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

    @functools.wraps(Layer.get_output_space)
    def get_output_space(self):
        return self.get_input_space()

    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):
        return state_below

    @functools.wraps(Layer.get_params)
    def get_params(self):
        return []


class MaxPoolC01BLayer(Layer):
    def __init__(self, layer_name, pool_shape, pool_stride):
        self.layer_name = layer_name
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride

        super(MaxPoolC01BLayer, self).__init__()

    @functools.wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

    @functools.wraps(Layer.get_output_space)
    def get_output_space(self):
        return self.get_input_space()

    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)

        return max_pool_c01b(c01b=state_below, pool_shape=self.pool_shape,
                             pool_stride=self.pool_stride)

    @functools.wraps(Layer.get_params)
    def get_params(self):
        return []


def load_numpy_obj(file, key):
    loaded = np.load(file)
    assert key in loaded, "%s not found in NumPy file loaded from %s" % (key, file)
    return loaded[key]


def load_generator_from_file(file):
    generator = serial.load(file)

    if isinstance(generator, ConditionalAdversaryPair):
        generator = generator.generator

    assert isinstance(generator, ConditionalGenerator), 'Invalid generator path provided; loaded a value %r' % generator

    return generator


def make_image_from_sample(sample):
    """
    Make a PIL Image object from the given sampled image data. Should be
    in (0, 1, 'c') format.
    """

    assert sample.ndim == 3, 'Sample should have axes (0, 1, "c"); instead has %i axes' % sample.ndim
    assert sample.shape[2] <= 3, "Sample has %i color channels -- this can't be right.." % sample.shape[2]

    # Rescale
    sample = sample / np.abs(sample).max()

    sample *= 0.5
    sample += 0.5
    sample = np.cast['uint8'](sample * 255)

    img = Image.fromarray(sample)
    return img
