
import functools

from pylearn2.models.mlp import Layer
from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b


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
