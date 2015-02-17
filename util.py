
import functools

from pylearn2.models.mlp import Layer


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
