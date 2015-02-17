
from pylearn2.models.mlp import MLP
from pylearn2.space import CompositeSpace, VectorSpace
from theano import tensor as T

from adversarial import AdversaryPair, AdversaryCost2, Generator


class ConditionalAdversaryPair(AdversaryPair):
    def __init__(self, *args, **kwargs):
        super(ConditionalAdversaryPair, self).__init__(*args, **kwargs)


class ConditionalGenerator(Generator):
    def __init__(self, mlp, input_condition_space, noise_dim=100, *args, **kwargs):
        super(ConditionalGenerator, self).__init__(mlp, *args, **kwargs)

        self.noise_dim = noise_dim
        self.noise_space = VectorSpace(dim=self.noise_dim)

        self.condition_space = condition_space

        self.input_space = CompositeSpace([self.noise_space, self.condition_space])
        self.mlp.set_input_space(self.input_space)

    def get_input_space(self):
        return self.input_space

    def sample_and_noise(self, conditional_data, default_input_include_prob=1., default_input_scale=1.,
                         all_g_layers=False):
        """
        Retrieve a sample (and the noise used to generate the sample)
        conditioned on some input data.

        Parameters
        ----------
        conditional_data: member of self.condition_space
            A minibatch of conditional data to feedforward.

        default_input_include_prob: float
            WRITEME

        default_input_scale: float
            WRITEME

        all_g_layers: boolean
            If true, return all generator layers in `other_layers` slot
            of this method's return value. (Otherwise returns `None` in
            this slot.)

        Returns
        -------
        net_output: 3-tuple
            Tuple of the form `(sample, noise, other_layers)`.
        """

        num_samples = conditional_data.shape[0]

        noise = self.get_noise((num_samples, self.noise_dim))
        # TODO necessary?
        formatted_noise = self.noise_space.format_as(noise, self.noise_space)

        # Build inputs: concatenate noise with conditional data
        inputs = (formatted_noise, conditional_data)

        # Feedforward
        # if all_g_layers:
        #     rval = self.mlp.dropout_fprop(inputs, default_input_include_prob=default_input_include_prob,
        #                                   default_input_scale=default_input_scale, return_all=all_g_layers)
        #     other_layers, rval = rval[:-1], rval[-1]
        # else:
        rval = self.mlp.dropout_fprop(inputs, default_input_include_prob=default_input_include_prob,
                                      default_input_scale=default_input_scale)
            # other_layers = None

        return rval, formatted_noise# , other_layers

    def sample(self, conditional_data, **kwargs):
        sample, _, _ = self.sample_and_noise(conditional_data, **kwargs)
        return sample

    def get_monitoring_channels(self, data):
        # TODO
        return {}


class ConditionalDiscriminator(MLP):
    def __init__(self, input_data_space, input_condition_space, *args, **kwargs):
        """
        A discriminator acting within a cGAN which may "condition" on
        extra information.

        Parameters
        ----------
        input_data_space : pylearn2.space.CompositeSpace
            Space which contains the empirical / model-generated data

        input_condition_space : pylearn2.space.CompositeSpace
            Space which contains the extra data being conditioned on

        kwargs : dict
            Passed on to MLP superclass.
        """

        assert 'input_space' not in kwargs

        super(ConditionalDiscriminator, self).__init__(
            input_space=CompositeSpace([input_data_space, input_condition_space]),
            *args, **kwargs)


class ConditionalAdversaryCost(AdversaryCost2):
    """
    Defines the cost expression for a cGAN.
    """

    # We need to see labels for the real-world examples, so that we can
    # condition the generator + discriminator on them
    supervised = True

    def __init__(self, **kwargs):
        super(ConditionalAdversaryCost, self).__init__(**kwargs)

    def get_samples_and_objectives(self, model, data):
        # TODO
        print model, data
