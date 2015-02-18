
from pylearn2.models.mlp import MLP, CompositeLayer
from pylearn2.space import CompositeSpace, VectorSpace
from theano import tensor as T

from adversarial import AdversaryPair, AdversaryCost2, Generator


class ConditionalAdversaryPair(AdversaryPair):
    def __init__(self, generator, discriminator, data_space, condition_space,
                 inferer=None,
                 inference_monitoring_batch_size=128,
                 monitor_generator=True,
                 monitor_discriminator=True,
                 monitor_inference=True,
                 shrink_d=0.):
        super(ConditionalAdversaryPair, self).__init__(generator, discriminator, inferer,
                                                       inference_monitoring_batch_size, monitor_generator, monitor_discriminator,
                                                       monitor_inference, shrink_d)

        self.data_space = data_space
        self.condition_space = condition_space

    def get_output_space(self):
        return self.discriminator.get_output_space()


class ConditionalGenerator(Generator):
    def __init__(self, mlp, input_condition_space, noise_dim=100, *args, **kwargs):
        super(ConditionalGenerator, self).__init__(mlp, *args, **kwargs)

        self.noise_dim = noise_dim
        self.noise_space = VectorSpace(dim=self.noise_dim)

        self.condition_space = input_condition_space

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
    def __init__(self, data_mlp, condition_mlp, joint_mlp,
                 input_data_space, input_condition_space, input_source=('data', 'condition'),
                 *args, **kwargs):
        """
        A discriminator acting within a cGAN which may "condition" on
        extra information.

        Parameters
        ----------
        data_mlp: pylearn2.models.mlp.MLP
            MLP which processes the data-space information. Must output
            a `VectorSpace` of some sort.

        condition_mlp: pylearn2.models.mlp.MLP
            MLP which processes the condition-space information. Must
            output a `VectorSpace` of some sort.

        joint_mlp: pylearn2.models.mlp.MLP
            MLP which processes the combination of the outputs of the
            data MLP and the condition MLP.

        input_data_space : pylearn2.space.CompositeSpace
            Space which contains the empirical / model-generated data

        input_condition_space : pylearn2.space.CompositeSpace
            Space which contains the extra data being conditioned on

        kwargs : dict
            Passed on to MLP superclass.
        """

        # Make sure user isn't trying to override any fixed keys
        for illegal_key in ['input_source', 'input_space', 'layers']:
            assert illegal_key not in kwargs

        # First feed forward in parallel along the data and condition
        # MLPs; then feed the composite output to the joint MLP
        layers = [
            CompositeLayer(layer_name='discriminator_composite',
                           layers=[data_mlp, condition_mlp],
                           inputs_to_layers={0: [0], 1: [1]}),
            joint_mlp
        ]

        super(ConditionalDiscriminator, self).__init__(
            layers=layers,
            input_space=CompositeSpace([input_data_space, input_condition_space]),
            input_source=input_source,
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
