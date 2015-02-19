
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

        self.input_source = self.discriminator.get_input_source()
        self.output_space = self.discriminator.get_output_space()


class ConditionalGenerator(Generator):
    def __init__(self, mlp, input_condition_space, noise_dim=100, *args, **kwargs):
        super(ConditionalGenerator, self).__init__(mlp, *args, **kwargs)

        self.noise_dim = noise_dim
        self.noise_space = VectorSpace(dim=self.noise_dim)

        self.condition_space = input_condition_space

        self.input_space = CompositeSpace([self.noise_space, self.condition_space])
        self.mlp.set_input_space(self.input_space)

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
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, ConditionalAdversaryPair)

        G, D = model.generator, model.discriminator

        # X_data: empirical data to be sent to the discriminator. We'll
        # make an equal amount of generated data and send this to the
        # discriminator as well.
        #
        # X_condition: Conditional data for each empirical sample.
        #
        # TODO something wrong here -- where does X_condition come from
        # on this invocation?
        (X_data, X_condition), y = data
        m = data.shape[space.get_batch_axis()]

        # Expected discriminator output: 1 for real data, 0 for
        # generated samples
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)

        S, z, other_layers = g.sample_and_noise(X_condition,
                                                default_input_include_prob=self.generator_default_input_include_prob,
                                                default_input_scale=self.generator_default_input_scale,
                                                all_g_layers=(self.infer_layer is not None))

        if self.noise_both != 0.:
            rng = MRG_RandomStreams(2014 / 6 + 2)
            S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
            X_data = X_data + rng.normal(size=X_data.shape, dtype=X_data.dtype) * self.noise_both

        fprop_args = [self.discriminator_default_input_include_prob,
                      self.discriminator_input_include_probs,
                      self.discriminator_default_input_scale,
                      self.discriminator_input_scales]

        # Run discriminator on empirical data (1 expected)
        y_hat1 = D.dropout_fprop(X_data, *fprop_args)

        # Run discriminator on generated data (0 expected)
        y_hat0 = D.dropout_fprop(S, *fprop_args)

        # Compute discriminator objective
        d_obj = 0.5 * (D.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        # Compute generator objective
        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = D.dropout_fprop(S)
            g_obj = d.layers[-1].cost(y1, y_hat0_no_drop)
        else:
            g_obj = d.layers[-1].cost(y1, y_hat0)

        if self.blend_obj:
            g_obj = (self.zurich_coeff * g_obj - self.minimax_coeff * d_obj) / (self.zurich_coeff + self.minimax_coeff)

        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            pred = model.inferer.dropout_fprop(S_nograd, *fprop_args)

            if self.infer_layer is None:
                target = z
            else:
                target = other_layers[self.infer_layer]

            i_obj = model.inferer.layers[-1].cost(target, pred)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj
