import functools

from pylearn2.models.mlp import MLP, CompositeLayer
from pylearn2.space import CompositeSpace, VectorSpace
import theano
from theano import tensor as T
from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

from adversarial import AdversaryPair, AdversaryCost2, Generator, theano_parzen


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

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        g_ch = self.generator.get_monitoring_channels(data)
        d_ch = self.discriminator.get_monitoring_channels((data, None))

        samples, _, conditional_data, _ = self.generator.sample_and_noise(100)
        d_samp_ch = self.discriminator.get_monitoring_channels(((samples, conditional_data), None))

        i_ch = OrderedDict()
        if self.inferer is not None:
            batch_size = self.inference_monitoring_batch_size
            sample, noise, conditional_data, _ = self.generator.sample_and_noise(batch_size)
            i_ch.update(self.inferer.get_monitoring_channels(((sample, conditional_data), noise)))

        if self.monitor_generator:
            for key in g_ch:
                rval['gen_' + key] = g_ch[key]
        if self.monitor_discriminator:
            for key in d_ch:
                rval['dis_on_data_' + key] = d_samp_ch[key]
            for key in d_ch:
                rval['dis_on_samp_' + key] = d_ch[key]
        if self.monitor_inference:
            for key in i_ch:
                rval['inf_' + key] = i_ch[key]
        return rval


class ConditionalGenerator(Generator):
    def __init__(self, mlp, input_condition_space, condition_distribution, noise_dim=100, *args, **kwargs):
        super(ConditionalGenerator, self).__init__(mlp, *args, **kwargs)

        self.noise_dim = noise_dim
        self.noise_space = VectorSpace(dim=self.noise_dim)

        self.condition_space = input_condition_space
        self.condition_distribution = condition_distribution

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

        if isinstance(conditional_data, int):
            conditional_data = self.condition_distribution.sample(conditional_data)

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

        return rval, formatted_noise, conditional_data, None# , other_layers

    def sample(self, conditional_data, **kwargs):
        sample, _, _, _ = self.sample_and_noise(conditional_data, **kwargs)
        return sample

    def get_monitoring_channels(self, data):
        if data is None:
            m = 100
            conditional_data = self.condition_distribution.sample(m)
        else:
            _, conditional_data = data
            m = conditional_data.shape[0]

        noise = self.get_noise((m, self.noise_dim))
        rval = OrderedDict()

        sampled_data = (noise, conditional_data)
        try:
            rval.update(self.mlp.get_monitoring_channels((sampled_data, None)))
        except Exception:
            warnings.warn("something went wrong with generator.mlp's monitoring channels")

        if  self.monitor_ll:
            rval['ll'] = T.cast(self.ll(data, self.ll_n_samples, self.ll_sigma),
                                        theano.config.floatX).mean()
            rval['nll'] = -rval['ll']
        return rval

    def ll(self, data, n_samples, sigma):
        real_data, conditional_data = data
        sampled_data = self.sample(conditional_data)

        output_space = self.mlp.get_output_space()
        if 'Conv2D' in str(output_space):
            samples = output_space.convert(sampled_data, output_space.axes, ('b', 0, 1, 'c'))
            samples = samples.flatten(2)
            data = output_space.convert(real_data, output_space.axes, ('b', 0, 1, 'c'))
            data = data.flatten(2)
        parzen = theano_parzen(data, samples, sigma)
        return parzen


class CompositeMLPLayer(CompositeLayer):
    """A CompositeLayer where each of the components are MLPs.

    Supports forwarding dropout parameters to each MLP independently."""

    def __init__(self, layers, *args, **kwargs):
        for layer in layers:
            assert isinstance(layer, MLP), "CompositeMLPLayer only supports MLP component layers"

        super(CompositeMLPLayer, self).__init__(layers=layers, *args, **kwargs)

    def _collect_mlp_layer_names(self):
        """Collect the layer names of the MLPs nested within this
        layer."""

        return [[sub_layer.layer_name for sub_layer in mlp.layers] for mlp in self.layers]

    def validate_layer_names(self, req_names):
        all_names = []
        for sub_names in self._collect_mlp_layer_names():
            all_names.extend(sub_names)

        if any(req_name not in all_names for req_name in req_names):
            unknown_names = [req_name for req_name in req_names
                             if req_name not in all_names]
            raise ValueError("No MLPs in this CompositeMLPLayer have layer(s) named %s" %
                             ", ".join(unknown_names))

    def dropout_fprop(self, state_below, input_include_probs=None, input_scales=None,
                      *args, **kwargs):
        """Extension of Layer#fprop which forwards on dropout parameters
        to MLP sub-layers."""

        if input_include_probs is None:
            input_include_probs = {}
        if input_scales is None:
            input_scales = {}

        # Use to determine which args should be routed to which places
        mlp_layer_names = self._collect_mlp_layer_names()

        rvals = []
        for i, mlp in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_state_below = [state_below[j]
                                   for j in self.layers_to_inputs[i]]
                # This is to mimic the behavior of CompositeSpace's restrict
                # method, which only returns a CompositeSpace when the number
                # of components is greater than 1
                if len(cur_state_below) == 1:
                    cur_state_below, = cur_state_below
            else:
                cur_state_below = state_below

            # Get dropout params for relevant layers
            relevant_keys_include = set(mlp_layer_names[i]) & set(input_include_probs)
            relevant_keys_scale = set(mlp_layer_names[i]) & set(input_scales)

            relevant_include = dict((k, input_include_probs[k]) for k in relevant_keys_include)
            relevant_scale = dict((k, input_scales[k]) for k in relevant_keys_scale)

            rvals.append(mlp.dropout_fprop(cur_state_below,
                                           input_include_probs=relevant_include,
                                           input_scales=relevant_scale,
                                           *args, **kwargs))

        return tuple(rvals)


class ConditionalDiscriminator(MLP):
    def __init__(self, data_mlp, condition_mlp, joint_mlp,
                 input_data_space, input_condition_space, input_source=('features', 'condition'),
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
            CompositeMLPLayer(layer_name='discriminator_composite',
                              layers=[data_mlp, condition_mlp],
                              inputs_to_layers={0: [0], 1: [1]}),
            joint_mlp
        ]

        super(ConditionalDiscriminator, self).__init__(
            layers=layers,
            input_space=CompositeSpace([input_data_space, input_condition_space]),
            input_source=input_source,
            *args, **kwargs)

    @functools.wraps(MLP.dropout_fprop)
    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """Extended version of MLP#dropout_fprop which supports passing
        on dropout parameters to nested MLPs within this MLP.

        Coupled with `CompositeMLPLayer`, which is a core part of the
        ConditionalDiscriminator setup.
        """

        if input_include_probs is None:
            input_include_probs = {}
        if input_scales is None:
            input_scales = {}

        layer_name_set = set(input_include_probs.keys())
        layer_name_set.update(input_scales.keys())

        # Remove layers from the outer net
        layer_name_set.difference_update(set(layer.layer_name for layer in self.layers))

        # Make sure remaining layers are contained within sub-MLPs
        # NOTE: Assumes composite layer is only at position zero
        self.layers[0].validate_layer_names(list(input_include_probs.keys()))
        self.layers[0].validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            # Forward propagate
            if isinstance(layer, CompositeMLPLayer):
                # This is a composite MLP layer -- forward on the
                # dropout parameters
                state_below = layer.dropout_fprop(state_below,
                                                  default_input_include_prob=default_input_include_prob,
                                                  input_include_probs=input_include_probs,
                                                  default_input_scale=default_input_scale,
                                                  input_scales=input_scales,
                                                  per_example=per_example)
            else:
                state_below = self.apply_dropout(
                    state=state_below,
                    include_prob=include_prob,
                    theano_rng=theano_rng,
                    scale=scale,
                    mask_value=layer.dropout_input_mask_value,
                    input_space=layer.get_input_space(),
                    per_example=per_example
                )

                state_below = layer.fprop(state_below)

        return state_below


class ConditionalAdversaryCost(AdversaryCost2):
    """
    Defines the cost expression for a cGAN.
    """

    supervised = False

    def __init__(self, condition_distribution, **kwargs):
        self.condition_distribution = condition_distribution

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
        X_data, X_condition = data
        m = X_data.shape[3]
        # TODO get_batch_axis is wrong here.. probably a dataset issue?

        # Expected discriminator output: 1 for real data, 0 for
        # generated samples
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)

        # Generate conditional data for the generator
        G_conditional_data = self.condition_distribution.sample(m)
        S, z, _, other_layers = G.sample_and_noise(G_conditional_data,
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
        y_hat1 = D.dropout_fprop((X_data, X_condition), *fprop_args)

        # Run discriminator on generated data (0 expected)
        y_hat0 = D.dropout_fprop((S, G_conditional_data), *fprop_args)

        # Compute discriminator objective
        d_obj = 0.5 * (D.layers[-1].cost(y1, y_hat1) + D.layers[-1].cost(y0, y_hat0))

        # Compute generator objective
        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = D.dropout_fprop(S)
            g_obj = D.layers[-1].cost(y1, y_hat0_no_drop)
        else:
            g_obj = D.layers[-1].cost(y1, y_hat0)

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

    def get_monitoring_channels(self, model, data, **kwargs):
        rval = OrderedDict()

        space, sources = self.get_data_specs(model)
        X_data, X_condition = data
        m = X_data.shape[space.get_batch_axis()]

        G, D = model.generator, model.discriminator

        # Compute false negatives w/ empirical samples
        y_hat = D.fprop((X_data, X_condition))
        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        # Compute false positives w/ generated sample
        G_conditional_data = self.condition_distribution.sample(m)
        samples = G.sample(G_conditional_data)
        y_hat = D.fprop((samples, G_conditional_data))
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')

        # y = T.alloc(0., m, 1)
        cost = D.cost_from_X(((samples, G_conditional_data), y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())

        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if model.monitor_inference and i_obj != 0:
            rval['objective_i'] = i_obj
        if model.monitor_discriminator:
            rval['objective_d'] = d_obj
        if model.monitor_generator:
            rval['objective_g'] = g_obj

        rval['now_train_generator'] = self.now_train_generator
        return rval
