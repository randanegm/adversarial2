"""
Demonstrate effects of adding noise to conditional data.

We build a grid of size `(m + 1) * n`, where `m` is the number of different noise
vectors to sample and `n` is the number of images on which to try each out.

1. Sample `n` conditional vectors (see -s argument) and noise values
2. Sample `m` conditional noise values
3. Draw `m * n` grid, where cell in (i, j) refers to generator sample
   using conditional noise `i` on source `j`
4. Draw original `n` images with no conditional noise in an extra row
"""

from argparse import ArgumentParser
import pprint

import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
import theano

from adversarial import sampler, util


# Parse arguments
parser = ArgumentParser(description=('Demonstrate effects of adding noise '
                                     'to conditional data.'))
parser.add_argument('-s', '--conditional-sampler', default='random',
                    choices=sampler.conditional_samplers.values(),
                    type=lambda k: sampler.conditional_samplers[k])
parser.add_argument('-e', '--embedding-file')
parser.add_argument('--conditional-noiser', choices=sampler.noisers.values(),
                    default='random_uniform', type=lambda k: sampler.noisers[k])
parser.add_argument('--conditional-noise-range', default=1.,
                    type=float)
parser.add_argument('model_path')
args = parser.parse_args()


m, n = 14, 10

generator = util.load_generator_from_file(args.model_path)


base_conditional_data = args.conditional_sampler(generator, n, 1,
                                                 embedding_file=(args.embedding_file if args.embedding_file is not None
                                                                 else sampler.DEFAULT_EMBEDDING_FILE))
base_noise_data = generator.get_noise((n, generator.noise_dim))

# Build `m * n` grid of conditional data + noise data, where rows are
# identical
condition_dim = base_conditional_data.shape[1]
conditional_data = base_conditional_data.reshape((1, n, condition_dim)).repeat(m, axis=0)
noise_data = base_noise_data.reshape((1, n, generator.noise_dim)).repeat(m, axis=0).eval()

# Build `m * n` grid of condition noise, where columns are identical
conditional_noise = (args.conditional_noiser(m, condition_dim, range=args.conditional_noise_range)
                         .astype(base_conditional_data.dtype).reshape(m, 1, condition_dim))
pprint.pprint(zip(np.nonzero(conditional_noise)[2], conditional_noise[np.nonzero(conditional_noise)]))
conditional_noise = conditional_noise.repeat(n, axis=1)

# Noise up conditional data
conditional_data_noised = conditional_data + conditional_noise

# Reshape to make generator happy (should be `(m * n) * dim`, where
# `dim` is noise dimension or condition dimension)
conditional_data_noised = conditional_data_noised.reshape((m * n, condition_dim))
noise_data = noise_data.reshape((m * n, generator.noise_dim))

noise_batch = generator.noise_space.make_theano_batch()
conditional_batch = generator.condition_space.make_theano_batch()
topo_sample_f = theano.function([noise_batch, conditional_batch],
                                generator.dropout_fprop((noise_batch, conditional_batch))[0])
topo_samples = topo_sample_f(noise_data, conditional_data_noised).swapaxes(0, 3)

pv = PatchViewer(grid_shape=(m + 1, n), patch_shape=(32,32),
                 is_color=True)

for i in xrange(topo_samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample)

untouched_samples = topo_sample_f(noise_data[:n], conditional_data[0, :, :].reshape((n, condition_dim))).swapaxes(0, 3)
for sample in untouched_samples:
    pv.add_patch(sample, activation=1)

np.save('conditional_noise', conditional_noise[:, 0, :].reshape((m, condition_dim)))

pv.show()
