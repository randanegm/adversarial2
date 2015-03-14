"""
Experiment with tweaking each conditional data axis individually.
"""

from argparse import ArgumentParser
import pprint

import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
import theano

from adversarial import sampler, util


# Parse arguments
parser = ArgumentParser(description=('Experiment with tweaking each '
                                     'axis individually.'))
# parser.add_argument('--conditional-noise-range', default=1.,
#                     type=float)
parser.add_argument('model_path')
parser.add_argument('embedding_file')
args = parser.parse_args()


embeddings = np.load(args.embedding_file)['arr_0']

condition_dim = embeddings.shape[1]
m, n = condition_dim, 10

condition_data = []
condition_data_mod = []

# Find embeddings which have minimal values for each axis, and invert the
# minimal values
for dim in range(condition_dim):
    min_embs = embeddings[embeddings[:, dim].argsort()[:n]]

    # Save unmodified form
    condition_data.extend(min_embs.copy())

    print 'Minimal values for axis', dim, ': ', min_embs[range(n), dim]
    new_val = embeddings[:, dim].max()
    print '\tReplacing with:', new_val

    min_embs[range(n), dim] *= new_val
    condition_data_mod.extend(min_embs)

condition_data = np.array(condition_data, dtype=theano.config.floatX)
condition_data_mod = np.array(condition_data_mod, dtype=theano.config.floatX)


# Now prepare generator
generator = util.load_generator_from_file(args.model_path)
noise_batch = generator.noise_space.make_theano_batch()
conditional_batch = generator.condition_space.make_theano_batch()
topo_sample_f = theano.function([noise_batch, conditional_batch],
                                generator.dropout_fprop((noise_batch, conditional_batch))[0])


# Sample some noise data -- this needs to be shared between orig and mod
# sample pairs
noise_data = generator.get_noise((m * n, generator.noise_dim)).eval()


samples_orig = topo_sample_f(noise_data, condition_data).swapaxes(0, 3)
samples_mod = topo_sample_f(noise_data, condition_data_mod).swapaxes(0, 3)

pv = PatchViewer(grid_shape=(2 * m, n), patch_shape=(32,32),
                 is_color=True)

for dim in range(condition_dim):
    for j in range(n):
        pv.add_patch(samples_orig[dim * n + j], activation=1)
    for j in range(n):
        pv.add_patch(samples_mod[dim * n + j])

pv.show()
