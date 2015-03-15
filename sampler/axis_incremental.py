"""
Incrementally tweak specified axes. Build new faces!
"""

from argparse import ArgumentParser
import pprint

import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
import theano

from adversarial import sampler, util


# Parse arguments
parser = ArgumentParser(description=('Experiment with tweaking each '
                                     'axis incrementally.'))
parser.add_argument('-s', '--conditional-sampler', default='random',
                    choices=sampler.conditional_samplers.values(),
                    type=lambda k: sampler.conditional_samplers[k])
# parser.add_argument('--conditional-noise-range', default=1.,
#                     type=float)
parser.add_argument('model_path')
parser.add_argument('embedding_file')
parser.add_argument('-a', '--axes',
                    help='Comma-separated list of axes to modify')
args = parser.parse_args()


embeddings = np.load(args.embedding_file)['arr_0']

if args.axes is None:
    args.axes = range(embeddings.shape[1])
else:
    args.axes = [int(x) for x in args.axes.strip().split(',')]


condition_dim = embeddings.shape[1]
m, n = len(args.axes), 10
shift = 7.5


# Prepare generator
generator = util.load_generator_from_file(args.model_path)
noise_batch = generator.noise_space.make_theano_batch()
conditional_batch = generator.condition_space.make_theano_batch()
topo_sample_f = theano.function([noise_batch, conditional_batch],
                                generator.dropout_fprop((noise_batch, conditional_batch))[0])

# Sample some noise data -- this needs to be shared between orig and mod
# sample pairs
noise_data = generator.get_noise((n, generator.noise_dim)).eval()


# Begin modifying axes
base_conditional_data = args.conditional_sampler(generator, n, 1,
                                                 embedding_file=args.embedding_file)
print 'Mean for each axis:'
pprint.pprint(zip(args.axes, base_conditional_data[:, args.axes].mean(axis=1)))
base_conditional_data[:, args.axes] -= 0.5 * shift

mod_conditional_data = base_conditional_data.copy()

# Build up a flat array of modified conditional data
mod_conditional_steps = []
for axis in args.axes:
    mod_conditional_data[:, axis] += shift
    mod_conditional_steps.extend(mod_conditional_data.copy())

mod_conditional_steps = np.array(mod_conditional_steps)

samples_orig = topo_sample_f(noise_data, base_conditional_data).swapaxes(0, 3)
samples_mod = topo_sample_f(np.tile(noise_data, (m, 1)), mod_conditional_steps).swapaxes(0, 3)

pv = PatchViewer(grid_shape=(m + 1, n), patch_shape=(32,32),
                 is_color=True)

for sample_orig in samples_orig:
    pv.add_patch(sample_orig, activation=1)

for sample_mod in samples_mod:
    pv.add_patch(sample_mod)

pv.show()
