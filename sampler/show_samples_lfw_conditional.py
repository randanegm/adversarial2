from argparse import ArgumentParser

import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.datasets import dense_design_matrix
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import serial
import theano

from adversarial import sampler


# Build string mapping for conditional samplers so that they can be
# triggered from CLI
conditional_samplers = {
    'fix_random': sampler.sample_conditional_fix_random,
    'fix_embeddings': sampler.sample_conditional_fix_embeddings,
}


# Parse arguments
parser = ArgumentParser(description=('Sample images from the generative component of a '
                                     'cGAN learned on the LFW/LFWcrop dataset.'))
parser.add_argument('-g', '--model-is-generator-only', action='store_true', default=False,
                    help='If true, provided model path is a generator only, not a full cGAN')
parser.add_argument('-s', '--conditional-sampler', default='fix_random',
                    choices=conditional_samplers.values(),
                    type=lambda k: conditional_samplers[k])
parser.add_argument('model_path')
args = parser.parse_args()


# Load model
model = serial.load(args.model_path)
generator = model if args.model_is_generator_only else model.generator
space = generator.get_output_space()

rows = 4
sample_cols = 5

# First sample conditional data
conditional_batch = generator.condition_space.make_theano_batch()
conditional_data = args.conditional_sampler(generator, rows, sample_cols)

topo_samples_batch = generator.sample(conditional_batch)
topo_sample_f = theano.function([conditional_batch], topo_samples_batch)
topo_samples = topo_sample_f(conditional_data).swapaxes(0, 3)

pv = PatchViewer(grid_shape=(rows, sample_cols), patch_shape=(32,32),
                 is_color=True)

for i in xrange(topo_samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample)

pv.show()
