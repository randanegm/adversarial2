from argparse import ArgumentParser

import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import serial

from adversarial import sampler


# Parse arguments
parser = ArgumentParser(description=('Sample images from the generative component of a '
                                     'cGAN learned on the LFW/LFWcrop dataset.'))
parser.add_argument('-s', '--conditional-sampler', default='fix_random',
                    choices=sampler.conditional_samplers.values(),
                    type=lambda k: sampler.conditional_samplers[k])
parser.add_argument('-e', '--embedding-file')
parser.add_argument('--show-nearest-training', default=False, action='store_true')
parser.add_argument('model_path')
args = parser.parse_args()


m, n = 4, 5
topo_samples, _ = sampler.get_conditional_topo_samples(args.model_path, m, n,
                                                       args.conditional_sampler,
                                                       embedding_file=(args.embedding_file if args.embedding_file is not None
                                                                       else sampler.DEFAULT_EMBEDDING_FILE))

pv = PatchViewer(grid_shape=(m, (n + 1 if args.show_nearest_training else n)),
                 patch_shape=(32,32), is_color=True)

# Optionally load dataset for --show-nearest-training
dataset = None
if args.show_nearest_training:
    model = serial.load(args.model_path)

    # Shape: b * (0 * 1 * c)
    # (topo view)
    dataset = yaml_parse.load(model.dataset_yaml_src)

for i in xrange(topo_samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample)

    if (args.show_nearest_training and dataset is not None
        and (i + 1) % n == 0):
        sample_topo = topo_samples[i].reshape(-1)
        dists = np.square(dataset.X - sample_topo).sum(axis=1)
        min_j = np.argmin(dists)

        match = dataset.X[min_j].reshape(32, 32, 3)
        pv.add_patch(match, activation=1)

pv.show()
