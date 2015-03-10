"""
Demonstrate strength of face embeddings (correspondence between real
images and model images with similar embeddings).

We build a grid of size `(m + 1) * n`, where `m` is the number of different
vanilla noise vectors to sample for each column and `n` is the number of
training images to retrieve.

1. Draw `n` random images (with associated conditional data) from the
   training dataset.
2. Sample `m` vanilla noise values.
3. Draw `m * n` grid, where cell in (i, j) refers to generator sample
   using vanilla noise `i` with training conditional data `j`
4. Draw original `n` images from dataset in an extra row
"""

from argparse import ArgumentParser

import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
import theano

from adversarial import sampler, util


# Parse arguments
parser = ArgumentParser(description=('Demonstrate effects of adding noise '
                                     'to conditional data.'))
parser.add_argument('model_path')
args = parser.parse_args()


m, n = 10, 20

model = serial.load(args.model_path)
dataset = yaml_parse.load(model.dataset_yaml_src)

# Sample from training set
ids = np.random.choice(len(dataset.X), n, replace=False)
X_sample = dataset.X[ids]
y_sample = dataset.y[ids]

# Generate from the fetched conditional data
conditional_data = y_sample.repeat(m, axis=0)

conditional_batch = generator.condition_space.make_theano_batch()
topo_sample_f = theano.function([conditional_batch],
                                generator.sample(conditional_batch))
topo_samples = topo_sample_f(conditional_data).swapaxes(0, 3)

pv = PatchViewer(grid_shape=(m + 1, n), patch_shape=(32,32),
                 is_color=True)

for i in xrange(topo_samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample)

for original_image in X_sample:
    print original_image.shape
    pv.add_patch(original_image)

pv.show()
