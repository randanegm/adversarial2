from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
space = model.generator.get_output_space()
from pylearn2.config import yaml_parse
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.gui.patch_viewer import PatchViewer
import theano
import numpy as np

dataset = yaml_parse.load(model.dataset_yaml_src)

grid_shape = None

# Number of choices for one-hot values
rows = model.generator.condition_space.get_total_dimension()

# Samples per condition
sample_cols = 5

# Generate conditional information
conditional_batch = model.generator.condition_space.make_theano_batch()
formatter = OneHotFormatter(rows,
                            dtype=model.generator.condition_space.dtype)
conditional = formatter.theano_expr(conditional_batch, mode='concatenate')

# Now sample from generator
# For some reason format_as from VectorSpace is not working right
topo_samples_batch = model.generator.sample(conditional)
topo_sample_f = theano.function([conditional], topo_samples_batch)
conditional_data = formatter.format(np.concatenate([np.repeat(i, sample_cols) for i in range(rows)])
                                      .reshape((rows * sample_cols, 1)),
                                    mode='concatenate')
topo_samples = topo_sample_f(conditional_data)

samples = dataset.get_design_matrix(topo_samples)
dataset.axes = ['b', 0, 1, 'c']
dataset.view_converter.axes = ['b', 0, 1, 'c']
topo_samples = dataset.get_topological_view(samples)

pv = PatchViewer(grid_shape=(rows, sample_cols + 1), patch_shape=(32,32),
        is_color=True)
scale = np.abs(samples).max()

X = dataset.X
topo = dataset.get_topological_view()
index = 0
for i in xrange(samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    print topo_sample.min(), topo_sample.max()
    pv.add_patch(topo_sample / scale, rescale=False)

    if (i +1) % sample_cols == 0:
        sample = samples[i, :]
        dists = np.square(X - sample).sum(axis=1)
        j = np.argmin(dists)
        match = topo[j, :]
        print match.min(), match.max()
        pv.add_patch(match / scale, rescale=False, activation=1)

pv.show()
