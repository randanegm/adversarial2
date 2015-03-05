from argparse import ArgumentParser

from pylearn2.gui.patch_viewer import PatchViewer

from adversarial import sampler


# Parse arguments
parser = ArgumentParser(description=('Sample images from the generative component of a '
                                     'cGAN learned on the LFW/LFWcrop dataset.'))
parser.add_argument('-s', '--conditional-sampler', default='fix_random',
                    choices=sampler.conditional_samplers.values(),
                    type=lambda k: sampler.conditional_samplers[k])
parser.add_argument('model_path')
args = parser.parse_args()


m, n = 4, 5
topo_samples = sampler.get_conditional_topo_samples(args.model_path, m, n,
                                                    args.conditional_sampler)

pv = PatchViewer(grid_shape=(m, n), patch_shape=(32,32),
                 is_color=True)

for i in xrange(topo_samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample)

pv.show()
