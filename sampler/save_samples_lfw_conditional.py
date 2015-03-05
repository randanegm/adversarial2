from argparse import ArgumentParser
import os
import sys

from pylearn2.utils import serial
import theano
from theano import tensor as T

from adversarial import sampler
from adversarial.conditional import ConditionalAdversaryPair
from adversarial.util import make_image_from_sample


parser = ArgumentParser(description=('Sample a large number of images from a cGAN and '
                                     'save to a directory.'))
parser.add_argument('-s', '--conditional-sampler', default='random',
                    choices=sampler.conditional_samplers.values(),
                    type=lambda k: sampler.conditional_samplers[k])
parser.add_argument('-n', type=int, default=1000, help='Number of images to generate')
parser.add_argument('model_path')
parser.add_argument('output_directory')
args = parser.parse_args()


if os.path.exists(args.output_directory):
    print 'Warning: output directory %s exists' % args.output_directory

    if os.path.isfile(args.output_directory):
        raise ValueError("Provided output directory %s is a file" % args.output_directory)
else:
    os.path.mkdirs(args.output_directory)


generator = serial.load(args.model_path)
generator = generator.generator if isinstance(generator, ConditionalAdversaryPair)

cond_samples = sampler.get_conditional_topo_samples(generator, args.n, 1,
                                                    args.conditional_sampler)

cond_samples_batch = T.matrix(dtype=cond_samples.dtype)
sample_f = theano.function([cond_samples_batch], generator.sample(cond_samples_batch))
samples = sample_f(cond_samples)

for i, sample in enumerate(samples):
    img = make_image_from_sample(sample)
    path = os.path.join(args.output_directory, '%04i.png' % i)
    img.save(path)

print >> sys.stderr, "Saved %i images to %s." % (args.n, args.output_directory)
