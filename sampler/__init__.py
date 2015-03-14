"""
Defines shared utilities for sampling from a vGAN or cGAN generator
component.
"""

import numpy as np
from pylearn2.utils import serial
import theano

from adversarial.conditional import ConditionalAdversaryPair, ConditionalGenerator
from adversarial.util import load_generator_from_file


DEFAULT_EMBEDDING_FILE = '/afs/cs.stanford.edu/u/jgauthie/scr/lfw-lsa/LFW_attributes_30d.npz'

def get_embeddings(file, n):
    embs = np.load(file)['arr_0']

    source_points = embs[np.random.choice(embs.shape[0], size=n)].copy()
    dim = source_points.shape[1]

    return source_points, dim


def get_noise_random_uniform(m, n, range=1, **kwargs):
    """
    Sample a random noise `m * n` matrix where each cell is distributed
    Unif(-range, range)."""

    return range * (np.random.rand(m, n) * 2. - 1.)


def get_noise_random_uniform_oneaxis(m, n, range=1, **kwargs):
    """
    Sample a random noise `m * n` matrix, where each row has exactly one
    nonzero value, and this value is sampled independently from
    Unif(-range, range).
    """

    ret = np.zeros((m, n))
    ret[xrange(m), np.random.choice(n, m)] = range * (np.random.rand(m) * 2. - 1.)
    return ret


def get_noise_consistent_uniform_oneaxis(m, n, range=1, **kwargs):
    """
    For each axis, sample a random noise value `a_i` and add for that
    axis only. Requires `m == n`.
    """

    assert m == n, 'Requires square matrix'

    rands = range * (np.random.rand(m) * 2. - 1.)
    mat = rands * np.eye(m)

    return mat


# Build string mapping for noisers so that they can be triggered from
# CLI
noisers = {
    'random_uniform': get_noise_random_uniform,
    'random_uniform_oneaxis': get_noise_random_uniform_oneaxis,
    'consistent_uniform_oneaxis': get_noise_consistent_uniform_oneaxis,
}


def sample_conditional_random(generator, m, n, **kwargs):
    """
    Sample `m * n` points from condition space completely randomly.
    """

    return generator.condition_distribution.sample(m * n).eval()



def sample_conditional_fix_random(generator, m, n, noise_range=1, **kwargs):
    """
    Sample `m * n` points in condition space by sampling `m` points
    and adding small random noise `n` times for each point.
    """

    conditional_data = generator.condition_distribution.sample(m).eval()
    conditional_dim = conditional_data.shape[1]
    conditional_data = (conditional_data.reshape((m, 1, conditional_dim)).repeat(n, axis=1)
                                        .reshape((m * n, conditional_dim)))
    conditional_data += noise_range * (np.random.rand(*conditional_data.shape) * 2. - 1.)

    return conditional_data


def sample_conditional_fix_embeddings_no_noise(generator, m, n,
                                               embedding_file=DEFAULT_EMBEDDING_FILE,
                                               **kwargs):
    """
    Sample `m * n` points in condition space by retrieving `m` points
    from a provided dataset and repeating each `n` times.
    """

    source_points, dim = get_embeddings(embedding_file, m)

    source_points = source_points.reshape((m, 1, dim)).repeat(n, axis=1).reshape((m * n, dim))
    return np.cast['float32'](source_points)


def sample_conditional_fix_embeddings(generator, m, n,
                                      embedding_file=DEFAULT_EMBEDDING_FILE,
                                      noise_range=1, **kwargs):
    """
    Sample `m * n` points in condition space by retrieving `m` points
    from a provided dataset and adding small random noise `n - 1` times
    for each point.
    """

    source_points, dim = get_embeddings(embedding_file, m)

    noisy_points = np.copy(source_points).reshape((m, 1, dim)).repeat(n - 1, axis=1)
    noisy_points += noise_range * (np.random.rand(*noisy_points.shape) * 2. - 1.)

    ret = (np.concatenate((source_points.reshape((m, 1, dim)), noisy_points), axis=1)
             .reshape((m * n, dim)))

    return np.cast['float32'](ret)


# Build string mapping for conditional samplers so that they can be
# triggered from CLI
conditional_samplers = {
    'random': sample_conditional_random,
    'fix_random': sample_conditional_fix_random,
    'fix_embeddings': sample_conditional_fix_embeddings,
    'fix_embeddings_nonoise': sample_conditional_fix_embeddings_no_noise,
}


def get_conditional_topo_samples(generator, m, n, condition_sampler_fn,
                                 embedding_file=DEFAULT_EMBEDDING_FILE):
    if isinstance(generator, basestring):
        generator = load_generator_from_file(generator)

    conditional_batch = generator.condition_space.make_theano_batch()
    conditional_data = condition_sampler_fn(generator, m, n,
                                            embedding_file=embedding_file)

    topo_samples_batch = generator.sample(conditional_batch)
    topo_sample_f = theano.function([conditional_batch], topo_samples_batch)
    topo_samples = topo_sample_f(conditional_data).swapaxes(0, 3)
    return topo_samples, conditional_data
