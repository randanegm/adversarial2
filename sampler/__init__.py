"""
Defines shared utilities for sampling from a vGAN or cGAN generator
component.
"""

import numpy as np
import theano


def sample_conditional_fix_random(generator, m, n, noise_range=1):
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


def sample_conditional_fix_embeddings(generator, m, n,
                                      embedding_file='/afs/cs.stanford.edu/u/jgauthie/scr/lfw-lsa/LFW_attributes_30d.npz',
                                      noise_range=1):
    """
    Sample `m * n` points in condition space by retrieving `m` points
    from a provided dataset and adding small random noise `n - 1` times
    for each point.
    """

    embs = np.load(embedding_file)['arr_0']
    np.random.shuffle(embs)

    source_points = embs[:m]
    dim = source_points.shape[1]

    noisy_points = np.copy(source_points).reshape((m, 1, dim)).repeat(n - 1, axis=1)
    noisy_points += noise_range * (np.random.rand(*noisy_points.shape) * 2. - 1.)

    ret = (np.concatenate((source_points.reshape((m, 1, dim)), noisy_points), axis=1)
             .reshape((m * n, dim)))

    return np.cast['float32'](ret)


def get_conditional_topo_samples(generator, m, n, condition_sampler_fn):
    conditional_batch = generator.condition_space.make_theano_batch()
    conditional_data = args.conditional_sampler(generator, rows, sample_cols)

    topo_samples_batch = generator.sample(conditional_batch)
    topo_sample_f = theano.function([conditional_batch], topo_samples_batch)
    topo_samples = topo_sample_f(conditional_data).swapaxes(0, 3)
