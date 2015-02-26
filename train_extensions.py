
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T

from adversarial import Generator


class GenerateAndSave(TrainExtension):
    """
    Keeps track of what the generator in a (vanilla) GAN returns for a
    particular set of noise values.
    """

    def __init__(self, generator, batch_size=25):
        assert isinstance(generator, ConditionalGenerator)

        self.batch_sym = T.matrix('generate_batch')
        self.generate_f = theano.function([self.batch_sym],
                                          generator.dropout_fprop(self.batch_sym))

        self.batch = generator.get_noise(batch_size).eval()

    def on_monitor(self, model, dataset, algorithm):
        samples = self.generate_f(self.batch)

        # TODO swap axes and save samples as images
