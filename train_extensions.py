
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T

from adversarial import Generator


class GenerateAndSave(TrainExtension):
    """
    Keeps track of what the generator in a (vanilla) GAN returns for a
    particular set of noise values.
    """

    def __init__(self, generator, save_prefix, batch_size=20, grid_shape=(5, 4)):
        assert isinstance(generator, Generator)

        self.batch_sym = T.matrix('generate_batch')
        self.generate_f = theano.function([self.batch_sym],
                                          generator.dropout_fprop(self.batch_sym)[0])

        self.batch = generator.get_noise(batch_size).eval()
        self.save_prefix = save_prefix
        self.patch_viewer = PatchViewer(grid_shape=grid_shape, patch_shape=(32, 32),
                                        is_color=True)

    def on_monitor(self, model, dataset, algorithm):
        samples = self.generate_f(self.batch).swapaxes(0, 3)

        self.patch_viewer.clear()
        for sample in samples:
            self.patch_viewer.add_patch(sample, rescale=True)

        fname = self.save_prefix + '.%05i.png' % model.monitor.get_epochs_seen()
        self.patch_viewer.save(fname)
