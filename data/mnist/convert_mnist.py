import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import paths
from common import utils
import numpy
import gzip


def convert_dataset():
    """
    Convert MNIST.
    """

    filenames = [
        [paths.raw_mnist_train_images_file(), paths.mnist_train_images_file()],
        [paths.raw_mnist_test_images_file(), paths.mnist_test_images_file()],
        [paths.raw_mnist_train_labels_file(), paths.mnist_train_labels_file()],
        [paths.raw_mnist_test_labels_file(), paths.mnist_test_labels_file()]
    ]
    for names in filenames[:2]:
        with gzip.open(names[0], 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16).reshape(-1, 28, 28, 1)
            #data = data.swapaxes(1, 2)
            data = data.astype(numpy.float32)/255.
            utils.write_hdf5(names[1], data)
            log('wrote %s' % names[1])
    for names in filenames[-2:]:
        with gzip.open(names[0], 'rb') as f:
            utils.write_hdf5(names[1], numpy.frombuffer(f.read(), numpy.uint8, offset=8).reshape(-1, 1).astype(numpy.int))
            log('wrote %s' % names[1])


if __name__ == '__main__':
    convert_dataset()