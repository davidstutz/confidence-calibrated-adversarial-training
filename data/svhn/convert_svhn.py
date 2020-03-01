import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import paths
from common import utils
import numpy
import scipy.io as sio


def convert_dataset():
    """
    Convert SVHN.
    """

    data = sio.loadmat(paths.raw_svhn_train_file())

    # access to the dict
    images = data['X']
    images = images.transpose(3, 0, 1, 2)
    images = images/255.
    labels = data['y'] - 1
    #print(images[0], numpy.max(images), numpy.min(images))

    utils.write_hdf5(paths.svhn_train_images_file(), images.astype(numpy.float32))
    log('wrote %s' % paths.svhn_train_images_file())
    utils.write_hdf5(paths.svhn_train_labels_file(), labels.reshape(-1, 1).astype(numpy.int))
    log('wrote %s' % paths.svhn_train_labels_file())

    data = sio.loadmat(paths.raw_svhn_test_file())

    # access to the dict
    images = data['X']
    images = images.transpose(3, 0, 1, 2)
    images = images/255.
    labels = data['y'] - 1

    utils.write_hdf5(paths.svhn_test_images_file(), images.astype(numpy.float32))
    log('wrote %s' % paths.svhn_test_images_file())
    utils.write_hdf5(paths.svhn_test_labels_file(), labels.reshape(-1, 1).astype(numpy.int))
    log('wrote %s' % paths.svhn_test_labels_file())


if __name__ == '__main__':
    convert_dataset()