import os
from .log import log, LogLevel

# This file holds a bunch of specific paths used for experiments and
# data. The intention is to have all important paths at a central location, while
# allowing to easily prototype new experiments.

# base directory for data
BASE_DATA = ''
# base directory for experiments (model files, adversarial examples etc.)
BASE_EXPERIMENTS = ''
# base directory for logs (mainly tensorboard for training)
BASE_LOGS = ''

if not os.path.exists(BASE_DATA):
    BASE_DATA = os.path.join(os.path.expanduser('~'), 'data') + '/'
    log('[Warning] changed data directory: %s' % BASE_DATA, LogLevel.WARNING)
    BASE_EXPERIMENTS = os.path.join(os.path.expanduser('~'), 'experiments') + '/'
    log('[Warning] changed experiments directory: %s' % BASE_EXPERIMENTS, LogLevel.WARNING)

if not os.path.exists(BASE_DATA):
    log('[Error] could not find data directory %s' % BASE_DATA, LogLevel.ERROR)
    raise Exception('Data directory %s not found.' % BASE_DATA)

# Common extension types used.
TXT_EXT = '.txt'
HDF5_EXT = '.h5'
STATE_EXT = '.pth.tar'
LOG_EXT = '.log'
PNG_EXT = '.png'
PICKLE_EXT = '.pkl'
TEX_EXT = '.tex'
MAT_EXT = '.mat'
GZIP_EXT = '.gz'


# Naming conventions.
def data_file(name, ext=HDF5_EXT):
    """
    Generate path to data file.

    :param name: name of file
    :type name: str
    :param ext: extension (including period)
    :type ext: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_DATA, name) + ext


def raw_mnistc_dir():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist_c/', '')


def raw_mnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train-images-idx3-ubyte', GZIP_EXT)


def raw_mnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/t10k-images-idx3-ubyte', GZIP_EXT)


def raw_mnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train-labels-idx1-ubyte', GZIP_EXT)


def raw_mnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/t10k-labels-idx-ubyte', GZIP_EXT)


def mnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train_images', HDF5_EXT)


def mnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/test_images', HDF5_EXT)


def mnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train_labels', HDF5_EXT)


def mnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/test_labels', HDF5_EXT)


def raw_cifar10c_dir():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10_C/', '')


def raw_cifar10_dir():
    """
    Raw Cifar10 training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('Cifar10/', '')


def cifar10_train_images_file():
    """
    Cifar10 train images.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/train_images', HDF5_EXT)


def cifar10_test_images_file():
    """
    Cifar10 test images.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/test_images', HDF5_EXT)


def cifar10_train_labels_file():
    """
    Cifar10 train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/train_labels', HDF5_EXT)


def cifar10_test_labels_file():
    """
    Cifar10 test labels.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/test_labels', HDF5_EXT)


def raw_svhn_train_file():
    """
    Raw SVHN training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('svhn/train_32x32', '.mat')


def raw_svhn_test_file():
    """
    Raw SVHN training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('svhn/test_32x32', '.mat')


def svhn_train_images_file():
    """
    SVHN train images.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/train_images', HDF5_EXT)


def svhn_test_images_file():
    """
    SVHN test images.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/test_images', HDF5_EXT)


def svhn_train_labels_file():
    """
    SVHN train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/train_labels', HDF5_EXT)


def svhn_test_labels_file():
    """
    SVHN test labels.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/test_labels', HDF5_EXT)


def random_train_images_file(N, size):
    """
    Random train images.

    :return: filepath
    :rtype: str
    """

    return data_file('random/train_images_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def random_test_images_file(N, size):
    """
    Random test images.

    :return: filepath
    :rtype: str
    """

    return data_file('random/test_images_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def random_train_labels_file(N, size):
    """
    Random train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('random/train_labels_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def random_test_labels_file(N, size):
    """
    Random test labels.

    :return: filepath
    :rtype: str
    """

    return data_file('random/test_labels_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def experiment_dir(directory):
    """
    Generate path to experiment directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory)


def experiment_file(directory, name, ext=''):
    """
    Generate path to experiment file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory, name) + ext


def log_dir(directory):
    """
    Generate path to log directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_LOGS, directory)


def log_file(directory, name, ext=''):
    """
    Generate path to log file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_LOGS, directory, name) + ext