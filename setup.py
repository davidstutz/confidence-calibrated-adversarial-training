def check_simple():
    """
    Checks packages that can be installed via PIP.
    """

    print('Importing simple stuff ...')
    import os
    import re
    import sys
    import json
    import numpy
    import zipfile
    import importlib
    import datetime
    import copy
    import time
    import enum
    import inspect
    import argparse
    import math
    import random
    import functools
    import warnings
    import pickle
    import gc
    import wget
    import gzip
    import unittest
    import sklearn
    import sklearn.svm
    import sklearn.multioutput
    import sklearn.decomposition
    import sklearn.manifold
    import sklearn.metrics
    import skimage
    import skimage.transform
    import imageio
    import scipy
    import scipy.ndimage
    import scipy.stats
    import scipy.special
    print('Imported simple stuff.')


def check_h5py():
    """
    Check h5py installation.
    """

    try:
        print('Importing h5py ...')
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        import h5py
        warnings.resetwarnings()
        print('h5py imported.')
        print('If you still get an error when importing h5py (e.g. saying that numpy has not attribute dtype), try uninstalling h5py, numpy and six, and reinstalling it using pip3.')
    except ImportError as e:
        print('Could not load h5py; it might to be installed manually and might have problems with the installed NumPy version.')


def check_torch():
    """
    Check torch installation.
    """

    try:
        print('Importing torch ...')
        import torch
        import numpy
        print('Torch imported.')

        if torch.cuda.is_available():
            print('CUDA seems to be available and supported.')

        # Might print something like
        # BS/dstutz/work/dev-box/pip9/lib/python2.7/site-packages/torch/cuda/__init__.py:89: UserWarning:
        #     Found GPU0 Tesla V100-PCIE-16GB which requires CUDA_VERSION >= 8000 for
        #     optimal performance and fast startup time, but your PyTorch was compiled
        #     with CUDA_VERSION 8000. Please install the correct PyTorch binary
        #     using instructions from http://pytorch.org
        # or:
        # Segmentation fault (not sure what's the cause).

        target = torch.from_numpy(numpy.array([[0, 0], [0, 1], [0, 1]], dtype=float))
        target = target.cuda()

        import torchvision
        print('Torchvision imported.')

        assert torch.__version__

        print('Unless there were warnings or segmentation faults, everything works!')
    except ImportError as e:
        print('Torch could not be imported.')

    tensor = torch.randn(32, 3, 32, 32)
    tensor = tensor.cuda()
    tensor = tensor + torch.ones_like(tensor)
    tensor = tensor*torch.ones_like(tensor)
    tensor = torch.relu(tensor)
    print('Basic CUDA operations seem to be working.')


def check_tensorboard():
    try:
        print('Importing tensorflow ...')
        import tensorflow
        print('Tensorflow imported.')
    except ImportError as e:
        print('Tensorflow could not be imported!')
    try:
        print('Importing tensorboard ...')
        import tensorboard
        import tensorboard.backend.event_processing.event_accumulator
        print('Imported tensorboard.')
    except ImportError as e:
        print('Tensorboard could not be imported!')
    try:
        print('Importing torch.utils.tensorboard ...')
        import torch.utils.tensorboard
        print('torch.utils.tensorboard imported.')
    except ImportError as e:
        print('torch.utils.tensorboard could not be imported!')


def check_common():
    import os
    import sys
    sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')

    print('Importing common stuff ...')
    import common.eval
    import common.train
    import common.datasets
    import common.experiments
    import common.imgaug
    import common.log
    import common.numpy
    import common.paths
    import common.state
    import common.summary
    import common.test
    import common.timer
    import common.torch
    import common.utils

    from common.log import log, LogLevel
    log('BASE_EXPERIMENTS=%s' % common.paths.BASE_EXPERIMENTS)
    log('BASE_DATA=%s' % common.paths.BASE_DATA)
    print('Imported common.')


if __name__ == '__main__':
    check_simple()
    check_h5py()
    check_torch()
    check_tensorboard()
    check_common()