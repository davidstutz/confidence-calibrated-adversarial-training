"""
Some I/O utilities.
"""

import os
import json
import numpy as np
import importlib
import pickle
import gc
import socket
import functools

# See https://github.com/h5py/h5py/issues/961
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def write_hdf5(filepath, tensors, keys='tensor'):
    """
    Write a simple tensor, i.e. numpy array ,to HDF5.

    :param filepath: path to file to write
    :type filepath: str
    :param tensors: tensor to write
    :type tensors: numpy.ndarray or [numpy.ndarray]
    :param keys: key to use for tensor
    :type keys: str or [str]
    """

    #opened_hdf5() # To be sure as there were some weird opening errors.
    assert type(tensors) == np.ndarray or isinstance(tensors, list)
    if isinstance(tensors, list) or isinstance(keys, list):
        assert isinstance(tensors, list) and isinstance(keys, list)
        assert len(tensors) == len(keys)

    if not isinstance(tensors, list):
        tensors = [tensors]
    if not isinstance(keys, list):
        keys = [keys]

    makedir(os.path.dirname(filepath))

    # Problem that during experiments, too many h5df files are open!
    # https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
    with h5py.File(filepath, 'w') as h5f:

        for i in range(len(tensors)):
            tensor = tensors[i]
            key = keys[i]

            chunks = list(tensor.shape)
            if len(chunks) > 2:
                chunks[2] = 1
                if len(chunks) > 3:
                    chunks[3] = 1
                    if len(chunks) > 4:
                        chunks[4] = 1

            h5f.create_dataset(key, data=tensor, chunks=tuple(chunks), compression='gzip')
        h5f.close()
        return


def read_hdf5(filepath, key='tensor', efficient=False):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param filepath: path to file to read
    :type filepath: str
    :param key: key to read
    :type key: str
    :param efficient: effienct reaidng
    :type efficient: bool
    :return: tensor
    :rtype: numpy.ndarray
    """

    #opened_hdf5() # To be sure as there were some weird opening errors.
    assert os.path.exists(filepath), 'file %s not found' % filepath

    if efficient:
        h5f = h5py.File(filepath, 'r')
        assert key in [key for key in h5f.keys()], 'key %s does not exist in %s with keys %s' % (key, filepath, ', '.join(h5f.keys()))
        return h5f[key]
    else:
        with h5py.File(filepath, 'r') as h5f:
            assert key in [key for key in h5f.keys()], 'key %s does not exist in %s with keys %s' % (key, filepath, ', '.join(h5f.keys()))
            return h5f[key][()]


def check_hdf5(filepath, key='tensor'):
    """
    Check a file without loading data.

    :param filepath: path to file to read
    :type filepath: str
    :param key: key to read
    :type key: str
    :return: can be loaded or not
    :rtype: bool
    """

    opened_hdf5()  # To be sure as there were some weird opening errors.
    assert os.path.exists(filepath), 'file %s not found' % filepath

    try:
        with h5py.File(filepath, 'r') as h5f:
            assert key in [key for key in h5f.keys()], 'key %s does not exist in %s' % (key, filepath)
            tensor = h5f.get('tensor')
            # That's it ...
            return True
    except:
        return False


def opened_hdf5():
    """
    Close all open HDF5 files and report number of closed files.

    :return: number of closed files
    :rtype: int
    """

    opened = 0
    for obj in gc.get_objects():  # Browse through ALL objects
        try:
            # is instance check may also fail!
            if isinstance(obj, h5py.File):  # Just HDF5 files
                obj.close()
                opened += 1
        except:
            pass  # Was already closed
    return opened


def write_pickle(file, mixed):
    """
    Write a variable to pickle.

    :param file: path to file to write
    :type file: str
    :return: mixed
    :rtype: mixed
    """

    makedir(os.path.dirname(file))
    handle = open(file, 'wb')
    pickle.dump(mixed, handle)
    handle.close()


def read_pickle(file):
    """
    Read pickle file.

    :param file: path to file to read
    :type file: str
    :return: mixed
    :rtype: mixed
    """

    assert os.path.exists(file), 'file %s not found' % file

    handle = open(file, 'rb')
    results = pickle.load(handle)
    handle.close()
    return results


def read_json(file):
    """
    Read a JSON file.

    :param file: path to file to read
    :type file: str
    :return: parsed JSON as dict
    :rtype: dict
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        return json.load(fp)


def write_json(file, data):
    """
    Read a JSON file.

    :param file: path to file to read
    :type file: str
    :param data: data to write
    :type data: mixed
    :return: parsed JSON as dict
    :rtype: dict
    """

    makedir(os.path.dirname(file))
    with open(file, 'w') as fp:
        json.dump(data, fp)


def makedir(dir):
    """
    Creates directory if it does not exist.

    :param dir: directory path
    :type dir: str
    """

    if dir and not os.path.exists(dir):
        os.makedirs(dir)


def remove(filepath):
    """
    Remove a file.

    :param filepath: path to file
    :type filepath: str
    """

    if os.path.isfile(filepath) and os.path.exists(filepath):
        os.unlink(filepath)


def get_class(module_name, class_name):
    """
    See https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa.

    :param module_name: module holding class
    :type module_name: str
    :param class_name: class name
    :type class_name: str
    :return: class or False
    """
    # load the module, will raise ImportError if module cannot be loaded
    try:
        m = importlib.import_module(module_name)
    except ImportError as e:
        log('%s' % e, LogLevel.ERROR)
        return False
    # get the class, will raise AttributeError if class cannot be found
    try:
        c = getattr(m, class_name)
    except AttributeError as e:
        log('%s' % e, LogLevel.ERROR)
        return False
    return c


def hostname():
    """
    Get hostname.

    :return: hostname
    :rtype: str
    """

    return socket.gethostname()


def pid():
    """
    PID.

    :return: PID
    :rtype: int
    """

    return os.getpid()


def partial(f, *args, **kwargs):
    """
    Create partial while preserving __name__ and __doc__.

    :param f: function
    :type f: callable
    :param args: arguments
    :type args: dict
    :param kwargs: keyword arguments
    :type kwargs: dict
    :return: partial
    :rtype: callable
    """
    p = functools.partial(f, *args, **kwargs)
    functools.update_wrapper(p, f)
    return p