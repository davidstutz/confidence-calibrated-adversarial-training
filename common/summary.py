import numpy
import os
from .log import log
from . import utils
import pickle
import torch


def to_dict(object, classkey='__class__'):
    """
    Get dict recursively from object.
    https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

    :param object: object
    :type object: object or dict
    :param classkey: save class name in this key
    :type classkey: str
    :return: object as dict
    :rtype: ditct
    """

    if isinstance(object, dict):
        data = {}
        for (k, v) in object.items():
            data[k] = to_dict(v, classkey)
        return data
    #elif hasattr(object, '_ast'):
    #    return to_dict(object._ast())
    #elif hasattr(object, '__iter__') and not isinstance(object, str):
    #    return [to_dict(v, classkey) for v in object]
    elif hasattr(object, '__dict__'):
        data = dict([(key, to_dict(value, classkey)) for key, value in object.__dict__.items() if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(object, '__class__'):
            data[classkey] = object.__class__.__name__
        return data
    else:
        return object


class SummaryWriter:
    """
    Summary dummy or interface to work like the Tensorboard SummaryWriter.
    """

    def __init__(self, log_dir=''):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        pass

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_histogram(self, tag, values, global_step=None, bins='auto', walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        pass

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def flush(self):
        """
        Flush.
        """

        pass


class SummaryWriters(SummaryWriter):

    def __init__(self, writers):
        """
        Constructor.

        :param writers: writers
        :type: [SummaryWriter]
        """

        assert len(writers) > 0

        self.writers = writers
        """ ([SummaryWriter]) Writers. """

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        for writer in self.writers:
            writer.add_scalar(tag, value, global_step=global_step, walltime=walltime)

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        for writer in self.writers:
            writer.add_scalars(tag, tag_scalar_dict, global_step=global_step, walltime=walltime)

    def add_histogram(self, tag, values, global_step=None, bins='auto', walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        for writer in self.writers:
            writer.add_histogram(tag, values, global_step=global_step, bins=bins, walltime=walltime, max_bins=max_bins)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        for writer in self.writers:
            writer.add_image(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        for writer in self.writers:
            writer.add_images(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        for writer in self.writers:
            writer.add_figure(tag, figure, global_step=global_step, close=close, walltime=walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        for writer in self.writers:
            writer.add_text(tag, text_string, global_step=global_step, walltime=walltime)


class SummaryLogWriter(SummaryWriter):
    """
    Log writer.
    """

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        log('%s: %g' % (tag, value))


class SummaryDictWriter:
    """
    This is a very simple pickle based writer; unfortunately, it requries an
    additional function call for updating the record, which is usually not
    needed for tensorboard's summary stuff.
    """

    def __init__(self, log_dir=''):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        self.data = dict()
        """ (dict) Data."""

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, value))

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, tag_scalar_dict))

    # auto binning does not seem to work well on different networks, numpy.histogram frequently raises errors
    def add_histogram(self, tag, values, global_step=None, bins=100, walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        values = values.reshape(-1)
        assert values.shape[0] > 0

        self.data[tag].append((global_step, numpy.histogram(values, bins=bins)[0]))

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        assert dataformats == 'CHW' or dataformats == 'HWC' or dataformats == 'HW'

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        # CHW, HWC, HW
        if dataformats == 'CHW':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.transpose(img_tensor, (1, 2, 0))
        elif dataformats == 'HWC':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.transpose(img_tensor, (1, 2, 0))
        elif dataformats == 'HW':
            assert len(img_tensor.shape) == 2
            img_tensor = numpy.expand_dims(img_tensor, axis=0)

        self.data[tag].append((global_step, img_tensor))

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        assert dataformats == 'NCHW' or dataformats == 'NHWC' or dataformats == 'NHW'

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        # CHW, HWC, HW
        if dataformats == 'NCHW':
            assert len(img_tensor.shape) == 4
            img_tensor = numpy.transpose(img_tensor, (0, 2, 3, 1))
        elif dataformats == 'NHWC':
            assert len(img_tensor.shape) == 4
            img_tensor = numpy.transpose(img_tensor, (0, 1, 2, 3))
        elif dataformats == 'NHW':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.expand_dims(img_tensor, axis=1)

        self.data[tag].append((global_step, img_tensor))

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, text_string))


class SummaryPickleWriter(SummaryDictWriter):
    """
    This is a very simple pickle based writer; unfortunately, it requries an
    additional function call for updating the record, which is usually not
    needed for tensorboards summary stuff.
    """

    def __init__(self, log_dir, max_queue=50):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        log_file = os.path.join(log_dir, 'events.pkl')
        utils.makedir(os.path.dirname(log_file))

        self.log_file = log_file
        """ (str) File to write to. """

        self.max_queue = max_queue
        """ (int) Max queue. """

        self.queue = 0
        """ (int) Queue. """

        self.data = dict()
        """ (dict) Data."""

    #def __del__(self):
    #    """
    #    Destructor.
    #    """
    #
    #    self.flush()

    def update_queue(self):
        """
        Update queue.
        """

        self.queue += 1
        if self.queue >= self.max_queue:
            self.flush()
            self.queue = 0

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, value))
        self.update_queue()

    def add_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, tag_scalar_dict))
        self.update_queue()

    # auto binning does not seem to work well on different networks, numpy.histogram frequently raises errors
    def add_histogram(self, tag, values, global_step=None, bins=100, walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        values = values.reshape(-1)
        assert values.shape[0] > 0

        self.data[tag].append((global_step, numpy.histogram(values, bins=bins)[0]))
        self.update_queue()

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        assert dataformats == 'CHW' or dataformats == 'HWC' or dataformats == 'HW'

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        # CHW, HWC, HW
        if dataformats == 'CHW':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.transpose(img_tensor, (1, 2, 0))
        elif dataformats == 'HWC':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.transpose(img_tensor, (1, 2, 0))
        elif dataformats == 'HW':
            assert len(img_tensor.shape) == 2
            img_tensor = numpy.expand_dims(img_tensor, axis=0)

        self.data[tag].append((global_step, img_tensor))
        self.update_queue()

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        assert dataformats == 'NCHW' or dataformats == 'NHWC' or dataformats == 'NHW'

        if not tag in self.data:
            self.data[tag] = []

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        # CHW, HWC, HW
        if dataformats == 'NCHW':
            assert len(img_tensor.shape) == 4
            img_tensor = numpy.transpose(img_tensor, (0, 2, 3, 1))
        elif dataformats == 'NHWC':
            assert len(img_tensor.shape) == 4
            img_tensor = numpy.transpose(img_tensor, (0, 1, 2, 3))
        elif dataformats == 'NHW':
            assert len(img_tensor.shape) == 3
            img_tensor = numpy.expand_dims(img_tensor, axis=1)

        self.data[tag].append((global_step, img_tensor))
        self.update_queue()

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        if not tag in self.data:
            self.data[tag] = []

        self.data[tag].append((global_step, text_string))
        self.update_queue()

    def flush(self):
        """
        Save.
        """

        with open(self.log_file, 'ab') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.data.clear()
            assert len(self.data) == 0


class SummaryReader:
    """
    Summary reader.
    """

    def __init__(self, log_dir=''):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        pass

    def get_scalar(self, tag, value, global_step=None, walltime=None):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        :param value: value
        :type value: mixed
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def get_scalars(self, tag, tag_scalar_dict, global_step=None, walltime=None):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        :param tag_scalar_dict: values
        :type tag_scalar_dict: dict
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def get_histogram(self, tag, values, global_step=None, bins='auto', walltime=None, max_bins=None):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        :param values: values
        :type values: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param bins: binning method
        :type bins: str
        :param walltime: time
        :type walltime: int
        :param max_bins: maximum number of bins
        :type max_bins: int
        """

        pass

    def get_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """
        Add image.

        :param tag: tag
        :type tag: str
        :param img_tensor: image
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def get_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param img_tensor: images
        :type img_tensor: torch.Tensor or numpy.ndarray
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        :param dataformats: format of image
        :type dataformats: str
        """

        pass

    def get_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        :param figure: test
        :type figure: matplotlib.pyplot.figure
        :param global_step: global step
        :type global_step: int
        :param close: whether to automatically close figure
        :type close: bool
        :param walltime: time
        :type walltime: int
        """

        pass

    def get_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add images.

        :param tag: tag
        :type tag: str
        :param text_string: test
        :type text_string: str
        :param global_step: global step
        :type global_step: int
        :param walltime: time
        :type walltime: int
        """

        pass

    def flush(self):
        """
        Flush.
        """

        pass


class SummaryDictReader(SummaryReader):
    """
        Summary reader.
        """

    def __init__(self, data):
        """
        Constructor.

        :param data: summary directory
        :type data: str
        """

        self.data = data
        """ (dict) Data. """

    def get_scalar(self, tag):
        """
        Add scalar value.

        :param tag: tag for scalar
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_scalars(self, tag):
        """
        Add scalar values.

        :param tag: tag for scalar
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_histogram(self, tag):
        """
        Add histogram data.

        :param tag: tag
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_image(self, tag):
        """
        Add image.

        :param tag: tag
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_images(self, tag):
        """
        Add images.

        :param tag: tag
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_figure(self, tag):
        """
        Add figure.

        :param tag: tag
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]

    def get_text(self, tag):
        """
        Add images.

        :param tag: tag
        :type tag: str
        """

        assert tag in self.data.keys()
        return self.data[tag]


class SummaryPickleReader(SummaryDictReader):
    """
    Summary reader.
    """

    def __init__(self, log_dir):
        """
        Constructor.

        :param log_dir: summary directory
        :type log_dir: str
        """

        log_file = os.path.join(log_dir, 'events.pkl')
        assert os.path.exists(log_file)

        data = dict()
        if os.path.exists(log_file):
            with open(log_file, 'rb') as handle:
                data = pickle.load(handle)

        self.log_file = log_file
        """ (str) File to write to. """

        self.data = data
        """ (dict) Data."""