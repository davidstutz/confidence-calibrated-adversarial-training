import os
import torch
import torch.utils.data # needs to be imported separately
from . import utils
import numpy
from . import paths
import skimage.transform
from .log import log
from . import numpy as cnumpy


class TransformedTensorDataset(torch.utils.data.Dataset):
    """
    TensorDataset with support for transforms.
    """

    def __init__(self, tensors, transform=None):
        assert len(tensors) > 1
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        result = []
        for i in range(len(self.tensors) - 1):
            xi = self.tensors[i][index]
            if self.transform:
                xi = self.transform(xi)
                result.append(xi)

        y = self.tensors[-1][index]
        result.append(y)

        return result

    def __len__(self):
        return self.tensors[0].size(0)


class CleanDataset(torch.utils.data.Dataset):
    """
    General, clean dataset used for training, testing and attacking.
    """

    def __init__(self, images, labels, indices=None, resize=None):
        """
        Constructor.

        :param images: images/inputs
        :type images: str or numpy.ndarray
        :param labels: labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        :param resize: resize in [channels, height, width
        :type resize: resize
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
            log('read %s' % self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
            log('read %s' % self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        images = images[indices]
        labels = labels[indices]

        if resize is not None:
            assert isinstance(resize, list)
            assert len(resize) == 3

            size = images.shape
            assert len(size) == 4

            # resize
            if resize[1] != size[1] or resize[2] != size[2]:
                out_images = numpy.zeros((size[0], resize[1], resize[2], size[3]), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n] = skimage.transform.resize(images[n], (resize[1], resize[2]))
                images = out_images

            # update!
            size = images.shape

            # color to grayscale
            if resize[0] == 1 and size[3] == 3:
                out_images = numpy.zeros((size[0], size[1], size[2], 1), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = 0.2125 * images[n, :, :, 0] + 0.7154 * images[n, :, :, 1] + 0.0721 * images[n, :, :, 2]
                images = out_images

            # grayscale to color
            if resize[0] == 3 and size[3] == 1:
                out_images = numpy.zeros((size[0], size[1], size[2], 3), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = images[n, :, :, 0]
                    out_images[n, :, :, 1] = images[n, :, :, 0]
                    out_images[n, :, :, 2] = images[n, :, :, 0]
                images = out_images

        self.images = images
        """ (numpy.ndarray) Inputs. """

        self.labels = labels
        """ (numpy.ndarray) Labels. """

        self.targets = None
        """ (numpy.ndarray) Possible attack targets. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

    def add_targets(self, targets):
        """
        Add attack targets.
        :param targets: targets
        :type targets: numpy.ndarray
        """

        assert numpy.min(self.indices) >= 0
        assert numpy.max(self.indices) < targets.shape[0]
        self.targets = targets[self.indices]

    def __getitem__(self, index):
        assert index < len(self)
        if self.targets is not None:
            return self.images[index], self.labels[index], self.targets[index]
        else:
            return self.images[index], self.labels[index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class AdversarialDataset(torch.utils.data.Dataset):
    """
    Dataset consisting of adversarial examples.
    """

    def __init__(self, images, perturbations, labels, indices=None):
        """
        Constructor.

        :param images: images
        :type images: str or numpy.ndarray
        :param perturbations: additive perturbations
        :type perturbations: str or numpy.ndarray
        :param labels: true labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.perturbations_file = None
        """ (str) File perturbations were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(images, str):
            self.perturbations_file = perturbations
            perturbations = utils.read_hdf5(self.perturbations_file)
        if not perturbations.dtype == numpy.float32:
            perturbations = perturbations.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(perturbations, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]
        assert len(perturbations.shape) == len(images.shape) + 1
        for d in range(len(images.shape)):
            assert perturbations.shape[d + 1] == images.shape[d]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        self.images = images[indices]
        """ (numpy.ndarray) Inputs. """

        self.perturbations = perturbations[:, indices]
        """ (numpy.ndarray) Perturbations. """

        self.labels = labels[indices]
        """ (numpy.ndarray) Labels. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

    def __getitem__(self, index):
        assert index < len(self)
        attempt_index = index // self.images.shape[0]
        sample_index = index % self.images.shape[0]
        assert attempt_index < self.perturbations.shape[0]
        assert sample_index < self.images.shape[0]

        return self.perturbations[attempt_index, sample_index] + self.images[sample_index], self.labels[sample_index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        assert self.images.shape[0] == self.perturbations.shape[1]
        return self.perturbations.shape[0]*self.perturbations.shape[1]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class RandomTrainSet(CleanDataset):
    def __init__(self, N, size):
        train_images_file = paths.random_train_images_file(N, size)
        train_labels_file = paths.random_train_labels_file(N, size)

        if not os.path.exists(train_images_file):
            train_images = numpy.random.uniform(0, 1, size=[N] + list(size))
            utils.write_hdf5(train_images_file, train_images)

        if not os.path.exists(train_labels_file):
            train_labels = numpy.random.randint(0, 9, size=(N, 1))
            utils.write_hdf5(train_labels_file, train_labels)

        super(RandomTrainSet, self).__init__(train_images_file, train_labels_file)


class RandomTestSet(CleanDataset):
    def __init__(self, N, size):
        test_images_file = paths.random_test_images_file(N, size)
        test_labels_file = paths.random_test_labels_file(N, size)

        if not os.path.exists(test_images_file):
            test_images = numpy.random.uniform(0, 1, size=[N] + list(size))
            utils.write_hdf5(test_images_file, test_images)

        if not os.path.exists(test_labels_file):
            test_labels = numpy.random.randint(0, 9, size=(N, 1))
            utils.write_hdf5(test_labels_file, test_labels)

        super(RandomTestSet, self).__init__(test_images_file, test_labels_file)


class MNISTTrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(MNISTTrainSet, self).__init__(paths.mnist_train_images_file(), paths.mnist_train_labels_file(), indices=indices, resize=resize)


class MNISTTestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(MNISTTestSet, self).__init__(paths.mnist_test_images_file(), paths.mnist_test_labels_file(), indices=indices, resize=resize)


class MNISTCTestSet(CleanDataset):
    def __init__(self, corruptions=None, indices=None, resize=None):

        allowed_corruptions = [
            'brightness',
            'canny_edges',
            'dotted_line',
            'fog',
            'glass_blur',
            #'identity',
            'impulse_noise',
            'motion_blur',
            'rotate',
            'scale',
            'shear',
            'shot_noise',
            'spatter',
            'stripe',
            'translate',
            'zigzag'
        ]
        if corruptions is None:
            corruptions = allowed_corruptions
        for corruption in corruptions:
            assert corruption in allowed_corruptions

        self.corruptions = corruptions
        """ ([str]) Corruptions. """

        images = None
        labels = None

        for corruption in self.corruptions:
            images_file = os.path.join(paths.raw_mnistc_dir(), corruption + '/test_images.npy')
            labels_file = os.path.join(paths.raw_mnistc_dir(), corruption + '/test_labels.npy')

            corruption_images = numpy.load(images_file)
            log('read %s' % images_file)
            corruption_labels = numpy.load(labels_file)
            log('read %s' % labels_file)

            corruption_images = corruption_images.astype(numpy.float32)/255.

            if indices is not None:
                images = cnumpy.concatenate(images, corruption_images[indices])
                labels = cnumpy.concatenate(labels, corruption_labels[indices])
            else:
                images = cnumpy.concatenate(images, corruption_images)
                labels = cnumpy.concatenate(labels, corruption_labels)

        super(MNISTCTestSet, self).__init__(images, labels, indices=None, resize=resize)


class FashionMNISTTrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(FashionMNISTTrainSet, self).__init__(paths.fashion_train_images_file(), paths.fashion_train_labels_file(), indices=indices, resize=resize)


class FashionMNISTTestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(FashionMNISTTestSet, self).__init__(paths.fashion_test_images_file(), paths.fashion_test_labels_file(), indices=indices, resize=resize)


class Cifar10TrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(Cifar10TrainSet, self).__init__(paths.cifar10_train_images_file(), paths.cifar10_train_labels_file(), indices=indices, resize=resize)


class Cifar10TestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(Cifar10TestSet, self).__init__(paths.cifar10_test_images_file(), paths.cifar10_test_labels_file(), indices=indices, resize=resize)


class Cifar10CTestSet(CleanDataset):
    def __init__(self, corruptions=None, indices=None, resize=None):

        allowed_corruptions = [
            'brightness',
            'contrast',
            'defocus_blur',
            'elastic_transform',
            'fog',
            'frost',
            'gaussian_blur',
            'gaussian_noise',
            'glass_blur',
            'impulse_noise',
            'jpeg_compression',
            'motion_blur',
            'pixelate',
            'saturate',
            'shot_noise',
            'snow',
            'spatter',
            'speckle_noise',
            'zoom_blur',
        ]
        if corruptions is None:
            corruptions = allowed_corruptions
        for corruption in corruptions:
            assert corruption in allowed_corruptions

        images = None
        labels = None

        self.corruptions = corruptions
        """ ([str]) Corruptions. """

        for corruption in self.corruptions:
            images_file = os.path.join(paths.raw_cifar10c_dir(), corruption + '.npy')
            labels_file = os.path.join(paths.raw_cifar10c_dir(), 'labels.npy')

            corruption_images = numpy.load(images_file)
            log('read %s' % images_file)
            corruption_labels = numpy.load(labels_file)
            log('read %s' % labels_file)

            corruption_images = corruption_images.astype(numpy.float32) / 255.

            if indices is not None:
                images = cnumpy.concatenate(images, corruption_images[indices])
                labels = cnumpy.concatenate(labels, corruption_labels[indices])
            else:
                images = cnumpy.concatenate(images, corruption_images)
                labels = cnumpy.concatenate(labels, corruption_labels)

        super(Cifar10CTestSet, self).__init__(images, labels, indices=None, resize=resize)


class SVHNTrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(SVHNTrainSet, self).__init__(paths.svhn_train_images_file(), paths.svhn_train_labels_file(), indices=indices, resize=resize)


class SVHNTestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(SVHNTestSet, self).__init__(paths.svhn_test_images_file(), paths.svhn_test_labels_file(), indices=indices, resize=resize)
