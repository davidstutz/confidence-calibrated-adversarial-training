import random
import numpy as np


# for Python < 3.6
def choices(sequence, k):
    return [random.choice(sequence) for _ in range(k)]


class MaskGenerator:
    """
    Generates a frame location.
    """

    def __init__(self, img_dims):
        """
        Constructor. Exactly one of include_list and exclude_list must be specified.

        :param img_dims: image dimensions
        :type img_dims: tuple
        """
        assert len(img_dims) == 2

        self.img_dims = img_dims

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """

        raise NotImplementedError

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """

        raise NotImplementedError


class FrameGenerator(MaskGenerator):
    """
    Generates a frame location.
    """

    def __init__(self, img_dims, frame_size):
        """
        Constructor. Exactly one of include_list and exclude_list must be specified.

        :param img_dims: image dimensions
        :type img_dims: tuple
        :param frame_size: frame size in pixels
        :type frame_size: int
        """

        super(FrameGenerator, self).__init__(img_dims)

        self.frame_size = frame_size

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """

        return [[0, 0] for _ in range(n)]

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert n_channels >= 1

        batch_size = len(mask_coords)
        masks = np.zeros((batch_size, n_channels, self.img_dims[0], self.img_dims[1]))
        masks[:, :, :, :self.frame_size] = 1
        masks[:, :, :self.frame_size, :] = 1
        masks[:, :, :, -self.frame_size:] = 1
        masks[:, :, -self.frame_size:, :] = 1

        return masks
