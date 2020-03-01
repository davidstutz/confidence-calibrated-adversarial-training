from imgaug import augmenters as iaa
from imgaug import dtypes as iadt
import numpy


class Transpose(iaa.Augmenter):
    """
    Transpose augmenter, assumes the transposition to be given with respect to a four-tuple input, i.e. including batch size.
    """

    def __init__(self, transpose, name=None, deterministic=False, random_state=None):
        super(Transpose, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.transpose = transpose
        """ ((int)) Transpose indices. """

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["float32"], disallowed=[
            "bool", "uint8", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        if isinstance(images, numpy.ndarray):
            assert len(images.shape) == len(self.transpose)
        elif isinstance(images, list):
            assert len(images[0].shape) + 1 == len(self.transpose)
        converted_images = numpy.transpose(images, self.transpose)
        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.transpose]


class Clip(iaa.Augmenter):
    """
    Clip augmenter.
    """

    def __init__(self, min=0, max=1, name=None, deterministic=False, random_state=None):
        super(Clip, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.min = min
        """ (float) Minimum."""

        self.max = max
        """ (float) Maximum. """

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["float32"], disallowed=[
            "bool", "uint8", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        converted_images = numpy.clip(images, self.min, self.max)
        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class UInt8FromFloat32(iaa.Augmenter):
    """
    Convert uint8 to float32; used for some augmenters that only work on uint8 or float32.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(UInt8FromFloat32, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["float32"], disallowed=[
            "bool", "uint8", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        converted_images = images*255
        converted_images = numpy.rint(converted_images)
        converted_images = numpy.clip(converted_images, 0, 255)
        converted_images = converted_images.astype(numpy.uint8)

        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class Float32FromUInt8(iaa.Augmenter):
    """
    Convert float32 to uint8; used for some augmenters that only work on uint8 or float32.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(Float32FromUInt8, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["uint8"], disallowed=[
            "bool", "float32", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        converted_images = images.astype(numpy.float32)/255.
        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []