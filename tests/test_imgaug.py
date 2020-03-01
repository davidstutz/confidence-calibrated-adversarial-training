import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.utils
import common.datasets
import common.imgaug
from imgaug import augmenters as iaa


class TestImgaug(unittest.TestCase):
    def testAugmentationTransposeSingleImage(self):
        augmentation = iaa.Sequential([
            common.imgaug.Transpose((0, 3, 1, 2)),
        ])

        image = numpy.random.randn(32, 33, 1).astype(numpy.float32)
        augmented_image = augmentation.augment_image(image)
        assert len(augmented_image.shape) == 3
        assert augmented_image.shape[0] == image.shape[2]
        assert augmented_image.shape[1] == image.shape[0]
        assert augmented_image.shape[2] == image.shape[1]

        image = numpy.random.randn(32, 33, 3).astype(numpy.float32)
        augmented_image = augmentation.augment_image(image)
        assert len(augmented_image.shape) == 3
        assert augmented_image.shape[0] == image.shape[2]
        assert augmented_image.shape[1] == image.shape[0]
        assert augmented_image.shape[2] == image.shape[1]

    def testAugmentationTransposeMultipleImages(self):
        augmentation = iaa.Sequential([
            common.imgaug.Transpose((0, 3, 1, 2)),
        ])

        images = numpy.random.randn(100, 32, 33, 1).astype(numpy.float32)
        augmented_images = augmentation.augment_images(images)
        assert len(augmented_images.shape) == 4
        assert augmented_images.shape[0] == images.shape[0]
        assert augmented_images.shape[1] == images.shape[3]
        assert augmented_images.shape[2] == images.shape[1]
        assert augmented_images.shape[3] == images.shape[2]

        images = numpy.random.randn(100, 32, 33, 3).astype(numpy.float32)
        augmented_images = augmentation.augment_images(images)
        assert len(augmented_images.shape) == 4
        assert augmented_images.shape[0] == images.shape[0]
        assert augmented_images.shape[1] == images.shape[3]
        assert augmented_images.shape[2] == images.shape[1]
        assert augmented_images.shape[3] == images.shape[2]

    def testAugmentationClipSingleImage(self):
        min = 0
        max = 1
        augmentation = iaa.Sequential([
            common.imgaug.Clip(min, max),
        ])

        image = numpy.random.randn(32, 33, 1).astype(numpy.float32)*5
        augmented_image = augmentation.augment_image(image)
        assert numpy.min(augmented_image) >= min
        assert numpy.max(augmented_image) <= max

        image = numpy.random.randn(32, 33, 3).astype(numpy.float32)*5
        augmented_image = augmentation.augment_image(image)
        assert numpy.min(augmented_image) >= min
        assert numpy.max(augmented_image) <= max

    def testAugmentationClipMultipleImages(self):
        min = 0
        max = 1
        augmentation = iaa.Sequential([
            common.imgaug.Clip(min, max),
        ])

        images = numpy.random.randn(100, 32, 33, 1).astype(numpy.float32)*5
        augmented_images = augmentation.augment_images(images)
        assert numpy.min(augmented_images) >= min
        assert numpy.max(augmented_images) <= max

        images = numpy.random.randn(100, 32, 33, 3).astype(numpy.float32)*5
        augmented_images = augmentation.augment_images(images)
        assert numpy.min(augmented_images) >= min
        assert numpy.max(augmented_images) <= max


if __name__ == '__main__':
    unittest.main()
