import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.utils
import common.datasets
import common.imgaug
from imgaug import augmenters as iaa


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.images = numpy.random.uniform(0, 1, size=(1000, 32, 33, 3))
        self.labels = numpy.random.randint(0, 9, size=(1000))
        self.assertEqual(self.images.shape[0], self.labels.shape[0])

        self.images_file = 'images.h5'
        self.labels_file = 'labels.h5'

        common.utils.write_hdf5(self.images_file, self.images)
        common.utils.write_hdf5(self.labels_file, self.labels)

    def testStrings(self):
        dataset = common.datasets.CleanDataset(self.images_file, self.labels_file)
        self.assertEqual(self.images.shape[0], len(dataset))
        for i in range(len(dataset)):
            image, label = dataset[i]
            numpy.testing.assert_array_almost_equal(image, self.images[i], decimal=5)
            self.assertEqual(label, self.labels[i])

    def testNumpy(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels)
        self.assertEqual(self.images.shape[0], len(dataset))
        for i in range(len(dataset)):
            image, label = dataset[i]
            numpy.testing.assert_array_almost_equal(image, self.images[i], decimal=5)
            self.assertEqual(label, self.labels[i])

    def testTargets(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels)
        dataset.add_targets(numpy.zeros(len(dataset)))
        for i in range(len(dataset)):
            image, label, target = dataset[i]
            self.assertEqual(target, 0)

    def testIndices(self):
        indices = numpy.random.permutation(self.images.shape[0])
        indices = indices[:100]
        dataset = common.datasets.CleanDataset(self.images, self.labels, indices=indices)
        self.assertEqual(len(dataset), 100)

        for i in range(len(dataset)):
            image, label = dataset[i]
            numpy.testing.assert_array_almost_equal(image, self.images[indices[i]], decimal=5)
            self.assertEqual(label, self.labels[indices[i]])

    def testResizeSmaller(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels, indices=None, resize=[3, 20, 21])
        self.assertEqual(dataset.images.shape[0], self.images.shape[0])
        self.assertEqual(dataset.images.shape[1], 20)
        self.assertEqual(dataset.images.shape[2], 21)
        self.assertEqual(dataset.images.shape[3], 3)
        self.assertAlmostEqual(numpy.mean(dataset.images), numpy.mean(self.images), 4)

    def testResizeLarger(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels, indices=None, resize=[3, 40, 43])
        self.assertEqual(dataset.images.shape[0], self.images.shape[0])
        self.assertEqual(dataset.images.shape[1], 40)
        self.assertEqual(dataset.images.shape[2], 43)
        self.assertEqual(dataset.images.shape[3], 3)
        self.assertAlmostEqual(numpy.mean(dataset.images), numpy.mean(self.images), 4)

    def testColor(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels, indices=None, resize=[1, 32, 33])
        self.assertEqual(dataset.images.shape[0], self.images.shape[0])
        self.assertEqual(dataset.images.shape[1], 32)
        self.assertEqual(dataset.images.shape[2], 33)
        self.assertEqual(dataset.images.shape[3], 1)
        grayscale = 0.2125 * self.images[:, :, :, 0] + 0.7154 * self.images[:, :, :, 1] + 0.0721 * self.images[:, :, :, 2]
        grayscale = numpy.expand_dims(grayscale, axis=4)
        numpy.testing.assert_almost_equal(dataset.images, grayscale)

        dataset = common.datasets.CleanDataset(grayscale, self.labels, indices=None, resize=[3, 32, 33])
        self.assertEqual(dataset.images.shape[0], self.images.shape[0])
        self.assertEqual(dataset.images.shape[1], 32)
        self.assertEqual(dataset.images.shape[2], 33)
        self.assertEqual(dataset.images.shape[3], 3)
        color = numpy.concatenate((
            grayscale,
            grayscale,
            grayscale
        ), axis=3)
        numpy.testing.assert_almost_equal(dataset.images, color)

    def testAdversarial(self):
        perturbations = numpy.expand_dims(numpy.ones(self.images.shape), axis=0)
        dataset = common.datasets.AdversarialDataset(self.images, perturbations, self.labels)
        self.assertTrue(len(dataset), self.images.shape[0])

        for i in range(len(dataset)):
            adversarial_input, label = dataset[i]
            numpy.testing.assert_array_almost_equal(adversarial_input, self.images[i] + 1, decimal=5)

    def testAdversarialWithMultipleAttempts(self):
        perturbations = numpy.concatenate((
            numpy.expand_dims(numpy.ones(self.images.shape) * 0, axis=0),
            numpy.expand_dims(numpy.ones(self.images.shape) * 1, axis=0),
            numpy.expand_dims(numpy.ones(self.images.shape) * 2, axis=0),
        ))
        dataset = common.datasets.AdversarialDataset(self.images, perturbations, self.labels)
        self.assertTrue(len(dataset), 3*self.images.shape[0])

        for i in range(len(dataset)):
            adversarial_input, label = dataset[i]
            offset = i//self.images.shape[0]
            numpy.testing.assert_array_almost_equal(adversarial_input, self.images[i%self.images.shape[0]] + offset, decimal=5)

    def testAdversarialIndices(self):
        indices = numpy.random.permutation(self.images.shape[0])
        indices = indices[:100]

        perturbations = numpy.expand_dims(numpy.ones(self.images.shape), axis=0)
        dataset = common.datasets.AdversarialDataset(self.images, perturbations, self.labels, indices=indices)
        self.assertTrue(len(dataset), 100)

        for i in range(len(dataset)):
            adversarial_input, label = dataset[i]
            numpy.testing.assert_array_almost_equal(adversarial_input, self.images[indices[i]] + 1, decimal=5)

    def testAdversarialIndicesWithMultipleAttempts(self):
        indices = numpy.random.permutation(self.images.shape[0])
        indices = indices[:100]

        perturbations = numpy.concatenate((
            numpy.expand_dims(numpy.ones(self.images.shape) * 0, axis=0),
            numpy.expand_dims(numpy.ones(self.images.shape) * 1, axis=0),
            numpy.expand_dims(numpy.ones(self.images.shape) * 2, axis=0),
        ))
        dataset = common.datasets.AdversarialDataset(self.images, perturbations, self.labels, indices=indices)
        self.assertTrue(len(dataset), 3*100)

        for i in range(len(dataset)):
            adversarial_input, label = dataset[i]
            offset = i // 100
            numpy.testing.assert_array_almost_equal(adversarial_input, self.images[indices[i%100]] + offset, decimal=5)

    def tearDown(self):
        if os.path.exists(self.images_file):
            os.unlink(self.images_file)
        if os.path.exists(self.labels_file):
            os.unlink(self.labels_file)


if __name__ == '__main__':
    unittest.main()
