import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.utils
import common.datasets
import common.imgaug
from imgaug import augmenters as iaa
import torch
import torch.utils.data
import math


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.images = numpy.random.randn(1000, 32, 33, 3)*10
        self.labels = numpy.random.randint(0, 9, size=(1000))
        self.assertEqual(self.images.shape[0], self.labels.shape[0])

    def testUnshuffled(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels)
        batch_sizes = [1, 25, 50, 13, 77, 99, 100]
        for batch_size in batch_sizes:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            batch_count = 0
            num_batches = math.ceil(self.images.shape[0]/batch_size)
            self.assertEqual(len(dataset), self.images.shape[0])
            self.assertEqual(num_batches, len(dataloader))

            for b, (images, labels) in enumerate(dataloader):
                self.assertTrue(isinstance(images, torch.Tensor))
                self.assertTrue(isinstance(images, torch.Tensor))

                perm = numpy.unique(numpy.take(range(self.images.shape[0]), range(b*batch_size, (b+1)*batch_size), mode='clip'))
                if b < num_batches - 1:
                    self.assertEqual(batch_size, images.shape[0], 'b=%d num_batches=%d' % (b, num_batches))
                    self.assertEqual(batch_size, labels.shape[0], 'b=%d num_batches=%d' % (b, num_batches))
                elif b == num_batches - 1:
                    self.assertEqual(self.images.shape[0] - batch_count*batch_size, images.shape[0])
                    self.assertEqual(self.labels.shape[0] - batch_count*batch_size, labels.shape[0])
                else:
                    self.fail()

                self.assertEqual(len(labels.shape), 1)
                self.assertEqual(len(images.shape), 4)

                self.assertEqual(perm.shape[0], images.shape[0], 'b=%d num_batches=%d' % (b, num_batches))
                self.assertEqual(perm.shape[0], labels.shape[0], 'b=%d num_batches=%d' % (b, num_batches))

                numpy.testing.assert_array_almost_equal(images, self.images[perm], decimal=5)
                numpy.testing.assert_array_equal(labels, self.labels[perm])
                batch_count += 1

            self.assertEqual(batch_count, num_batches)

    def testShuffled(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels)
        batch_sizes = [1, 25, 50, 13, 77, 99, 100]
        for batch_size in batch_sizes:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            batch_count = 0
            num_batches = math.ceil(self.images.shape[0]/batch_size)
            self.assertEqual(len(dataset), self.images.shape[0])
            self.assertEqual(num_batches, len(dataloader))

            for b, (images, labels) in enumerate(dataloader):
                self.assertTrue(isinstance(images, torch.Tensor))
                self.assertTrue(isinstance(images, torch.Tensor))

                if b < num_batches - 1:
                    self.assertEqual(batch_size, images.shape[0], 'b=%d num_batches=%d' % (b, num_batches))
                    self.assertEqual(batch_size, labels.shape[0], 'b=%d num_batches=%d' % (b, num_batches))
                elif b == num_batches - 1:
                    self.assertEqual(self.images.shape[0] - batch_count*batch_size, images.shape[0])
                    self.assertEqual(self.labels.shape[0] - batch_count*batch_size, labels.shape[0])
                else:
                    self.fail()

                self.assertEqual(len(labels.shape), 1)
                self.assertEqual(len(images.shape), 4)

                batch_count += 1

            self.assertEqual(batch_count, num_batches)

    def testZip(self):
        dataset = common.datasets.CleanDataset(self.images, self.labels)
        dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=13, shuffle=False)
        dataloader2 = torch.utils.data.DataLoader(dataset, batch_size=13, shuffle=False)

        for x in zip(enumerate(dataloader1), enumerate(dataloader2)):
            self.assertEqual(len(x), 2)
            self.assertEqual(len(x[0]), 2)
            self.assertEqual(len(x[1]), 2)

            b1 = x[0][0]
            data1 = x[0][1]
            self.assertEqual(len(data1), 2)

            b2 = x[1][0]
            data2 = x[1][1]
            self.assertEqual(len(data2), 2)

            self.assertEqual(b1, b2)

            numpy.testing.assert_almost_equal(data1[0].numpy(), data2[0].numpy())
            numpy.testing.assert_almost_equal(data1[1].numpy(), data2[1].numpy())


if __name__ == '__main__':
    unittest.main()
