import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.imgaug
import common.eval
import attacks
import torch
import torch.utils.data
from imgaug import augmenters as iaa


class TestSchedulers(unittest.TestCase):
    def setUp(self):
        train_images = numpy.zeros((10000, 12, 12, 1))
        train_labels = numpy.zeros((10000))
        test_images = numpy.ones((1000, 12, 12, 1))
        test_labels = numpy.ones((1000))

        self.trainset = torch.utils.data.DataLoader(common.datasets.CleanDataset(train_images, train_labels), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.CleanDataset(test_images, test_labels), batch_size=100, num_workers=4)

    def testExponentialScheduler(self):
        model = models.LeNet(10, [1, 12, 12], channels=2)

        cuda = True
        if cuda:
            model = model.cuda()

        lr = 0.1
        momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        gamma = 0.9
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset), gamma=gamma)
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=cuda)
        trainer.summary_gradients = True

        epochs = 10
        for e in range(epochs):
            trainer.step(e)
            self.assertAlmostEqual(scheduler.get_lr()[0], lr*gamma**(e + 1))

    def testCyclicScheduler(self):
        model = models.LeNet(10, [1, 12, 12], channels=2)

        cuda = True
        if cuda:
            model = model.cuda()

        lr = 0.1
        momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # should move in triangle between 0.01*lr and lr every two epochs
        scheduler = common.train.get_cyclic_scheduler(optimizer, batches_per_epoch=len(self.trainset), base_lr=0.01*lr, max_lr=lr, step_size_factor=2)
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epoch_lrs = [
            (0.01*lr + lr)/2, # lr AFTER first epoch (i.e., for e = 0)
            lr,
            (0.01 * lr + lr) / 2,
            0.01*lr,
            (0.01 * lr + lr) / 2,
            lr,
            (0.01 * lr + lr) / 2,
            0.01 * lr,
            (0.01 * lr + lr) / 2,
            lr,
            (0.01 * lr + lr) / 2,
            0.01 * lr,
        ]

        trainer = common.train.NormalTraining(model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=cuda)
        trainer.summary_gradients = True

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

            self.assertAlmostEqual(scheduler.get_lr()[0], epoch_lrs[e])


if __name__ == '__main__':
    unittest.main()
