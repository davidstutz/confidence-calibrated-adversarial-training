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


class TestTrainModelsMNIST(unittest.TestCase):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=100, shuffle=False, num_workers=4)
        self.cuda = True

    def testTrainModel(self, model):
        if self.cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 20
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        print(eval.test_error())
        return eval

    def testResNet(self):
        model = models.ResNet(10, [1, 28, 28], channels=12, blocks=[3, 3, 3])
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.03, eval.test_error())

    def testWideResNet(self):
        model = models.ResNet(10, [1, 28, 28], depth=28, width=10, channels=8)
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.03, eval.test_error())


class TestTrainModelsSVHN(TestTrainModelsMNIST):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.SVHNTrainSet(), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.SVHNTestSet(), batch_size=100, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.ResNet(10, [3, 32, 32], channels=12, blocks=[3, 3, 3])
        if self.cuda:
            self.model = self.model.cuda()

    def testResNet(self):
        model = models.ResNet(10, [3, 32, 32], channels=12, blocks=[3, 3, 3])
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testWideResNet(self):
        model = models.ResNet(10, [3, 32, 32], depth=28, width=10, channels=16)
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.05, eval.test_error())


class TestTrainModelsCifar10(TestTrainModelsMNIST):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.Cifar10TrainSet(), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.Cifar10TestSet(), batch_size=100, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.ResNet(10, [3, 32, 32], channels=12, blocks=[3, 3, 3])
        if self.cuda:
            self.model = self.model.cuda()

    def testResNet(self):
        model = models.ResNet(10, [3, 32, 32], channels=12, blocks=[3, 3, 3])
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.1, eval.test_error())

    def testWideResNet(self):
        model = models.ResNet(10, [3, 32, 32], depth=28, width=10, channels=16)
        eval = self.testTrainModel(model)
        self.assertGreaterEqual(0.08, eval.test_error())


if __name__ == '__main__':
    unittest.main()
