import unittest
import torch
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models


class TestModels(unittest.TestCase):
    def testLeNet(self):
        resolutions = [
            [1, 2, 2],
            [1, 3, 3],
            [1, 4, 4],
            [1, 5, 5],
            [1, 4, 5],
            [1, 5, 4],
            [1, 27, 32],
            [1, 32, 27],
            [1, 32, 32],
            [3, 32, 32],
        ]
        channels = [1, 2]
        activations = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        normalizations = [
            True,
            False
        ]

        classes = 10
        batch_size = 100
        for resolution in resolutions:
            for channel in channels:
                for activation in activations:
                    for normalization in normalizations:
                        model = models.LeNet(classes, resolution, clamp=True, channels=channel, activation=activation, normalization=normalization)
                        output = model(torch.autograd.Variable(torch.zeros([batch_size] + resolution)))
                        self.assertEqual(output.size()[0], batch_size)
                        self.assertEqual(output.size()[1], classes)

    def testMLP(self):
        resolutions = [
            [1, 2, 2],
            [1, 3, 3],
            [1, 4, 4],
            [1, 5, 5],
            [1, 4, 5],
            [1, 5, 4],
            [1, 27, 32],
            [1, 32, 27],
            [1, 32, 32],
            [3, 32, 32],
        ]
        units = [
            [10],
            [10, 10],
            [10, 10, 10],
            [10, 10, 10, 10],
            [1000],
        ]
        activations = [
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
        ]
        normalizations = [
            None,
            torch.nn.BatchNorm1d
        ]

        classes = 10
        batch_size = 100
        for resolution in resolutions:
            for unit in units:
                for activation in activations:
                    for normalization in normalizations:
                        model = models.MLP(classes, resolution, clamp=True, units=unit, activation=activation, normalization=normalization)
                        output = model(torch.autograd.Variable(torch.zeros([batch_size] + resolution)))
                        self.assertEqual(output.size()[0], batch_size)
                        self.assertEqual(output.size()[1], classes)

    def testResNet(self):
        resolutions = [
            [3, 32, 32],
        ]

        blocks = [
            [3],
            [3, 3],
            [3, 3, 3],
            [3, 3, 3, 3],
        ]
        normalizations = [
            True,
            False
        ]

        classes = 10
        batch_size = 100
        for resolution in resolutions:
            for block in blocks:
                for normalization in normalizations:
                    model = models.ResNet(classes, resolution, clamp=True, blocks=block, normalization=normalization)
                    output = model(torch.autograd.Variable(torch.zeros([batch_size] + resolution)))
                    self.assertEqual(output.size()[0], batch_size)
                    self.assertEqual(output.size()[1], classes)

    def testWideResNet(self):
        resolutions = [
            [3, 32, 32],
        ]
        depths = [
            28,
            40,
        ]
        widths = [
            10,
            20,
        ]
        normalizations = [
            True,
            False
        ]

        classes = 10
        batch_size = 100
        for resolution in resolutions:
            for depth in depths:
                for width in widths:
                    for normalization in normalizations:
                        model = models.WideResNet(classes, resolution, clamp=True, depth=depth, width=width, normalization=normalization)
                        output = model(torch.autograd.Variable(torch.zeros([batch_size] + resolution)))
                        self.assertEqual(output.size()[0], batch_size)
                        self.assertEqual(output.size()[1], classes)


if __name__ == '__main__':
    unittest.main()
