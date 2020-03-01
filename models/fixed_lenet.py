import torch
import common.torch
from .classifier import Classifier
import torch.nn as nn


class FixedLeNet(Classifier):
    """
    Fixed LeNet architecture, working on MNIST architectures only.
    """

    def __init__(self, N_class, resolution=(1, 28, 28), **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        """

        assert resolution[0] == 1
        assert resolution[1] == 28
        assert resolution[2] == 28

        super(FixedLeNet, self).__init__(N_class, resolution, **kwargs)

        self.append_layer('0', nn.Conv2d(resolution[0], 32, 5, padding=2))
        self.append_layer('1', nn.ReLU())
        self.append_layer('2', nn.MaxPool2d(2, 2))
        self.append_layer('3', nn.Conv2d(32, 64, 5, padding=2))
        self.append_layer('4', nn.ReLU())
        self.append_layer('5', nn.MaxPool2d(2, 2))
        self.append_layer('6', common.torch.Flatten())
        self.append_layer('7', nn.Linear(7 * 7 * 64, 1024))
        self.append_layer('8', nn.ReLU())
        self.append_layer('9', nn.Linear(1024, self.N_class))

