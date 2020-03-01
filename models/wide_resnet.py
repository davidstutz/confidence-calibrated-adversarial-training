"""
Wide ResNet.
Taken from https://github.com/meliketoy/wide-resnet.pytorch.
"""
import numpy
import torch
import common.torch
from .classifier import Classifier
from .wide_resnet_block import WideResNetBlock


class WideResNet(Classifier):
    """
    Wide Res-Net.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization=True, channels=16, dropout=0, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        """

        super(WideResNet, self).__init__(N_class, resolution, **kwargs)

        self.depth = depth
        """ (int) Depth. """

        self.width = width
        """ (int) Width. """

        self.channels = channels
        """ (int) Channels. """

        self.dropout = dropout
        """ (int) Dropout. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.in_planes = channels
        """ (int) Helper for channels. """

        self.inplace = False
        """ (bool) Inplace. """

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = width

        planes = [self.channels, self.channels*k, 2*self.channels*k, 4*self.channels*k]

        downsampled = 1
        conv = torch.nn.Conv2d(resolution[0], planes[0], kernel_size=3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(conv.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv0', conv)

        block1 = self._wide_layer(WideResNetBlock, planes[1], n, stride=1)
        self.append_layer('block1', block1)
        block2 = self._wide_layer(WideResNetBlock, planes[2], n, stride=2)
        downsampled *= 2
        self.append_layer('block2', block2)
        block3 = self._wide_layer(WideResNetBlock, planes[3], n, stride=2)
        downsampled *= 2
        self.append_layer('block3', block3)

        if self.normalization:
            bn = torch.nn.BatchNorm2d(planes[3], momentum=0.9)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.append_layer('bn3', bn)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.append_layer('relu3', relu)

        representation = planes[3]
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.append_layer('avgpool', pool)

        view = common.torch.View(-1, representation)
        self.append_layer('view', view)

        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(planes[3], self._N_output)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def _wide_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout, self.normalization))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)