"""
Pre-activation ResNet in PyTorch.
Taken from https://github.com/locuslab/robust_union/.
"""

import common.torch
from .classifier import Classifier
from .preact_resnet_block import *
from .preact_resnet_bottleneck import *


class PreActResNet(Classifier):
    """
    More or less fixed pre-activation ResNet.
    """

    def __init__(self, N_class, resolution, blocks, bottleneck=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes
        :type N_class: int
        :param resolution: resolution
        :type resolution: [int]
        :param blocks: number of layers per blocks, for exactly 4 blocks
        :type blocks: [int] of length 4
        :param bottleneck: whether to use bottleneck blocks
        :type bottleneck: False
        """

        super(PreActResNet, self).__init__(N_class, resolution, **kwargs)

        assert len(blocks) == 4
        self.in_planes = 64

        self.blocks = blocks
        """ ([int]) Blocks. """

        self.bottleneck = bottleneck
        """ (bool) Inplace. """

        if self.bottleneck:
            block = PreActBottleneck
        else:
            block = PreActBlock

        self.append_layer('conv1', nn.Conv2d(resolution[0], 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.append_layer('layer1', self._make_layer(block, 64, blocks[0], stride=1))
        self.append_layer('layer2', self._make_layer(block, 128, blocks[1], stride=2))
        self.append_layer('layer3', self._make_layer(block, 256, blocks[2], stride=2))
        self.append_layer('layer4', self._make_layer(block, 512, blocks[3], stride=2))

        downsampled = 2*2*2
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.append_layer('avgpool', pool)

        view = common.torch.View(-1, self.in_planes)
        self.append_layer('view', view)

        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)

        self.append_layer('linear', nn.Linear(512*block.expansion, N_class))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)