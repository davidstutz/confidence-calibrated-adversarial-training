"""
Pre-activation ResNet bottleneck block in PyTorch.
Taken from https://github.com/locuslab/robust_union/.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneck(nn.Module):
    """
    Pre-Activation ResNet bottleneck block.
    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """
        Constructor.

        :param in_planes: number of input planes
        :type in_planes: int
        :param planes: output planes
        :type planes: int
        :param stride: stride for block
        :type stride: int
        """

        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out