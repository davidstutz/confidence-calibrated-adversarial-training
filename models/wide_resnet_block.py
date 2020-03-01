"""
Wide ResNet block.
Taken from https://github.com/meliketoy/wide-resnet.pytorch.
"""
import torch
import numpy


class WideResNetBlock(torch.nn.Module):
    """
    Wide ResNet block.
    """

    def __init__(self, inplanes, planes, stride=1, dropout=0, normalization=True):
        """
        Constructor.

        :param inplanes: input channels
        :type inplanes: int
        :param planes: output channels
        :type planes: int
        :param stride: stride
        :type stride: int
        :param dropout: dropout rate
        :type dropout: float
        :param normalization: whether to use normalization
        :type normalization: bool
        """

        assert inplanes > 0
        assert planes > 0
        assert planes >= inplanes
        assert stride >= 1
        assert dropout >= 0 and dropout < 1

        super(WideResNetBlock, self).__init__()

        self.normalization = normalization
        """ (bool) Normalization or not. """

        self.dropout = dropout
        """ (float) Dropout factor. """

        self.inplace = False
        """ (bool) Inplace. """

        if self.normalization:
            self.bn1 = torch.nn.BatchNorm2d(inplanes)
            torch.nn.init.constant_(self.bn1.weight, 1)
            torch.nn.init.constant_(self.bn1.bias, 0)

        self.relu1 = torch.nn.ReLU(inplace=self.inplace)
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.conv1.bias, 0)

        if self.dropout > 1e-3:
            self.drop1 = torch.nn.Dropout(p=dropout)
        if self.normalization:
            self.bn2 = torch.nn.BatchNorm2d(planes)
            torch.nn.init.constant_(self.bn2.weight, 1)
            torch.nn.init.constant_(self.bn2.bias, 0)

        self.relu2 = torch.nn.ReLU(inplace=self.inplace)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=numpy.sqrt(2))
        # torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.conv2.bias, 0)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        if self.normalization:
            out = self.bn1(x)
            out = self.relu1(out)
        else:
            out = self.relu1(x)
        out = self.conv1(out)
        if self.dropout > 1e-3:
            out = self.drop1(out)
        if self.normalization:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += self.shortcut(x)

        return out