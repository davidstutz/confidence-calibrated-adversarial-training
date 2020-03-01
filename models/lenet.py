import torch
import common.torch
from .classifier import Classifier


class LeNet(Classifier):
    """
    General LeNet classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), channels=64, activation=torch.nn.ReLU, normalization=True, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param channels: channels to start with
        :type channels: int
        :param units: units per layer
        :type units: [int]
        :param activation: activation function
        :type activation: None or torch.nn.Module
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param bias: whether to use bias
        :type bias: bool
        """

        super(LeNet, self).__init__(N_class, resolution, **kwargs)

        # the constructor parameters must be available as attributes for state to work
        self.channels = channels
        """ (int) Channels. """

        self.activation = activation
        """ (callable) activation"""

        self.normalization = normalization
        """ (bool) Normalization. """

        layer = 0
        layers = []
        resolutions = []

        gain = 1
        if self.activation is not None:
            gain = torch.nn.init.calculate_gain(self.activation().__class__.__name__.lower())

        while True:
            input_channels = self.resolution[0] if layer == 0 else layers[layer - 1]
            output_channels = self.channels if layer == 0 else layers[layer - 1] * 2

            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding=2)
            #torch.nn.init.normal_(conv.weight, mean=0, std=0.1)
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.constant_(conv.bias, 0.1)
            self.append_layer('conv%d' % layer, conv)

            if self.normalization:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.append_layer('bn%d' % layer, bn)

            if self.activation:
                relu = self.activation()
                self.append_layer('act%d' % layer, relu)

            pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.append_layer('pool%d' % layer, pool)

            layers.append(output_channels)
            resolutions.append([
                self.resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                self.resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][0] % 2 == 1 or resolutions[-1][1] // 2 < 3 or resolutions[-1][1] % 2 == 1:
                break

            layer += 1

        representation = int(resolutions[-1][0] * resolutions[-1][1] * layers[-1])
        view = common.torch.View(-1, representation)
        self.append_layer('view', view)

        fc = torch.nn.Linear(representation, 1024)
        self.append_layer('fc%d' % layer, fc)

        if self.activation:
            relu = self.activation()
            self.append_layer('act%d' % layer, relu)

        logits = torch.nn.Linear(1024, self._N_output)
        #torch.nn.init.normal_(conv.weight, mean=0, std=0.1)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0.1)
        self.append_layer('logits', logits)

