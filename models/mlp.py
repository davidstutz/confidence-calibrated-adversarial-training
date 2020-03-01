import torch
import common.torch
from .classifier import Classifier
from operator import mul
from functools import reduce


class MLP(Classifier):
    """
    MLP classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), units=[64, 64, 64], activation=torch.nn.ReLU, normalization=torch.nn.BatchNorm1d, bias=True, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param activation: activation function
        :type activation: None or torch.nn.Module
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param bias: whether to use bias
        :type bias: bool
        """

        super(MLP, self).__init__(N_class, resolution, **kwargs)

        self.units = units
        """ ([int]) Units. """

        self.activation = activation
        """ (callable) activation"""

        self.normalization = normalization
        """ (callable) Normalization. """

        self.bias = bias
        """ (bool) Bias. """

        gain = 1
        if self.activation is not None:
            gain = torch.nn.init.calculate_gain(self.activation().__class__.__name__.lower())

        # not overwriting self.units!
        units = [reduce(mul, self.resolution, 1)] + self.units
        view = common.torch.View(-1, units[0])
        self.append_layer('view0', view)

        for layer in range(1, len(units)):
            in_features = units[layer - 1]
            out_features = units[layer]

            lin = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=self.bias)
            torch.nn.init.kaiming_normal_(lin.weight, gain)
            if self.bias:
                torch.nn.init.constant_(lin.bias, 0)
            self.append_layer('lin%d' % layer, lin)

            if self.activation:
                act = self.activation()
                self.append_layer('act%d' % layer, act)

            if self.normalization is not None:
                bn = self.normalization(out_features)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.append_layer('bn%d' % layer, bn)

        logits = torch.nn.Linear(units[-1], self._N_output)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

