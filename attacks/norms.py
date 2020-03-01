import torch
import math


class Norm:
    """
    Abstract norm.
    """

    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        raise NotImplementedError()

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        raise NotImplementedError()


class L2Norm(Norm):
    """
    L_2 norm.
    """

    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        return torch.norm(perturbations.view(perturbations.size()[0], -1), p=2, dim=1)

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1, 1, 1))

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1, 1, 1))


class LInfNorm(Norm):
    """
    L_inf norm.
    """

    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        return torch.max(torch.abs(perturbations.view(perturbations.size()[0], -1)), dim=1)[0]

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.sign(gradients.data)

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.max(torch.abs(gradients.data.view(gradients.size()[0], -1)), dim=1)[0].view(-1, 1, 1, 1))


class L1Norm(Norm):
    """
    L_1 norm.
    """

    def __init__(self, fraction=0.025):
        """
        Constructor.

        :param fraction: fraction of elements to keep in normalization
        :type fraction: float
        """

        assert fraction > 0
        assert fraction <= 1

        self.fraction = fraction

    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        #return torch.norm(perturbations.view(perturbations.size()[0], -1), p=1, dim=1)
        return torch.sum(torch.abs(perturbations.view(perturbations.size()[0], -1)), dim=1)

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        sorted, _ = torch.sort(gradients.data.view(gradients.size()[0], -1), dim=1)
        k = int(math.ceil(sorted.size(1)*self.fraction))
        assert k > 0
        thresholds = sorted[:, -k]
        mask = (gradients.data >= thresholds.view(-1, 1, 1, 1)).float()

        gradients.data = gradients.data*mask
        gradients.data = torch.div(gradients.data, torch.norm(gradients.view(gradients.size()[0], -1), p=1, dim=1).view(-1, 1, 1, 1))

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), p=1, dim=1).view(-1, 1, 1, 1))


class L0Norm(Norm):
    """
    L_0 norm.
    """

    def __init__(self, fraction=0.01):
        """
        Constructor.

        :param fraction: fraction of elements to keep in normalization
        :type fraction: float
        """

        assert fraction > 0
        assert fraction <= 1

        self.fraction = fraction

    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        return torch.norm(perturbations.view(perturbations.size()[0], -1), p=0, dim=1)

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        # ! Note p=1
        gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), p=1, dim=1).view(-1, 1, 1, 1))

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), p=0, dim=1).view(-1, 1, 1, 1))