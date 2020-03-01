import numpy
import torch
import common.torch


class Objective:
    """
    Abstract objective.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.true_classes = None
        """ (torch.autograd.Variable) True classes. """

        self.target_classes = None
        """ (torch.autograd.Variable) Target classes. """

    def set(self, true_classes=None, target_classes=None):
        """
        Set true or target classes. Target classes might be None for untargeted attacks.

        :param true_classes: true classes
        :type true_classes: torch.autograd.Variable
        :param target_classes: target classes
        :type target_classes: torch.autograd.Variable
        """

        if target_classes is not None:
            assert true_classes is not None
        if true_classes is not None and target_classes is not None:
            assert target_classes.size()[0] == self.true_classes.size()[0]

        self.true_classes = true_classes
        self.target_classes = target_classes

    def __call__(self, logits, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()

    # some metrics for targeted and untargeted
    # not necessarily meaningful for distal

    def success(self, logits):
        """
        Get success.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :return: success
        :rtype: float
        """

        if self.true_classes is not None:
            return torch.clamp(torch.abs(torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[1] - self.true_classes), max=1)
        else:
            return torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[0]

    def true_confidence(self, logits):
        """
        True confidence.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :return: true confidence
        :rtype: float
        """

        if self.true_classes is not None:
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            return probabilities[torch.arange(logits.size(0)).long(), self.true_classes]
        else:
            return torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[0]

    def target_confidence(self, logits):
        """
        True confidence.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :return: true confidence
        :rtype: float
        """

        if self.target_classes is None:
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            if self.true_classes is not None:
                probabilities[torch.arange(logits.size(0)).long(), self.true_classes] = 0
            target_classes = torch.max(probabilities, dim=1)[1]
            return probabilities[torch.arange(logits.size(0)).long(), target_classes]
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            return probabilities[torch.arange(logits.size(0)).long(), self.target_classes]


class TargetedF0Objective(Objective):
    """
    Targeted objective based on a loss, e.g., the cross-entropy loss.
    """

    def __init__(self, loss=common.torch.classification_loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(TargetedF0Objective, self).__init__()

        self.loss = loss
        """ (callable) Loss. """

    def __call__(self, logits, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        return self.loss(logits, self.target_classes, reduction='none')


class UntargetedF0Objective(Objective):
    """
    Untargeted loss based objective, e.g., cross-entropy loss.
    """

    def __init__(self, loss=common.torch.classification_loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(UntargetedF0Objective, self).__init__()

        self.loss = loss
        """ (callable) Loss. """

    def __call__(self, logits, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        assert self.loss is not None

        return -self.loss(logits, self.true_classes, reduction='none')


class UntargetedF7Objective(Objective):
    """
    Untargeted objective based on logits, similar to Carlini+Wagner.
    """

    def __call__(self, logits, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        assert self.true_classes is not None

        if logits.size(1) > 1:
            current_logits = logits - common.torch.one_hot(self.true_classes, logits.size(1))*1e12
            return - torch.max(current_logits, dim=1)[0]
        else:
            return common.torch.binary_labels(self.true_classes.float())*logits.view(-1)


class UntargetedF7PObjective(Objective):
    """
    Untargeted confidence objective as in the paper.
    """

    def __call__(self, logits, perturbations=None):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        assert self.true_classes is not None

        if logits.size(1) > 1:
            current_probabilities = torch.nn.functional.softmax(logits, dim=1)
            current_probabilities = current_probabilities * (1 - common.torch.one_hot(self.true_classes, current_probabilities.size(1)))
            return - torch.max(current_probabilities, dim=1)[0]
        else:
            return self.true_classes.float()*(-1 + torch.nn.functional.sigmoid(logits.view(-1))) + (1 - self.true_classes.float())*(-torch.nn.functional.sigmoid(logits.view(-1)))