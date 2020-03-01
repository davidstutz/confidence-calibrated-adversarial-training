import torch
from .attack import *
import common.torch


class BatchCube(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchCube, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.probability = None
        """ (float) Probability. """

        self.projection = None
        """ (Projection) Projection. """

        self.epsilon = None
        """ (float) Epsilon. """

        self.projection = None
        """ (attacks.Projection) Projection. """

    def run(self, model, images, objective, writer=common.summary.SummaryWriter(), prefix=''):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param images: images
        :type images: torch.autograd.Variable
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        :param writer: summary writer
        :type writer: common.summary.SummaryWriter
        :param prefix: prefix for writer
        :type prefix: str
        """

        assert self.max_iterations is not None
        assert self.probability is not None
        assert self.epsilon is not None

        super(BatchCube, self).run(model, images, objective, writer, prefix)

        is_cuda = common.torch.is_cuda(model)
        self.perturbations = common.torch.as_variable(numpy.zeros(images.size(), dtype=numpy.float32), is_cuda)

        batch_size = images.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(images.size(), dtype=numpy.float32)

        for i in range(self.max_iterations):
            perturbations = self.perturbations + common.torch.as_variable(numpy.random.choice([-2*self.epsilon, 0, 2*self.epsilon], size=self.perturbations.size(), p=[self.probability/2, 1-self.probability, self.probability/2]).astype(numpy.float32), is_cuda)
            if self.projection:
                self.projection(images, perturbations)

            output_logits = model(images + perturbations)
            error = objective(output_logits, perturbations)

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b]
                    success_perturbations[b] = perturbations[b].detach().cpu().numpy()
                    self.perturbations[b] = perturbations[b]

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)

        return success_perturbations, success_errors