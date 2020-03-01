import torch
from .attack import *
import common.torch


class BatchSimple(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchSimple, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

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
        assert self.epsilon is not None

        super(BatchSimple, self).run(model, images, objective, writer, prefix)

        is_cuda = common.torch.is_cuda(model)
        self.perturbations = common.torch.as_variable(numpy.zeros(images.size(), dtype=numpy.float32), is_cuda)
        taken = numpy.zeros(images.size()[1:])

        batch_size = images.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(images.size(), dtype=numpy.float32)

        for i in range(self.max_iterations):

            if numpy.all(taken > 0):
                taken = numpy.zeros(images.size()[1:])

            index_c = numpy.random.randint(0, images.size()[1])
            index_i = numpy.random.randint(0, images.size()[2])
            index_j = numpy.random.randint(0, images.size()[3])
            while taken[index_c, index_i, index_j] > 0:
                index_c = numpy.random.randint(0, images.size()[1])
                index_i = numpy.random.randint(0, images.size()[2])
                index_j = numpy.random.randint(0, images.size()[3])

            taken[index_c, index_i, index_j] = 1

            self.perturbations[:, index_c, index_i, index_j] = self.epsilon
            if self.projection is not None:
                self.projection(images, self.perturbations)
            output_logits = model.forward(images + self.perturbations)
            error = objective(output_logits, self.perturbations)

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b]
                    success_perturbations[b, index_c, index_i, index_j] = self.perturbations[b, index_c, index_i, index_j]

            self.perturbations[:, index_c, index_i, index_j] = -self.epsilon
            if self.projection is not None:
                self.projection(images, self.perturbations)

            output_logits = model.forward(images + self.perturbations)
            error = objective(output_logits, self.perturbations)

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b]
                    success_perturbations[b, index_c, index_i, index_j] = self.perturbations[b, index_c, index_i, index_j]

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)

        return success_perturbations, success_errors