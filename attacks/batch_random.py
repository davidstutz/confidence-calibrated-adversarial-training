import torch
from .attack import *
import common.torch


class BatchRandom(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchRandom, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.initialization = None
        """ (Initialization) Initialization. """

        self.projection = None
        """ (Projection) Projection. """

        self.norm = None
        """ (Norm) Norm. """

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

        super(BatchRandom, self).run(model, images, objective, writer, prefix)

        is_cuda = common.torch.is_cuda(model)
        self.perturbations = torch.from_numpy(numpy.zeros(images.size(), dtype=numpy.float32))

        batch_size = self.perturbations.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(self.perturbations.size(), dtype=numpy.float32)

        for i in range(self.max_iterations):

            self.initialization(images, self.perturbations)
            if is_cuda:
                self.perturbations = self.perturbations.cuda()

            if self.projection is not None:
                self.projection(images, self.perturbations)

            output_logits = model.forward(images + self.perturbations)
            norm = self.norm(self.perturbations)
            error = objective(output_logits, self.perturbations)

            for b in range(batch_size):
                 if error[b].item() < success_errors[b]:
                     success_errors[b] = error[b].item()
                     success_perturbations[b] = self.perturbations[b].detach().cpu().numpy()

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)
                writer.add_scalar('%snorm_%d' % (prefix, b), norm[b], global_step=i)

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                break

        # adversarial_example = image + success_perturbation
        # success_perturbation = adversarial_example - image
        return success_perturbations, success_errors