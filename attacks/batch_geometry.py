import torch
from .attack import *
import common.torch


class BatchGeometry(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchGeometry, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.database = None
        """ (numpy.ndarray) Image database. """

        self.projection = None
        """ (Projection) Projection. """

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

        assert self.projection is not None
        assert self.database is not None
        assert isinstance(self.database, numpy.ndarray)
        database_size = self.database.shape
        images_size = list(images.size())
        assert len(database_size) == len(images_size)
        for d in range(1, len(database_size)):
            assert database_size[d] == images_size[d]

        super(BatchGeometry, self).run(model, images, objective, writer, prefix)

        is_cuda = common.torch.is_cuda(model)

        batch_size = images.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(images.size(), dtype=numpy.float32)

        for i in range(self.database.shape[0]):
            perturbations = common.torch.as_variable(numpy.repeat(numpy.expand_dims(self.database[i], axis=0), batch_size, axis=0), is_cuda) - images
            self.projection(images, perturbations)

            output_logits = model.forward(perturbations)
            error = objective(output_logits, perturbations)

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].cpu().item()
                    success_perturbations[b] = perturbations[b].detach().cpu().numpy()

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)

        return success_perturbations, success_errors