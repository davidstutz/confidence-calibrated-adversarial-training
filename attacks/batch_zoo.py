import torch
from .attack import *
import common.torch


class BatchZOO(Attack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchZOO, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.c = None
        """ (float) Weight of norm. """

        self.base_lr = None
        """ (float) Base learning rate. """

        self.lr_factor = None
        """ (float) Learning rate decay. """

        self.momentum = None
        """ (float) Momentum. """

        self.backtrack = False
        """ (bool) Backtrack. """

        self.h = 0.01
        """ (float) Discretization step. """

        self.normalized = False
        """ (bool) Normalize gradients. """

        self.norm = None
        """ (Norm) Norm. """

        self.initialization = None
        """ (Initialization) Initializer. """

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

        super(BatchZOO, self).run(model, images, objective, writer, prefix)

        assert self.max_iterations is not None
        assert self.c is not None
        assert self.base_lr is not None
        assert self.lr_factor is not None
        assert self.momentum is not None
        assert common.torch.is_cuda(images) == common.torch.is_cuda(model)
        is_cuda = common.torch.is_cuda(model)

        self.perturbations = torch.from_numpy(numpy.zeros(images.size(), dtype=numpy.float32))
        if self.initialization is not None:
            self.initialization(images, self.perturbations)
        if is_cuda:
            self.perturbations = self.perturbations.cuda()

        batch_size = self.perturbations.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(self.perturbations.size(), dtype=numpy.float32)

        self.lrs = torch.from_numpy(numpy.ones(batch_size, dtype=numpy.float32) * self.base_lr)
        """ (torch.autograd.Variable) Holds per element learning rates. """

        self.gradients = torch.zeros(batch_size)
        """ (torch.autograd.Variable) Gradients. """

        if is_cuda:
            self.perturbations = self.perturbations.cuda()
            self.lrs = self.lrs.cuda()
            self.gradients = self.gradients.cuda()

        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT

            index_c = numpy.random.randint(0, images.size()[1])
            index_i = numpy.random.randint(0, images.size()[2])
            index_j = numpy.random.randint(0, images.size()[3])

            # 0/
            # Projections if necessary.
            if self.projection is not None:
                self.projection(images, self.perturbations)

            perturbations_p = self.perturbations.clone()
            perturbations_p[:, index_c, index_i, index_j] += self.h
            perturbations_m = self.perturbations.clone()
            perturbations_m[:, index_c, index_i, index_j] -= self.h

            # 1/
            # Compute logits.
            output_logits_p = model.forward(images + perturbations_p)
            output_logits_m = model.forward(images + perturbations_m)
            output_logits = model.forward(images + self.perturbations)

            # 2/
            # Compute objective.
            error_p = objective(output_logits_p, perturbations_p)
            error_m = objective(output_logits_m, perturbations_m)
            error = objective(output_logits, self.perturbations)

            # 4/
            # Logging and break condition.
            norm = self.norm(self.perturbations) # Will be a vector of individual norms.

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].data.cpu()
                    success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy())

            # 7/
            # Get the gradients and normalize.
            gradients = (error_p - error_m)/(2*self.h)

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%slr_%d' % (prefix, b), self.lrs[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)
                writer.add_scalar('%snorm_%d' % (prefix, b), norm[b], global_step=i)
                writer.add_scalar('%sgradient_%d' % (prefix, b), gradients[b], global_step=i)

            if self.normalized:
                self.norm.normalize(gradients)

            self.gradients.data = self.momentum*self.gradients.data + (1 - self.momentum)*gradients.data

            # 8/
            # Update step according to learning rate.
            if self.backtrack:
                next_perturbations = self.perturbations.clone()
                next_perturbations[:, index_c, index_i, index_j].data -= torch.mul(self.lrs, gradients)

                if self.projection is not None:
                    self.projection(images, next_perturbations)

                next_output_logits = model.forward(images + next_perturbations)
                next_error = objective(next_output_logits, next_perturbations)

                # Update learning rate if requested.
                for b in range(batch_size):
                    if next_error[b].item() <= error[b]:
                        self.perturbations[b, index_c, index_i, index_j].data -= self.lrs[b]*gradients[b].data
                    else:
                        self.lrs[b] = max(self.lrs[b] / self.lr_factor, 1e-8)
            else:
                self.perturbations[:, index_c, index_i, index_j].data -= torch.mul(self.lrs, gradients)

        return success_perturbations, success_errors