import torch
from .attack import *
import common.torch


class BatchQueryLimited(Attack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchQueryLimited, self).__init__()

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

        self.population = None
        """ (int) Population. """

        self.variance = None
        """ (float) Variance. """

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

        super(BatchQueryLimited, self).run(model, images, objective, writer, prefix)

        assert self.max_iterations is not None
        assert self.c is not None
        assert self.base_lr is not None
        assert self.lr_factor is not None
        assert self.momentum is not None
        assert self.population is not None
        assert self.variance is not None
        if self.population%2 == 1:
            self.population += 1
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
        """ (numpy.ndarray) Holds per element learning rates. """

        if is_cuda:
            self.perturbations = self.perturbations.cuda()
            self.lrs = self.lrs.cuda()

        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)
        self.gradients = torch.zeros_like(self.perturbations)
        """ (torch.autograd.Variable) Gradients. """

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT

            # 0/
            # Projections if necessary.
            if self.projection is not None:
                self.projection(images, self.perturbations)

            output_logits = model(images + self.perturbations)
            error = self.c * self.norm(self.perturbations) + objective(output_logits, self.perturbations)

            # 4/
            # Logging and break condition.
            norm = self.norm(self.perturbations) # Will be a vector of individual norms.

            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].data.cpu()
                    success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy())

            def sample_standard_gaussian_antithetic(size):
                """
                Sample from Gaussian using antithetic sampling as described in [1].

                [1] Andrew Ilyas, Logan Engstrom, Anish Athalye, Jessy Lin.
                    Black-box Adversarial Attacks with Limited Queries and Information.
                    arXiv.org, abs/1804.08598, 2018.

                Args:
                    size: ([int]) Size to sample; first dimension has to be a multiple of two.
                    mean: (float) Mean of Gaussian.
                    variance: (float) Variance of Gaussian.
                Returns:
                    samples. (numpy.ndarray(size)) Gaussian samples.
                """

                sample_size = list(size)
                assert len(sample_size) >= 2
                assert sample_size[0] % 2 == 0

                sample_size[0] = sample_size[0] // 2
                samples = numpy.random.normal(size=sample_size)
                samples = numpy.concatenate((samples, -samples), axis=0)

                return samples

            samples = sample_standard_gaussian_antithetic((self.population, numpy.prod(images.size())))
            samples = samples.reshape(self.population, images.size()[0], images.size()[1], images.size()[2], images.size()[3])
            samples = common.torch.as_variable(samples.astype(numpy.float32), is_cuda)

            gradients = None
            for n in range(self.population):
                perturbation = images + self.variance * samples[n]
                perturbation_logits = model(perturbation)
                g = samples[n] * common.torch.expand_as(objective(perturbation_logits, self.perturbations), samples[n])
                if gradients is None:
                    gradients = g.data
                else:
                    gradients += g.data
            gradients /= 2. * self.variance * self.population

            # 7/
            # Get the gradients and normalize.
            gradient_magnitudes = torch.mean(torch.abs(gradients.view(batch_size,  -1)), dim=1) / float(numpy.prod(self.perturbations.size()[1:]))

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
                writer.add_scalar('%sgradient_%d' % (prefix, b), gradient_magnitudes[b], global_step=i)

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                break

            if self.normalized:
                self.norm.normalize(gradients)

            # 8/
            # Update step according to learning rate.
            if self.backtrack:
                next_perturbations = self.perturbations - torch.mul(common.torch.expand_as(self.lrs, gradients), gradients)

                if self.projection is not None:
                    self.projection(images, next_perturbations)

                next_output_logits = model.forward(images + next_perturbations)
                next_error = self.c * self.norm(next_perturbations) + objective(next_output_logits, next_perturbations)

                # Update learning rate if requested.
                for b in range(batch_size):
                    if next_error[b].item() <= error[b]:
                        self.perturbations[b].data -= self.lrs[b] * gradients[b].data
                    else:
                        self.lrs[b] = max(self.lrs[b] / self.lr_factor, 1e-20)
            else:
                self.perturbations.data -= torch.mul(common.torch.expand_as(self.lrs, gradients), gradients)

        return success_perturbations, success_errors