import torch
from .attack import *
import common.torch


class BatchReferencePGD(Attack):
    """
    Reference implementation of the PGD attack as originally used by Madry et al.; the implementation
    is based on the following TensorFlow code:

    .. code-block::

        def perturb(self, x_nat, y, sess):
            \"\"\"Given a set of examples (x_nat, y), returns a set of adversarial
               examples within epsilon of x_nat in l_infinity norm.\"\"\"
            if self.rand:
              x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            else:
              x = np.copy(x_nat)

            a = self.a
            for i in range(self.k):
              grad, loss, logits = sess.run([self.grad, self.model.y_xent, self.model.pre_softmax], feed_dict={self.model.x_input: x, self.model.y_input: y})
              #print(logits)
              #print(grad[0])

              x = x + self.a * np.sign(grad)
              x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
              x = np.clip(x, 0, 1)  # ensure valid pixel range
              #print(np.mean(loss), np.mean(np.abs(grad)), np.mean(np.max(np.abs(x - x_nat).reshape(x.shape[0], -1), axis=1)))

            return x
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchReferencePGD, self).__init__()

        self.epsilon = None
        """ (float) Epsilon. """

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.base_lr = None
        """ (float) Learning rate. """

        self.norm = None
        """ (Norm) Mainly for training. """

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

        super(BatchReferencePGD, self).run(model, images, objective, writer, prefix)

        assert self.max_iterations is not None
        assert self.epsilon is not None
        assert self.base_lr is not None
        is_cuda = common.torch.is_cuda(model)

        # Objective is fixed, we only need the true classes.
        classes = objective.true_classes
        assert classes is not None

        perturbations = torch.from_numpy(numpy.random.uniform(-self.epsilon, self.epsilon, size=images.size()).astype(numpy.float32))
        if is_cuda:
            perturbations = perturbations.cuda()

        batch_size = images.size(0)
        self.perturbations = images + perturbations
        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

        for i in range(self.max_iterations):
            if i > 0:
                self.perturbations.grad.zero_()

            output_logits = model.forward(self.perturbations)
            output_log_probabilities = torch.nn.functional.log_softmax(output_logits, dim=1)
            error = - output_log_probabilities[torch.arange(batch_size).long(), classes.long()]
            loss = torch.sum(error)
            loss.backward()

            self.perturbations.data += self.base_lr*torch.sign(self.perturbations.grad.data)

            self.perturbations.data = torch.max(images.data - self.epsilon, self.perturbations.data)
            self.perturbations.data = torch.min(images.data + self.epsilon, self.perturbations.data)
            self.perturbations.data = torch.clamp(self.perturbations.data, 0, 1)

            gradients = self.perturbations.grad.clone()
            gradient_magnitudes = torch.mean(torch.abs(gradients.view(batch_size, -1)), dim=1) / batch_size
            gradient_magnitudes = gradient_magnitudes.detach().cpu().numpy()

            norm = torch.max(torch.abs(images - self.perturbations).reshape(batch_size, -1), dim=1)[0]
            norm = norm.detach().cpu().numpy()

            for b in range(batch_size):
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)
                writer.add_scalar('%snorm_%d' % (prefix, b), norm[b], global_step=i)
                writer.add_scalar('%sgradient_%d' % (prefix, b), gradient_magnitudes[b], global_step=i)

        return (self.perturbations - images).detach().cpu().numpy(), error.detach().cpu().numpy()

