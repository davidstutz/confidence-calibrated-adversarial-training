import torch
from .attack import *
import common.torch


class BatchFrames(Attack):
    """
    Adversarial frames.
    """

    def __init__(self):
        """
        Constructor.

        :param mask_gen: MaskGenerator object to generate frames
        :type mask_gen: MaskGenerator
        :param epsilon: learning rate
        :type epsilon: float
        :param max_iterations: total number of iterations to learn patch
        :type max_iterations: int
        """

        super(BatchFrames, self).__init__()

        self.mask_gen = None
        self.base_lr = None
        self.max_iterations = None
        self.min = 0
        self.max = 1

    def run(self, model, images, objective, writer=common.summary.SummaryWriter(), prefix=''):
        """
        Run LaVAN attack

        :param model: model to attack, must contain normalization layer to work correctly
        :type model: torch.nn.Module
        :param images: images
        :type images: torch.autograd.Variable
        :param objective: objective
        :type objective: UntargetedObjective
        :param writer: summary writer, defaults to common.summary.SummaryWriter()
        :type writer: common.summary.SummaryWriter, optional
        :param prefix: prefix for writer, defaults to ''
        :type prefix: str, optional
        """

        super(BatchFrames, self).run(model, images, objective, writer, prefix)

        assert model.training is False
        assert self.mask_gen is not None
        assert self.base_lr is not None
        assert self.max_iterations is not None and self.max_iterations >= 0

        assert len(images.shape) == 4
        batch_size, channels, _, _ = images.shape

        is_cuda = common.torch.is_cuda(model)

        mask_coords = self.mask_gen.random_location(batch_size)
        masks = common.torch.as_variable(self.mask_gen.get_masks(mask_coords, channels).astype(numpy.float32), cuda=is_cuda)
        patches = common.torch.as_variable(numpy.random.uniform(low=0.0, high=1.0, size=images.shape).astype(numpy.float32), cuda=is_cuda, requires_grad=True)

        current_iteration = 0

        success_errors = numpy.ones((batch_size), dtype=numpy.float32) * 1e12
        success_perturbations = numpy.zeros(images.shape, dtype=numpy.float32)

        iters = []
        losses = []

        # Train the current batch
        while current_iteration < self.max_iterations:
            current_iteration += 1

            # Apply patch
            imgs_patched = images + masks*patches

            # Get predictions on patched images
            preds = model(imgs_patched)

            # Predicted classes. Currently unused.
            pred_classes = torch.argmax(preds, dim=1)

            # Calculate error per image. `objective` is aware of the true labels.
            error = objective(preds)
            success = objective.success(preds)

            # Compute loss and backprop
            loss = torch.sum(error)
            print(current_iteration, loss.item()/batch_size, torch.sum(success).item())

            writer.add_scalar('%sloss' % (prefix), loss, global_step=current_iteration)
            loss.backward()

            iters.append(current_iteration)
            losses.append(loss)

            # For every image, if the best known loss so far is found, update best loss and its
            # corresponding patch. Currently, not checking if flip occurs.
            for b in range(batch_size):
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=current_iteration)
                writer.add_scalar('%sprediction_%d' % (prefix, b), pred_classes[b], global_step=current_iteration)
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].item()
                    success_perturbations[b] = (masks[b]*patches[b]).detach().cpu().numpy()

            # Get gradient with respect to patches
            loss_grad = patches.grad

            # Patch update direction aims to reduce activation of source class
            # patches.data is used on the left so that a new node is not created in the graph
            patches.data = patches - self.base_lr*masks*torch.sign(loss_grad)

            # Clip patches, since it must always be in [0,1)
            patches.data.clamp_(self.min, self.max)

            # Set gradients for patches to zero
            patches.grad.data.zero_()

        return success_perturbations, success_errors
