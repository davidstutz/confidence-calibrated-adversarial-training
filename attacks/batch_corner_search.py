import torch
from .attack import *
import common.torch


def onepixel_perturbation(attack, orig_x, pos, sigma):
    ''' returns a batch with the possible perturbations of the pixel in position pos '''

    if attack.type_attack == 'L0':
        if orig_x.shape[-1] == 3:
            batch_x = numpy.tile(orig_x, (8, 1, 1, 1))
            t = numpy.zeros([3])
            for counter in range(8):
                t2 = counter + 0
                for c in range(3):
                    t[c] = t2 % 2
                    t2 = (t2 - t[c]) / 2
                batch_x[counter, pos[0], pos[1]] = t.astype(numpy.float32)
        elif orig_x.shape[-1] == 1:
            batch_x = numpy.tile(orig_x, (2, 1, 1, 1))
            batch_x[0, pos[0], pos[1], 0] = 0.0
            batch_x[1, pos[0], pos[1], 0] = 1.0

    elif attack.type_attack == 'L0+Linf':
        if orig_x.shape[-1] == 3:
            batch_x = numpy.tile(orig_x, (8, 1, 1, 1))
            t = numpy.zeros([3])
            for counter in range(8):
                t2 = counter + 0
                for c in range(3):
                    t3 = t2 % 2
                    t[c] = (t3 * 2.0 - 1.0) * attack.epsilon
                    t2 = (t2 - t3) / 2
                batch_x[counter, pos[0], pos[1]] = numpy.clip(t.astype(numpy.float32) + orig_x[pos[0], pos[1]], 0.0, 1.0)
        elif orig_x.shape[-1] == 1:
            batch_x = numpy.tile(orig_x, (2, 1, 1, 1))
            batch_x[0, pos[0], pos[1], 0] = numpy.clip(batch_x[0, pos[0], pos[1], 0] - attack.epsilon, 0.0, 1.0)
            batch_x[1, pos[0], pos[1], 0] = numpy.clip(batch_x[1, pos[0], pos[1], 0] + attack.epsilon, 0.0, 1.0)

    elif attack.type_attack == 'L0+sigma':
        batch_x = numpy.tile(orig_x, (2, 1, 1, 1))
        if orig_x.shape[-1] == 3:
            batch_x[0, pos[0], pos[1]] = numpy.clip(batch_x[0, pos[0], pos[1]] * (1.0 - attack.kappa * sigma[pos[0], pos[1]]), 0.0, 1.0)
            batch_x[1, pos[0], pos[1]] = numpy.clip(batch_x[0, pos[0], pos[1]] * (1.0 + attack.kappa * sigma[pos[0], pos[1]]), 0.0, 1.0)

        elif orig_x.shape[-1] == 1:
            batch_x[0, pos[0], pos[1]] = numpy.clip(batch_x[0, pos[0], pos[1]] - attack.kappa * sigma[pos[0], pos[1]], 0.0, 1.0)
            batch_x[1, pos[0], pos[1]] = numpy.clip(batch_x[0, pos[0], pos[1]] + attack.kappa * sigma[pos[0], pos[1]], 0.0, 1.0)

    else:
        raise ValueError('unknown attack')

    return batch_x


def onepixel_perturbation_image(attack, orig_x, sigma):
    ''' returns a batch with all the possible perturbations of the image orig_x '''

    n_channels = orig_x.shape[-1]
    assert n_channels in [1, 3]
    n_corners = 2 ** n_channels if attack.type_attack in ['L0', 'L0+Linf'] else 2

    batch_x = numpy.zeros([n_corners * orig_x.shape[0] * orig_x.shape[1], orig_x.shape[0], orig_x.shape[1], orig_x.shape[2]])
    for counter in range(orig_x.shape[0]):
        for counter2 in range(orig_x.shape[1]):
            batch_x[(counter * orig_x.shape[0] + counter2) * n_corners:(counter * orig_x.shape[1] + counter2) * n_corners + n_corners] = numpy.clip(
                onepixel_perturbation(attack, orig_x, [counter, counter2], sigma), 0.0, 1.0)

    return batch_x


def flat2square(attack, ind):
    ''' returns the position and the perturbation given the index of an image
        of the batch of all the possible perturbations '''

    if attack.type_attack in ['L0', 'L0+Linf']:
        if attack.shape_img[-1] == 3:
            new_pixel = ind % 8
            ind = (ind - new_pixel) // 8
            c = ind % attack.shape_img[1]
            r = (ind - c) // attack.shape_img[1]
            t = numpy.zeros([ind.shape[0], 3])
            for counter in range(3):
                t[:, counter] = new_pixel % 2
                new_pixel = (new_pixel - t[:, counter]) / 2
        elif attack.shape_img[-1] == 1:
            t = ind % 2
            ind = (ind - t) // 2
            c = ind % attack.shape_img[1]
            r = (ind - c) // attack.shape_img[1]

    elif attack.type_attack == 'L0+sigma':
        t = ind % 2
        c = ((ind - t) // 2) % attack.shape_img[1]
        r = ((ind - t) // 2 - c) // attack.shape_img[1]

    return r, c, t


def npixels_perturbation(attack, orig_x, ind, k, sigma):
    ''' creates n_iter images which differ from orig_x in at most k pixels '''

    # sampling the n_iter k-pixels perturbations
    ind2 = numpy.random.randint(0, attack.n_max ** 2, (attack.n_iter, k))
    ind2 = attack.n_max - numpy.floor(ind2 ** 0.5).astype(int) - 1

    # creating the n_iter k-pixels perturbed images
    batch_x = numpy.tile(orig_x, (attack.n_iter, 1, 1, 1))
    if attack.type_attack == 'L0':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            batch_x[counter, p11, p12] = d1 + 0 if attack.shape_img[-1] == 3 else numpy.expand_dims(d1 + 0, 1)

    elif attack.type_attack == 'L0+Linf':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            d1 = d1 + 0 if attack.shape_img[-1] == 3 else numpy.expand_dims(d1 + 0, 1)
            batch_x[counter, p11, p12] = numpy.clip(batch_x[counter, p11, p12] + (2.0 * d1 - 1.0) * attack.epsilon, 0.0, 1.0)

    elif attack.type_attack == 'L0+sigma':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            d1 = numpy.expand_dims(d1, 1)
            if attack.shape_img[-1] == 3:
                batch_x[counter, p11, p12] = numpy.clip(
                    batch_x[counter, p11, p12] - attack.kappa * sigma[p11, p12] * (1 - d1) + attack.kappa * sigma[p11, p12] * d1, 0.0, 1.0)
            elif attack.shape_img[-1] == 1:
                batch_x[counter, p11, p12] = numpy.clip(
                    batch_x[counter, p11, p12] - attack.kappa * sigma[p11, p12] * (1 - d1) + attack.kappa * sigma[p11, p12] * d1, 0.0, 1.0)

    return batch_x


def sigma_map(x):
    ''' creates the sigma-map for the batch x '''

    sh = [4]
    sh.extend(x.shape)
    t = numpy.zeros(sh)
    t[0, :, :-1] = x[:, 1:]
    t[0, :, -1] = x[:, -1]
    t[1, :, 1:] = x[:, :-1]
    t[1, :, 0] = x[:, 0]
    t[2, :, :, :-1] = x[:, :, 1:]
    t[2, :, :, -1] = x[:, :, -1]
    t[3, :, :, 1:] = x[:, :, :-1]
    t[3, :, :, 0] = x[:, :, 0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = numpy.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = numpy.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = numpy.minimum(sd1, sd2)
    sd = numpy.sqrt(sd)

    return sd


class CSattack():
    def __init__(self, model, args):
        self.model = model
        self.type_attack = args['type_attack']  # 'L0', 'L0+Linf', 'L0+sigma'
        self.n_iter = args['n_iter']  # number of iterations (N_iter in the paper)
        self.n_max = args['n_max']  # the modifications for k-pixels perturbations are sampled among the best n_max (N in the paper)
        self.epsilon = args['epsilon']  # for L0+Linf, the bound on the Linf-norm of the perturbation
        self.kappa = args['kappa']  # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks
        self.k = args['sparsity']  # maximum number of pixels that can be modified (k_max in the paper)
        self.size_incr = args['size_incr']  # size of progressive increment of sparsity levels to check

    def forward(self, x_nat, y_nat):
        batch_images = common.torch.as_variable(x_nat.astype(numpy.float32), common.torch.is_cuda(self.model))
        batch_images = batch_images.permute(0, 3, 1, 2)

        logits = self.model.forward(batch_images)

        logits = logits.cpu().detach().numpy()
        predictions = numpy.argmax(logits, axis=1)
        correct_predictions = numpy.equal(predictions, y_nat)

        return correct_predictions, logits

    def perturb(self, x_nat, y_nat):
        adv = numpy.copy(x_nat)
        fl_success = numpy.ones([x_nat.shape[0]])
        self.shape_img = x_nat.shape[1:]
        self.sigma = sigma_map(x_nat)
        self.n_classes = 10
        self.n_corners = 2 ** self.shape_img[2] if self.type_attack in ['L0', 'L0+Linf'] else 2
        corr_pred, _ = self.forward(x_nat, y_nat)
        bs = self.shape_img[0] * self.shape_img[1]

        for c in range(x_nat.shape[0]):
            if corr_pred[c]:
                sigma = numpy.copy(self.sigma[c])
                batch_x = onepixel_perturbation_image(self, x_nat[c], sigma)
                batch_y = numpy.squeeze(y_nat[c])
                logit_2 = numpy.zeros([batch_x.shape[0], self.n_classes])
                found = False

                # checks one-pixels modifications
                for counter in range(self.n_corners):
                    pred, logit_2[counter * bs:(counter + 1) * bs] = self.forward(batch_x[counter * bs:(counter + 1) * bs], numpy.tile(batch_y, (bs)))
                    if not pred.all() and not found:
                        ind_adv = numpy.where(pred.astype(int) == 0)
                        adv[c] = batch_x[counter * bs + ind_adv[0][0]]
                        found = True
                        print('Point {} - adversarial example found changing 1 pixel'.format(c))

                # creates the orderings
                t1 = numpy.copy(logit_2[:, batch_y])
                logit_2[:, batch_y] = -1000.0 * numpy.ones(numpy.shape(logit_2[:, batch_y]))
                t2 = numpy.amax(logit_2, axis=1)
                t3 = t1 - t2
                logit_3 = numpy.tile(numpy.expand_dims(t1, axis=1), (1, self.n_classes)) - logit_2
                logit_3[:, batch_y] = t3
                ind = numpy.argsort(logit_3, axis=0)

                # checks multiple-pixels modifications
                for n3 in range(2, self.k, self.size_incr):
                    if not found:
                        for c2 in range(self.n_classes):
                            if not found:
                                ind_cl = numpy.copy(ind[:, c2])

                                batch_x = npixels_perturbation(self, x_nat[c], ind_cl, n3, sigma)
                                pred, _ =  self.forward(batch_x, numpy.tile(batch_y, (batch_x.shape[0])))

                                if numpy.sum(pred.astype(numpy.int32)) < self.n_iter and not found:
                                    found = True
                                    ind_adv = numpy.where(pred.astype(int) == 0)
                                    adv[c] = batch_x[ind_adv[0][0]]
                                    print('Point {} - adversarial example found changing {} pixels'.format(c, numpy.sum(
                                        numpy.amax(numpy.abs(adv[c] - x_nat[c]) > 1e-10, axis=-1), axis=(0, 1))))

                if not found:
                    fl_success[c] = 0
                    print('Point {} - adversarial example not found'.format(c))

            else:
                print('Point {} - misclassified'.format(c))

        pixels_changed = numpy.sum(numpy.amax(numpy.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1, 2))
        print('Pixels changed: ', pixels_changed)
        # print('attack successful: ', fl_success)
        # print('attack successful: {:.2f}%'.format((1.0 - numpy.mean(fl_success))*100.0))
        corr_pred, _ = self.forward(adv, y_nat)
        print('Robust accuracy at {} pixels: {:.2f}%'.format(self.k, numpy.sum(corr_pred) / x_nat.shape[0] * 100.0))
        print('Maximum perturbation size: {:.5f}'.format(numpy.amax(numpy.abs(adv - x_nat))))

        return adv, pixels_changed, fl_success


class BatchCornerSearch(Attack):
    """
    Corner Attack from https://github.com/fra31/sparse-imperceivable-attacks/blob/master/cornersearch_attacks.py.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchCornerSearch, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.epsilon = None
        """ (float) Epsilon. """

        self.sigma = False
        """ (bool) Sigma variant. """

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

        super(BatchCornerSearch, self).run(model, images, objective, writer, prefix)

        attack = CSattack(model, {
            'n_iter': self.max_iterations,
            'type_attack': 'L0+sigma' if self.sigma else 'L0',
            'n_max': 100,
            'epsilon': 0,
            'kappa': 0.4,
            'sparsity': self.epsilon,
            'size_incr': 1,
        })

        x_nat = images.detach().cpu().numpy()
        x_nat = numpy.transpose(x_nat, (0, 2, 3, 1))
        y_nat = objective.true_classes.detach().cpu().numpy()
        adversarial_examples, pixel_changes, successes = attack.perturb(x_nat, y_nat)

        adversarial_examples = numpy.transpose(adversarial_examples, (0, 3, 1, 2))
        x_nat = numpy.transpose(x_nat, (0, 3, 1, 2))

        return adversarial_examples - x_nat, successes