import torch
from .attack import *
from .norms import *
import common.torch
import time


class Cube2Model:
    def __init__(self, model):
        self.model = model

    def fmargin(self, x, y):
        logits = self.model.forward(common.torch.as_variable(x.astype(numpy.float32), common.torch.is_cuda(self.model))).detach().cpu().numpy()
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        diff[numpy.arange(diff.shape[0]), numpy.argmax(y, axis=1)] = 10e12#numpy.inf  # to exclude zeros coming from f_correct - f_correct
        logits = diff.min(1, keepdims=True)

        return logits.flatten()


def p_selection(p_init, it):
    """ The schedule was adapted for mnist/cifar10, but not for imagenet"""
    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 10000:
        p = p_init / 16
    elif 10000 < it <= 15000:  # for robust mnist and cifar10
        p = p_init / 32
    elif 15000 < it <= 20000:  # for robust mnist and cifar10
        p = p_init / 64
    elif 20000 < it:  # for robust mnist and cifar10
        p = 0  # means that only one pixel will be taken
    else:
        p = p_init
    return p


def pseudo_gaussian_pert(s):
    delta = numpy.zeros([s, s])
    total_pert = 0
    s2 = s // 2 + 1
    # t = max(1, s//10)
    s3 = 1
    counter2 = [s2 - s3, s2 - s3]
    for counter in range(s3 - 1, s2):
        # total_pert += (2*counter + 1)**2
        delta[counter2[0]:counter2[0] + (2 * counter + 1), counter2[1]:counter2[1] + (2 * counter + 1)] += 1.0 / (counter - s3 + 2) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    # print(delta)
    # sys.exit('test')
    # delta += 1.0/s2**2
    delta /= numpy.sqrt(numpy.sum(delta ** 2, keepdims=True))

    return delta


def cube_linf_attack(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path):
    """ A simple, but efficient black-box attack that just adds random steps of values in {-2eps, 0, 2eps}
    (i.e., the considered points are always corners). Note that considering just {-eps, 0, eps} works terribly.
    The random change is added if the loss decreases for a particular point.
    The only disadvantage of this method is that it will never find decision regions inside the Linf-ball which
    do not intersect any corner. But tight LRTE suggests that this doesn't happen.
        `f` is any function that has f.fmargin() method that returns class scores.
        `eps` can be a scalar or a vector of size X.shape[0].
    """
    spatial = True
    tied_delta = True  # untied delta work better only on cifar10
    tied_colors = False
    numpy.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 if eps < 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    #x, y = x[corr_classified], y[corr_classified]

    # [c, 1, w] works best
    x_best = numpy.clip(x + numpy.random.choice([-eps, eps], size=[x.shape[0], c, 1, w]), min_val, max_val)
    margin_min = model.fmargin(x_best, y)
    n_queries = numpy.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(numpy.sqrt(p_init * n_features / c))
    metrics = numpy.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        #idx_to_fool = margin_min > 0.0
        idx_to_fool = numpy.array(list(range(x.shape[0])))
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]

        if spatial:
            p = p_selection(p_init, i_iter)
            s = max(int(round(numpy.sqrt(p * n_features / c))), 1)  # at least cx1x1 window is taken
            center_h = numpy.random.randint(0, h-s)
            center_w = numpy.random.randint(0, w-s)
            new_deltas = numpy.zeros(x_curr.shape[1:])
            size = [1, 1, 1] if tied_delta and tied_colors else [c, 1, 1] if tied_delta else [c, s, s]
            new_deltas[:, center_h:center_h+s, center_w:center_w+s] = numpy.random.choice([-2*eps, 2*eps], size=size)
            hps_str = 'p={} s={}->{}'.format(p_init, s_init, s)
        else:
            p = p_selection(p_init, i_iter)
            new_deltas = numpy.random.choice([-2*eps, 0, 2*eps], p=[p/2, 1-p, p/2], size=[1, *x_curr.shape[1:]])
            hps_str = 'p={}->{}'.format(p_init, p)

        x_new = x_best_curr + new_deltas
        x_new = numpy.clip(x_new, x_curr - eps, x_curr + eps)
        x_new = numpy.clip(x_new, min_val, max_val)

        margin = model.fmargin(x_new, y_curr)

        idx_improved = margin < margin_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = numpy.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        # print(f_x_vals[:8].flatten())
        # print(f_x_vals_min[:8].flatten())
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq = numpy.mean(n_queries), numpy.mean(n_queries[margin_min <= 0]), numpy.median(n_queries)
        time_total = time.time() - time_start
        print('{}: marign_min={:.2} acc={:.2%} acc_corr={:.2%} avg#q={:.2f} avg#q_ae={:.2f} med#q={:.1f} ({}, n_ex={}, eps={:.3f}, {:.2f}s)'.
           format(i_iter+1, numpy.mean(margin_min), acc, acc_corr, mean_nq, mean_nq_ae, median_nq, hps_str, x.shape[0], eps, time_total))
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
        #if (i_iter <= 500 and i_iter % 20) or (i_iter > 100 and i_iter % 200) or i_iter + 1 == n_iters or acc == 0:
        #    numpy.save(metrics_path, metrics)
        #if acc == 0:
        #    break

    return n_queries, x_best


def cube_l2_attack(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path):
    """ A simple, but efficient black-box attack that just adds random steps of values in {-2eps, 0, 2eps}
    (i.e., the considered points are always corners). Note that considering just {-eps, 0, eps} works terribly.
    The random change is added if the loss decreases for a particular point.
    The only disadvantage of this method is that it will never find decision regions inside the Linf-ball which
    do not intersect any corner. But tight LRTE suggests that this doesn't happen.
        `f` is any function that has f.fmargin() method that returns class scores.
        `eps` can be a scalar or a vector of size X.shape[0].
    """
    spatial = True
    tied_delta = True  # untied delta work better only on cifar10
    tied_colors = False
    numpy.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    #x, y = x[corr_classified], y[corr_classified]

    # random norma initialization
    delta_init = numpy.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    # delta_init = numpy.sign(delta_init)*numpy.abs(delta_init)**0.5
    # delta_init = numpy.ones(x.shape)*numpy.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])*eps/numpy.sqrt(n_features)
    x_best = numpy.clip(x + delta_init / numpy.sqrt(numpy.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)

    '''s = 3
    x_best = numpy.copy(x)
    for counter in range(5000):
      center_h = numpy.random.randint(0, h-s)
      center_w = numpy.random.randint(0, w-s)
      x_best[:, :, center_h:center_h+s, center_w:center_w+s] += pseudo_gaussian_pert(s).reshape([1, 1, s, s])*numpy.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])*eps/100
      #x_best[:, :, center_h:center_h+s, center_w:center_w+s] *= numpy.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])*eps
      #x_best[:, :, center_h:center_h+s, center_w:center_w+s] += numpy.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])*eps'''

    '''x_best = numpy.copy(x)
    ind = numpy.arange(0, h)
    ind = ind[(ind % 2 < 1)]
    x_best[:, :, ind] += numpy.random.choice([-1, 1], size=[x.shape[0], c, ind.shape[0], 1])*eps/ind.shape[0]
    x_best[:, :, :, ind] += numpy.random.choice([-1, 1], size=[x.shape[0], c, 1, ind.shape[0]])*eps/ind.shape[0]'''

    # delta_init = numpy.clip(x_best, 0, 1) - x
    # x_best = numpy.clip(x + delta_init/numpy.sqrt(numpy.sum(delta_init**2, axis=(1,2,3), keepdims=True))*eps, 0, 1)

    delta_init = x_best - x
    norms_init = numpy.sqrt(numpy.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True))
    print('Initial perturbations - min: {:.2f}, max {:.2f}'.format(numpy.amin(norms_init), numpy.amax(norms_init)))

    margin_min = model.fmargin(x_best, y)
    n_queries = numpy.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(numpy.sqrt(p_init * n_features / c))
    metrics = numpy.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        #idx_to_fool = (margin_min > 0.0)
        idx_to_fool = numpy.array(list(range(x.shape[0])))
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]

        if spatial:
            p = p_selection(p_init, i_iter)
            # p = p_selection_2(p_init, i_iter, n_iters)
            s = max(int(round(numpy.sqrt(p * n_features / c))), 3)
            if s % 2 == 0: s += 1
            center_h = numpy.random.randint(0, h - s)
            center_w = numpy.random.randint(0, w - s)
            new_deltas_mask = numpy.zeros(x_curr.shape)
            # size = [1, 1, 1] if tied_delta and tied_colors else [c, 1, 1] if tied_delta else [c, s, s]
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

            curr_norms_window = numpy.sqrt(numpy.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
            curr_norms_image = numpy.sqrt(numpy.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

            # new_deltas = numpy.random.randn(x_curr.shape[0], x.shape[1], x.shape[2], x.shape[3]) * new_deltas_mask
            # new_norms_window = numpy.sqrt(numpy.sum(new_deltas**2, axis=(2,3), keepdims=True))
            # new_deltas = new_deltas/new_norms_window*(eps - curr_norms_image)/3

            # new_deltas = new_deltas_mask * (numpy.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1]) * ((eps - curr_norms_image)/3 + curr_norms_window)/s)

            # old_deltas = (x_best_curr - x_curr)*new_deltas_mask            

            new_deltas = numpy.zeros(x_curr.shape)
            new_deltas[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0
            new_deltas[:, :, center_h:center_h + s, center_w:center_w + s] *= pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas[:, :, center_h:center_h + s, center_w:center_w + s] *= numpy.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
            new_deltas[:, :, center_h:center_h + s, center_w:center_w + s] *= ((eps - curr_norms_image) / 3 + curr_norms_window)

            # new_deltas = (new_deltas + old_deltas/curr_norms_window)/2
            # new_deltas = new_deltas/numpy.sqrt(numpy.sum(new_deltas**2, axis=(1,2,3), keepdims=True))*((eps - curr_norms_image)/3 + curr_norms_window)

            '''new_deltas = numpy.zeros(x_curr.shape)
            d = (x_best_curr - x_curr)*new_deltas_mask
            new_deltas = -d/numpy.sqrt(numpy.sum(d**2, axis=(1,2,3), keepdims=True))*(eps - curr_norms_image + curr_norms_window)'''

            hps_str = 'p={} s={}->{}'.format(p_init, s_init, s)
        else:
            p = p_selection(p_init, i_iter)
            new_deltas = numpy.random.choice([-2 * eps, 0, 2 * eps], p=[p / 2, 1 - p, p / 2], size=[1, *x_curr.shape[1:]])
            hps_str = 'p={}->{}'.format(p_init, p)

        x_new = x_best_curr * (1.0 - new_deltas_mask) + new_deltas + x_curr * new_deltas_mask
        # x_new = numpy.clip(x_new, x_curr - eps, x_curr + eps)
        x_new = numpy.clip(x_new, min_val, max_val)
        curr_norms_image = numpy.sqrt(numpy.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        margin = model.fmargin(x_new, y_curr)

        idx_improved = margin < margin_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = numpy.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        # print(f_x_vals[:8].flatten())
        # print(f_x_vals_min[:8].flatten())
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        print(margin_min.mean())

        mean_nq, mean_nq_ae, median_nq, median_nq_ae = numpy.mean(n_queries), numpy.mean(n_queries[margin_min <= 0]), numpy.median(n_queries), numpy.median(
            n_queries[margin_min <= 0])

        time_total = time.time() - time_start
        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.1f}s, loss={:.3f}, max_pert={:.1f}'.
                  format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total, numpy.mean(margin_min),
                         numpy.amax(curr_norms_image)))
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
        #if (i_iter <= 500 and i_iter % 20) or (i_iter > 100 and i_iter % 200) or i_iter + 1 == n_iters or acc == 0:
        #    numpy.save(metrics_path, metrics)
        #if acc == 0:
        #    break

    return n_queries, x_best


class BatchCube2(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchCube2, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.probability = None
        """ (float) Probability. """

        self.epsilon = None
        """ (float) Epsilon. """

        self.projection = None
        """ (attacks.Projection) Projection. """
        
        self.norm = None
        """ (attacks.Norm or str) Norm. """

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
        assert self.norm is not None

        super(BatchCube2, self).run(model, images, objective, writer, prefix)

        x = images.detach().cpu().numpy()
        y = common.torch.one_hot(objective.true_classes, model._N_output).detach().cpu().numpy().astype(int)

        f = Cube2Model(model)

        attack = None
        if (isinstance(self.norm, str) and self.norm.lower() == 'l2') \
            or isinstance(self.norm, L2Norm):
            attack = cube_l2_attack
        elif (isinstance(self.norm, str) and self.norm.lower() == 'linf') \
            or isinstance(self.norm, LInfNorm):
            attack = cube_linf_attack
        assert attack is not None

        queries, adversarial_images = attack(f, x, y, None, self.epsilon, self.max_iterations, self.probability, None)

        logits = model.forward(common.torch.as_variable(adversarial_images.astype(numpy.float32), common.torch.is_cuda(model)))
        errors = objective(logits, None).detach().cpu().numpy()

        perturbations = adversarial_images - x
        return perturbations.astype(numpy.float32), errors.astype(numpy.float32)