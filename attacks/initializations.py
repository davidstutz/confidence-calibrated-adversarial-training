import common.torch
import torch
import numpy
import math
import scipy.ndimage
import random
import common.numpy


class Initialization:
    """
    Abstract initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()


class SequentialInitializations(Initialization):
    """
    Combination of multiple initializers.
    """

    def __init__(self, initializations):
        """
        Constructor.

        :param initializations: list of initializations
        :type initializations: [Initializations]
        """

        assert isinstance(initializations, list)
        assert len(initializations) > 0
        for initialization in initializations:
            assert isinstance(initialization, Initialization)

        self.initializations = initializations
        """ ([Initializations]) Initializations. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        for initialization in self.initializations:
            initialization(images, perturbations)


class RandomInitializations(Initialization):
    """
    Combination of multiple initializers.
    """

    def __init__(self, initializations):
        """
        Constructor.

        :param initializations: list of initializations
        :type initializations: [Initializations]
        """

        assert isinstance(initializations, list)
        assert len(initializations) > 0
        for initialization in initializations:
            assert isinstance(initialization, Initialization)

        self.initializations = initializations
        """ ([Initializations]) Initializations. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        random.choice(self.initializations)(images, perturbations)


class FixedInitialization(Initialization):
    """
    Fixed initialization.
    """

    def __init__(self, perturbations):
        """
        Constructor.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        assert isinstance(perturbations, torch.autograd.Variable)

        self.perturbations = perturbations
        """ (torch.autograd.Variable) Perturbations. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = self.perturbations.data


class ZeroInitialization(Initialization):
    """
    Zero initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data.zero_()


class UniformInitialization(Initialization):
    """
    Zero initialization.
    """

    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :type min_bound: float
        :param max_bound: maximum bound
        :type max_bound: float
        """

        self.min_bound = min_bound
        """ (float) Min. """

        self.max_bound = max_bound
        """ (float) Max. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data.uniform_(self.min_bound, self.max_bound)


class SmoothInitialization(Initialization):
    """
    Gaussian smoothing as initialization; can be used after any random initialization; does not enforce any cosntraints.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        sigma = numpy.random.uniform(1, 2)
        gamma = numpy.random.uniform(5, 30)
        gaussian_smoothing = common.torch.GaussianLayer(sigma=sigma, channels=perturbations.size()[1])
        if common.torch.is_cuda(perturbations):
            gaussian_smoothing = gaussian_smoothing.cuda()
        perturbations.data = 1 / (1 + torch.exp(-gamma * (gaussian_smoothing.forward(perturbations) - 0.5)))


class GaussianInitialization(Initialization):
    """
    Initialization using random noise; does not enforce any constraints, projections should be used instead.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        is_cuda = common.torch.is_cuda(perturbations)
        D = perturbations.size(1) * perturbations.size(2) * perturbations.size(3)
        perturbations.data = torch.from_numpy(numpy.random.normal(loc=0, scale=self.epsilon / (2 * math.log(D)), size=list(perturbations.size())).astype(numpy.float32))
        if is_cuda:
            perturbations.data = perturbations.data.cuda()


class L2UniformVolumeInitialization(Initialization):
    """
    Uniform initialization, wrt. volume, in L_2 ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_ball(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


class L2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


class LInfUniformInitialization(Initialization):
    """
    Standard L_inf initialization as by Madry et al.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(numpy.random.uniform(-self.epsilon, self.epsilon, size=perturbations.size()).astype(numpy.float32))


class LInfUniformVolumeInitialization(Initialization):
    """
    Uniform initialization, wrt. volume, in L_inf ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_ball(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))


class LInfUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))


class L1UniformVolumeInitialization(Initialization):
    """
    Uniform initialization, wrt. volume, in L_1 ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_ball(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=1).reshape(perturbations.size()).astype(numpy.float32))


class L1UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_1 ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=1).reshape(perturbations.size()).astype(numpy.float32))


class L0UniformNormInitialization(Initialization):
    """
    Uniform initialization on L_1 sphere.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        random = numpy.random.uniform(-1, 1, size=perturbations.size())
        random *= numpy.random.binomial(1, 0.66*self.epsilon/numpy.prod(numpy.array(perturbations.size())[1:]), size=perturbations.size())
        perturbations.data = torch.from_numpy(random.astype(numpy.float32))
