"""
Configuration for models and attacks on all datasets.
"""

import attacks
import models
import numpy
import torch
import torch.utils.data
import torch.utils.tensorboard
import common.datasets
import common.experiments
import common.torch
import common.imgaug
import common.mask
from imgaug import augmenters as iaa
import experiments.config.helper as helper

helper.check()


def __get_augmentation(noise=False, crop=False, flip=False, contrast=False, add=False, saturation=False, value=False):
    """
    Get augmentation using imgaug.

    :param noise: whether to add additive noise
    :type noise: bool
    :param crop: whether to apply cropping
    :type crop: bool
    :param flip: whether to apply flipping
    :type flip: bool
    :param contrast: whether to apply contrast
    :type contrast: bool
    :param add: whether to apply add
    :type add: bool
    :param saturation: whether to apply saturation
    :type saturation: bool
    :param value: whether to apply value changes
    :type value: bool
    :return: agumenter
    :rtype: iaa.Sequential
    """

    augmenters = []
    if noise:
        std = 0.075
        augmenters.append(iaa.AdditiveGaussianNoise(scale=(0, std)))
    if crop:
        augmenters.append(iaa.CropAndPad(
            percent=(0, 0.2),
            pad_mode='edge',
        ))
    if flip:
        augmenters.append(iaa.Fliplr(0.5))
    if contrast:
        augmenters.append(iaa.ContrastNormalization((0.7, 1.3)))
    if add:
        augmenters.append(iaa.Add((-0.075, 0.075)))
    if saturation:
        augmenters.append(iaa.Sequential([
            common.imgaug.UInt8FromFloat32(),
            iaa.ChangeColorspace(from_colorspace='RGB', to_colorspace='HSV'),
            iaa.WithChannels(1, iaa.Add((-0.15, 0.15))),  # HSV
            iaa.ChangeColorspace(from_colorspace='HSV', to_colorspace='RGB'),
            common.imgaug.Float32FromUInt8(),
        ]))
    if value:
        augmenters.append(iaa.Sequential([
            common.imgaug.UInt8FromFloat32(),
            iaa.ChangeColorspace(from_colorspace='RGB', to_colorspace='HSV'),
            iaa.WithChannels(2, iaa.Add((-0.15, 0.15))),  # HSV
            iaa.ChangeColorspace(from_colorspace='HSV', to_colorspace='RGB'),
            common.imgaug.Float32FromUInt8(),
        ]))

    return iaa.Sequential([
        iaa.SomeOf(max(1, len(augmenters) // 2), augmenters),
        common.imgaug.Clip(),
    ])


batch_size = helper.batch_size
lr = helper.lr
momentum = helper.momentum
lr_decay = helper.lr_decay
weight_decay = helper.weight_decay
epochs = helper.epochs
snapshot = helper.snapshot
attempts = helper.attempts
epsilon = helper.epsilon
l2_epsilon = helper.l2_epsilon
l1_epsilon = helper.l1_epsilon
l0_epsilon = helper.l0_epsilon
max_iterations = helper.max_iterations
base_lr = helper.base_lr
population = helper.population
augmentation = __get_augmentation(noise=False, crop=helper.augmentation_crop, flip=helper.augmentation_flip, contrast=helper.augmentation_contrast,
                                  add=helper.augmentation_add, saturation=helper.augmentation_saturation, value=helper.augmentation_value)
base_directory = helper.base_directory


def __get_resnet20(N_class, resolution):
    """
    Get a ResNet20, the default model.

    :param N_class: number of classes.
    :type N_class: int
    :param resolution: resolution
    :type resolution (int, int, int)
    :return: model
    :rtype: models.Classifier
    """
    return models.ResNet(N_class, resolution, blocks=[3, 3, 3])


trainset = helper.testset
testset = helper.testset
adversarialset = helper.adversarialset
randomset = helper.randomset

trainloader = torch.utils.data.DataLoader(helper.trainset, batch_size=helper.batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(helper.testset, batch_size=helper.batch_size, shuffle=False, num_workers=0)
adversarialloader = torch.utils.data.DataLoader(helper.adversarialset, batch_size=helper.batch_size, shuffle=False, num_workers=0)
randomloader = torch.utils.data.DataLoader(helper.randomset, batch_size=helper.batch_size, shuffle=False, num_workers=0)


def __frames(base_lr=base_lr, max_iterations=max_iterations):
    """
    Adversarial frames attack.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :return: attack
    :rtype: attacks.Attack
    """

    img_shape = (trainset.images.shape[1], trainset.images.shape[2])
    mask_gen = common.mask.FrameGenerator(img_shape, int(0.1*img_shape[0]))
    attack = attacks.BatchFrames()
    attack.mask_gen = mask_gen
    attack.base_lr = base_lr
    attack.max_iterations = max_iterations
    return attack


def __reference_pgd(base_lr=base_lr, max_iterations=max_iterations, epsilon=epsilon):
    """
    L_inf reference PGD attack.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon for attack
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchReferencePGD()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.epsilon = epsilon
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __normalized_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf normalized PGD implementation.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __normalized_zero_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf normalized PGD implementation with zero initialization.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.ZeroInitialization()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __l0_normalized_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l0_epsilon, momentum=momentum):
    """
    L_0 normalized PGD implementation.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.L0UniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L0Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L0Norm()
    return attack


def __l0_normalized_zero_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l0_epsilon, momentum=momentum):
    """
    L_0 normalized PGD implementation with zero initialization.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.ZeroInitialization()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L0Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L0Norm()
    return attack


def __l1_normalized_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l1_epsilon, momentum=momentum):
    """
    L_1 normalized PGD implementation.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = False
    attack.scaled = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L1Norm()
    return attack


def __l1_normalized_zero_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l1_epsilon, momentum=momentum):
    """
    L_1 normalized PGD implementation with zero initialization.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = False
    attack.scaled = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.ZeroInitialization()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L1Norm()
    return attack


def __l2_normalized_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l2_epsilon, momentum=momentum):
    """
    L_2 normalized PGD implementation.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L2Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L2Norm()
    return attack


def __l2_normalized_zero_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=l2_epsilon, momentum=momentum):
    """
    L_2 normalized PGD implementation with zero initialization.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.ZeroInitialization()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L2Projection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.L2Norm()
    return attack


def __normalized_random_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf normalized PGD with random and zero initialization for training.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.RandomInitializations([
        attacks.initializations.GaussianInitialization(epsilon),
        attacks.initializations.LInfUniformNormInitialization(epsilon),
        attacks.initializations.ZeroInitialization(),
    ])
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __acet_testing_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf distal adversarial examples for testing, non-smoothed.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.RandomInitializations([
        attacks.initializations.LInfUniformNormInitialization(epsilon)
    ])
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __acet_testing_smoothed_pgd(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf distal adversarial examples for testing, smoothed.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchGradientDescent()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.momentum = momentum
    attack.c = 0
    attack.lr_factor = lr_factor
    attack.normalized = True
    attack.backtrack = backtrack
    attack.initialization = attacks.initializations.RandomInitializations([
        attacks.initializations.LInfUniformNormInitialization(epsilon),
        attacks.initializations.SmoothInitialization()
    ])
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __simple(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_inf simple black-box attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon for attack
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchSimple()
    attack.max_iterations = max_iterations
    attack.epsilon = epsilon
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    return attack


def __query_limited(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf query limited attack.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchQueryLimited()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.lr_factor = lr_factor
    attack.momentum = momentum
    attack.c = 0
    attack.population = 50
    attack.variance = 0.1
    attack.backtrack = backtrack
    attack.normalized = True
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __zero_query_limited(base_lr=base_lr, max_iterations=max_iterations, backtrack=False, lr_factor=1., epsilon=epsilon, momentum=momentum):
    """
    L_inf query limited attack with zero initialization.

    :param base_lr: learning rate
    :type base_lr: float
    :param max_iterations: iterations
    :type max_iterations: int
    :param backtrack: backtrack
    :type backtrack: bool
    :param lr_factor: learning rate factor
    :type lr_factor: float
    :param epsilon: epsilon for attack
    :type epsilon: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchQueryLimited()
    attack.max_iterations = max_iterations
    attack.base_lr = base_lr
    attack.lr_factor = lr_factor
    attack.momentum = momentum
    attack.c = 0
    attack.population = 50
    attack.variance = 0.1
    attack.backtrack = backtrack
    attack.normalized = True
    attack.initialization = None
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __cube2(max_iterations=max_iterations, probability=0.1, epsilon=epsilon):
    """
    L_inf cube attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param probability: probability of change
    :type probability: float
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchCube2()
    attack.max_iterations = max_iterations
    attack.probability = probability
    attack.epsilon = epsilon
    attack.norm = attacks.norms.LInfNorm()
    return attack


def __l2_cube2(max_iterations=max_iterations, probability=0.1, epsilon=epsilon):
    """
    L_2 cube attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param probability: probability of change
    :type probability: float
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchCube2()
    attack.max_iterations = max_iterations
    attack.probability = probability
    attack.epsilon = epsilon
    attack.norm = attacks.norms.L2Norm()
    return attack


def __l0_corner_search(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_0 corner search attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchCornerSearch()
    attack.max_iterations = max_iterations
    attack.epsilon = epsilon
    return attack


def __l0_sigma_corner_search(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_0 corner search attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchCornerSearch()
    attack.max_iterations = max_iterations
    attack.epsilon = epsilon
    attack.sigma = True
    return attack


def __geometry(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_inf geometry attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    database_size = max_iterations
    attack = attacks.BatchGeometry()
    attack.database = numpy.transpose(testset.images[:database_size], (0, 3, 1, 2))
    attack.norm = attacks.norms.LInfNorm()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
    return attack


def __l2_geometry(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_2 geometry attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    database_size = max_iterations
    attack = attacks.BatchGeometry()
    attack.database = numpy.transpose(testset.images[:database_size], (0, 3, 1, 2))
    attack.norm = attacks.norms.L2Norm()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L2Projection(epsilon), attacks.projections.BoxProjection()])
    return attack


def __l1_geometry(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_1 geometry attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    database_size = max_iterations
    attack = attacks.BatchGeometry()
    attack.database = numpy.transpose(testset.images[:database_size], (0, 3, 1, 2))
    attack.norm = attacks.norms.L1Norm()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
    return attack


def __l0_geometry(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_0 geometry attack.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    database_size = max_iterations
    attack = attacks.BatchGeometry()
    attack.database = numpy.transpose(testset.images[:database_size], (0, 3, 1, 2))
    attack.norm = attacks.norms.L0Norm()
    attack.projection = attacks.projections.SequentialProjections([attacks.projections.L0Projection(epsilon), attacks.projections.BoxProjection()])
    return attack


def __random(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_inf random sampling.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchRandom()
    attack.max_iterations = max_iterations
    attack.norm = attacks.norms.LInfNorm()
    attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
    return attack


def __l2_random(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_2 random sampling.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchRandom()
    attack.max_iterations = max_iterations
    attack.norm = attacks.norms.L2Norm()
    attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
    return attack


def __l1_random(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_1 random sampling.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchRandom()
    attack.max_iterations = max_iterations
    attack.norm = attacks.norms.L1Norm()
    attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
    return attack


def __l0_random(max_iterations=max_iterations, epsilon=epsilon):
    """
    L_0 random sampling.

    :param max_iterations: iterations
    :type max_iterations: int
    :param epsilon: epsilon
    :type epsilon: float
    :return: attack
    :rtype: attacks.Attack
    """

    attack = attacks.BatchRandom()
    attack.max_iterations = max_iterations
    attack.norm = attacks.norms.L0Norm()
    attack.initialization = attacks.initializations.L0UniformNormInitialization(epsilon)
    return attack


def __untargeted_f0(loss=common.torch.classification_loss):
    """
    Untargeted loss objective, default is cross-entropy.

    :param loss: loss to maximize
    :type loss: callable
    :return: objective
    :rtype: attacks.objectives.Objective
    """

    return attacks.objectives.UntargetedF0Objective(loss=loss)


# Untargeted F7P objective for CCAT.
__untargeted_f7p = attacks.objectives.UntargetedF7PObjective()


def __get_training_writer(log_dir, sub_dir=''):
    """
    Training is logged in Tensorboard.

    :param log_dir: log directory
    :type log_dir: str
    :param sub_dir: log subdirectory
    :type sub_dir: str
    :return: summary write
    :rtype: torch.utils.tensorboard.SummaryWriter
    """

    return torch.utils.tensorboard.SummaryWriter('%s/%s' % (log_dir, sub_dir), max_queue=100)


def __get_attack_writer(log_dir, sub_dir=''):
    """
    Writer for monitoring attacks.

    :param log_dir: log directory
    :type log_dir: str
    :param sub_dir: log subdirectory
    :type sub_dir: str
    :return: summary write
    :rtype: common.summary.SummaryPickleWriter
    """

    return common.summary.SummaryPickleWriter('%s/%s' % (log_dir, sub_dir), max_queue=100)


def __get_optimizer(model, lr=lr, momentum=momentum):
    """
    Get optimizer.

    :param model: model
    :type model: torch.nn.Module
    :param lr: learning rate
    :type lr: float
    :param momentum: momentum parameter
    :type momentum: float
    :return: optimizer
    :rtype: torch.optim.Optimizer
    """

    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def __get_scheduler(optimizer):
    """
    Scheduler.

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :return: scheduler
    :rtype: callable
    """

    return common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(trainloader), gamma=lr_decay)


def __normal_training(directory):
    """
    Normal training configuration.

    :param directory: directory for models
    :type directory: str
    """

    config = common.experiments.NormalTrainingConfig()
    config.directory = '%s/%s' % (base_directory, directory)
    config.cuda = True
    config.augmentation = augmentation
    config.trainloader = trainloader
    config.testloader = testloader
    config.epochs = epochs
    config.get_writer = __get_training_writer
    config.get_optimizer = __get_optimizer
    config.get_scheduler = __get_scheduler
    config.get_model = __get_resnet20
    config.snapshot = snapshot
    config.validate()

    assert directory not in globals().keys()
    globals()[directory] = config


def __adversarial_training(directory, attack, objective, fraction):
    """
    Adversarial training configuration.

    :param directory: directory for models
    :type directory: str
    :param attack: attack for adversarial training
    :type attack: attacks.Attack
    :param objective: objective for attack
    :type objective: attacks.objectives.Objective
    :param fraction: fraction of adversarial examples to train on
    :type fraction: float
    """

    config = common.experiments.AdversarialTrainingConfig()
    config.directory = '%s/%s' % (base_directory, directory)
    config.cuda = True
    config.augmentation = augmentation
    config.trainloader = trainloader
    config.testloader = testloader
    config.epochs = epochs
    config.get_writer = __get_training_writer
    config.get_optimizer = __get_optimizer
    config.get_scheduler = __get_scheduler
    config.get_model = __get_resnet20
    config.attack = attack
    config.objective = objective
    config.fraction = fraction
    config.snapshot = snapshot
    config.validate()

    assert directory not in globals().keys()
    globals()[directory] = config


def __confidence_calibrated_adversarial_training(directory, attack, objective, loss, transition, fraction):
    """
    CCAT configuration.

    :param directory: directory for models
    :type directory: str
    :param attack: attack for adversarial training
    :type attack: attacks.Attack
    :param objective: objective for attack
    :type objective: attacks.objectives.Objective
    :param loss: loss to use
    :type loss: callable
    :param transition: transition between one-hot and uniform distribution
    :type transition: callable
    :param fraction: fraction of adversarial examples to train on
    :type fraction: float
    """

    config = common.experiments.ConfidenceCalibratedAdversarialTrainingConfig()
    config.directory = '%s/%s' % (base_directory, directory)
    config.cuda = True
    config.augmentation = augmentation
    config.trainloader = trainloader
    config.testloader = testloader
    config.epochs = epochs
    config.get_writer = __get_training_writer
    config.get_optimizer = __get_optimizer
    config.get_scheduler = __get_scheduler
    config.get_model = __get_resnet20
    config.attack = attack
    config.objective = objective
    config.loss = loss
    config.transition = transition
    config.fraction = fraction
    config.snapshot = snapshot
    config.validate()

    assert directory not in globals().keys()
    globals()[directory] = config


# confidence-calibrated adversarial training
for base_lr_, base_lr_name_ in zip([0.1, 0.05, 0.01, 0.005, 0.001], ['_lr01', '_lr005', '_lr001', '', '_lr0001']):
    for gamma_ in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
        __confidence_calibrated_adversarial_training(
            directory='confidence_calibrated_adversarial_training%s_ce_f7p_i40_random_momentum_backtrack_power2_%d' % (base_lr_name_, gamma_),
            attack=__normalized_random_pgd(base_lr=base_lr_, max_iterations=2 * max_iterations, backtrack=True,
                                           lr_factor=1.5, epsilon=epsilon, momentum=0.9), objective=__untargeted_f7p,
            loss=common.torch.cross_entropy_divergence,
            transition=common.utils.partial(common.torch.power_transition, norm=attacks.norms.LInfNorm(),
                                            epsilon=epsilon,
                                            gamma=gamma_), fraction=0.5)

# normal training
__normal_training(directory='normal_training_check')

# MSD pre-trained models
__normal_training(directory='multi_steepest_descent')

# adversarial training
for base_lr_, base_lr_name_ in zip([0.1, 0.05, 0.01, 0.005], ['01', '005', '001', '0005']):
    __adversarial_training(directory='adversarial_training_lr%s_i40_half_momentum_backtrack_check' % base_lr_name_,
                           attack=__normalized_pgd(base_lr=base_lr_, max_iterations=2 * max_iterations, backtrack=True, lr_factor=1.5, epsilon=epsilon,
                                                   momentum=0.9),
                           objective=__untargeted_f0(), fraction=0.5)
    __adversarial_training(directory='adversarial_training_lr%s_f7p_i40_half_momentum_backtrack_check' % base_lr_name_,
                           attack=__normalized_pgd(base_lr=base_lr_, max_iterations=2 * max_iterations, backtrack=True, lr_factor=1.5, epsilon=epsilon,
                                                   momentum=0.9),
                           objective=__untargeted_f7p, fraction=0.5)


def __attack_config(directory, attack, objective, attempts=attempts, testloader=adversarialloader):
    """
    Attack configuration.

    :param directory: directory for perturbations
    :type directory: str
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: objective
    :type objective: attacks.objectives.Objective
    :param attempts: attempts
    :type attempts: int
    :param testloader: testloader
    :type testloader: torch.utils.data.DataLoader
    """

    config = common.experiments.AttackConfig()
    config.get_writer = __get_training_writer
    config.directory = directory
    config.attack = attack
    config.objective = objective
    config.testloader = testloader
    config.attempts = attempts

    assert directory not in globals().keys(), directory
    globals()[directory] = config


# setup L_inf cube attacks with various hyper-parameters
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [100, 250, 500]:
        for attack_ in [__cube2]:
            for probability_, probability_name_ in zip([0.01, 0.05, 0.1], ['001', '005', '01']):
                for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'],
                                                   [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
                    for attempts_name_, attempts_ in zip(['', '_10'], [1, 10]):
                        for run_ in ['', '_b', '_c', '_d', '_e']:
                            directory = '%s_%d_%s_%s%s%s%s' % (
                                attack_.__name__.replace('__', ''),
                                iterations_,
                                objective_name_,
                                probability_name_,
                                attempts_name_,
                                epsilon_name_,
                                run_
                            )
                            __attack_config(directory=directory,
                                            attack=attack_(max_iterations=iterations_ * max_iterations, probability=probability_, epsilon=epsilon_),
                                            objective=objective_, attempts=attempts_)

# setup L_2 cube attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [100, 250, 500]:
        for attack_ in [__l2_cube2]:
            for probability_, probability_name_ in zip([0.01, 0.05, 0.1], ['001', '005', '01']):
                for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3',  '_12e3', '_24e3'],
                                                   [l2_epsilon, 6 * l2_epsilon / 3., 9 * l2_epsilon / 3., 12 * l2_epsilon / 3., 24 * l2_epsilon / 3.]):
                    for attempts_name_, attempts_ in zip(['', '_10'], [1, 10]):
                        for run_ in ['', '_b', '_c', '_d', '_e']:
                            directory = '%s_%d_%s_%s%s%s%s' % (
                                attack_.__name__.replace('__', ''),
                                iterations_,
                                objective_name_,
                                probability_name_,
                                attempts_name_,
                                epsilon_name_,
                                run_
                            )
                            __attack_config(directory=directory,
                                            attack=attack_(max_iterations=iterations_ * max_iterations, probability=probability_, epsilon=epsilon_),
                                            objective=objective_, attempts=attempts_)

# setup L_0 corner search attacks
for objective_name_, objective_ in zip(['f0'], [__untargeted_f0()]):
    for iterations_ in [5]:
        for attack_ in [__l0_corner_search, __l0_sigma_corner_search]:
            for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3'],
                                               [l0_epsilon, 6 * l0_epsilon / 3., 9 * l0_epsilon / 3., 12 * l0_epsilon / 3.]):
                for run_ in ['', '_b', '_c', '_d', '_e']:
                    directory = '%s_%s_%d%s%s' % (
                        attack_.__name__.replace('__', ''),
                        objective_name_,
                        iterations_,
                        epsilon_name_,
                        run_,
                    )
                    __attack_config(directory=directory, attack=attack_(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                    objective=objective_, attempts=1)

# L_inf geometry attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [50, 100, 250, 500]:
        for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'],
                                           [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
            for run_ in ['', '_b', '_c', '_d', '_e']:
                directory = 'geometry_%s_%d%s%s' % (
                    objective_name_,
                    iterations_,
                    epsilon_name_,
                    run_,
                )
                __attack_config(directory=directory, attack=__geometry(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                objective=objective_, attempts=1)

# L_2 geometry attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [50, 100, 250, 500]:
        for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3', '_24e3'], [l2_epsilon, 6 * l2_epsilon / 3., 9 * l2_epsilon / 3., 12 * l2_epsilon / 3., 24 * l2_epsilon / 3.]):
            for run_ in ['', '_b', '_c', '_d', '_e']:
                directory = 'l2_geometry_%s_%d%s%s_check' % (
                    objective_name_,
                    iterations_,
                    epsilon_name_,
                    run_,
                )
                __attack_config(directory=directory, attack=__l2_geometry(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                objective=objective_, attempts=1)

# L_1 geometry attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [50, 100, 250, 500]:
        for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3', '_e12', '_e18', '_e24'], [l1_epsilon, 6 * l1_epsilon / 3., 9 * l1_epsilon / 3., 12 * l1_epsilon / 3., 12, 18, 24]):
            for run_ in ['', '_b', '_c', '_d', '_e']:
                directory = 'l1_geometry_%s_%d%s%s_check' % (
                    objective_name_,
                    iterations_,
                    epsilon_name_,
                    run_,
                )
                __attack_config(directory=directory, attack=__l1_geometry(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                objective=objective_, attempts=1)

# L_0 geometry attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [50, 100, 250, 500]:
        for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3'], [l0_epsilon, 6 * l0_epsilon / 3., 9 * l0_epsilon / 3., 12 * l0_epsilon / 3.]):
            for run_ in ['', '_b', '_c', '_d', '_e']:
                directory = 'l0_geometry_%s_%d%s%s_check' % (
                    objective_name_,
                    iterations_,
                    epsilon_name_,
                    run_,
                )
                __attack_config(directory=directory, attack=__l0_geometry(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                objective=objective_, attempts=1)

# L_inf simple attacks
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for iterations_ in [50, 100, 250, 500]:
        for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'], [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
            for attempts_name_, attempts_ in zip(['', '_10'], [1, 10]):
                for run_ in ['', '_b', '_c', '_d', '_e']:
                    directory = 'simple_%d_%s%s%s%s' % (
                        iterations_,
                        objective_name_,
                        attempts_name_,
                        epsilon_name_,
                        run_,
                    )
                    __attack_config(directory=directory, attack=__simple(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                    objective=objective_, attempts=attempts_)

# L_inf random attacks
for iterations_ in [50, 100, 250, 500]:
    for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'], [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
        for run_ in ['', '_b', '_c', '_d', '_e']:
            directory = 'random_%d%s%s' % (
                iterations_,
                epsilon_name_,
                run_
            )
            __attack_config(directory=directory, attack=__random(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                            objective=__untargeted_f7p, attempts=1)

# L_2 random attacks
for iterations_ in [50, 100, 250, 500]:
    for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3', '_24e3'], [l2_epsilon, 6 * l2_epsilon / 3., 9 * l2_epsilon / 3., 12 * l2_epsilon / 3., 24 * l2_epsilon / 3.]):
        for run_ in ['', '_b', '_c', '_d', '_e']:
            directory = 'l2_random_%d%s%s' % (
                iterations_,
                epsilon_name_,
                run_
            )
            __attack_config(directory=directory, attack=__l2_random(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                            objective=__untargeted_f7p, attempts=1)

# L_1 random attacks
for iterations_ in [50, 100, 250, 500]:
    for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3', '_e12', '_e18', '_e24'], [l1_epsilon, 6 * l1_epsilon / 3., 9 * l1_epsilon / 3., 12 * l1_epsilon / 3., 12, 18, 24]):
        for run_ in ['', '_b', '_c', '_d', '_e']:
            directory = 'l1_random_%d%s%s' % (
                iterations_,
                epsilon_name_,
                run_
            )
            __attack_config(directory=directory, attack=__l1_random(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                            objective=__untargeted_f7p, attempts=1)

# L_0 random attacks
for iterations_ in [50, 100, 250, 500]:
    for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3'], [l0_epsilon, 6 * l0_epsilon / 3., 9 * l0_epsilon / 3., 12 * l0_epsilon / 3.]):
        for run_ in ['', '_b', '_c', '_d', '_e']:
            directory = 'l0_random_%d%s%s' % (
                iterations_,
                epsilon_name_,
                run_
            )
            __attack_config(directory=directory, attack=__l0_random(max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                            objective=__untargeted_f7p, attempts=1)

# L_inf reference PGD
for lr_name_, lr_ in zip(['00001', '00005', '0001', '0005', '001', '005', '01'], [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]):
    for iterations_ in [1, 2, 5, 10, 50, 100]:
        for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
            for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'], [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
                for run_ in ['', '_b', '_c', '_d', '_e']:
                    directory = 'reference_pgd_%d_%s%s%s%s' % (
                        iterations_,
                        lr_name_,
                        attempts_name_,
                        epsilon_name_,
                        run_
                    )
                    __attack_config(directory=directory, attack=__reference_pgd(base_lr=lr_, max_iterations=iterations_ * max_iterations, epsilon=epsilon_),
                                    objective=__untargeted_f0(), attempts=attempts_)

# L_2 our PGD
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for lr_name_, lr_ in zip(['00001', '00005', '0001', '0005', '001', '005', '01', '05', '1', '5', '10'], [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]):
        for attack_ in [__l2_normalized_pgd, __l2_normalized_zero_pgd]:
            for momentum_name_, momentum_ in zip(['', '_momentum'], [0, 0.9]):
                for backtrack_name_, backtrack_ in zip(['', '_backtrack'], [False, True]):
                    for lr_factor_, iterations_ in zip([1.5, 1.3, 1.25, 1.1, 1.05, 1.025], [1, 5, 10, 50, 100, 200]):
                        for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
                            for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3', '_9e3', '_12e3'], [l2_epsilon, 4 * l2_epsilon / 3., 6 * l2_epsilon / 3., 9 * l2_epsilon / 3., 12 * l2_epsilon / 3.]):
                                for run_ in ['', '_b', '_c', '_d', '_e']:
                                    directory = '%s_%d_%s_%s%s%s%s%s%s' % (
                                        attack_.__name__.replace('__', ''),
                                        iterations_,
                                        objective_name_,
                                        lr_name_,
                                        momentum_name_,
                                        backtrack_name_,
                                        attempts_name_,
                                        epsilon_name_,
                                        run_
                                    )
                                    __attack_config(directory=directory,
                                                    attack=attack_(base_lr=lr_, max_iterations=iterations_ * max_iterations, backtrack=backtrack_,
                                                                   lr_factor=lr_factor_,
                                                                   epsilon=epsilon_, momentum=momentum_),
                                                    objective=objective_, attempts=attempts_)

# L_1 our PGD
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for lr_name_, lr_ in zip(['005', '01', '05', '1', '5', '10', '15', '20', '25', '50', '100'], [0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 50, 100]):
        for attack_ in [__l1_normalized_pgd, __l1_normalized_zero_pgd]:
            for momentum_name_, momentum_ in zip(['', '_momentum'], [0, 0.9]):
                for backtrack_name_, backtrack_ in zip(['', '_backtrack'], [False, True]):
                    for lr_factor_, iterations_ in zip([1.5, 1.3, 1.25, 1.1, 1.05, 1.025], [1, 5, 10, 50, 100, 200]):
                        for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
                            for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3', '_e12', '_e18', '_e24'], [l1_epsilon, 6 * l1_epsilon / 3., 9 * l1_epsilon / 3., 12 * l1_epsilon / 3., 12, 18, 24]):
                                for run_ in ['', '_b', '_c', '_d', '_e']:
                                    directory = '%s_%d_%s_%s%s%s%s%s%s' % (
                                        attack_.__name__.replace('__', ''),
                                        iterations_,
                                        objective_name_,
                                        lr_name_,
                                        momentum_name_,
                                        backtrack_name_,
                                        attempts_name_,
                                        epsilon_name_,
                                        run_
                                    )
                                    __attack_config(directory=directory,
                                                    attack=attack_(base_lr=lr_, max_iterations=iterations_ * max_iterations, backtrack=backtrack_,
                                                                   lr_factor=lr_factor_,
                                                                   epsilon=epsilon_, momentum=momentum_),
                                                    objective=objective_, attempts=attempts_)

# L_0 our PGD
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for lr_name_, lr_ in zip(['1', '5', '10', '25', '50', '100', '250', '500', '750', '1000'], [1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]):
        for attack_ in [__l0_normalized_pgd, __l0_normalized_zero_pgd]:
            for momentum_name_, momentum_ in zip(['', '_momentum'], [0, 0.9]):
                for backtrack_name_, backtrack_ in zip(['', '_backtrack'], [False, True]):
                        for lr_factor_, iterations_ in zip([1.5, 1.3, 1.25, 1.1, 1.05, 1.025], [1, 5, 10, 50, 100, 200]):
                            for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
                                for epsilon_name_, epsilon_ in zip(['', '_6e3', '_9e3', '_12e3'], [l0_epsilon, 6 * l0_epsilon / 3., 9 * l0_epsilon / 3., 12 * l0_epsilon / 3.]):
                                    for run_ in ['', '_b', '_c', '_d', '_e']:
                                        directory = '%s_%d_%s_%s%s%s%s%s%s' % (
                                            attack_.__name__.replace('__', ''),
                                            iterations_,
                                            objective_name_,
                                            lr_name_,
                                            momentum_name_,
                                            backtrack_name_,
                                            attempts_name_,
                                            epsilon_name_,
                                            run_
                                        )
                                        __attack_config(directory=directory,
                                                        attack=attack_(base_lr=lr_, max_iterations=iterations_ * max_iterations, backtrack=backtrack_,
                                                                       lr_factor=lr_factor_,
                                                                       epsilon=epsilon_, momentum=momentum_),
                                                        objective=objective_, attempts=attempts_)

# L_inf our PGD
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for lr_name_, lr_ in zip(['00001', '00005', '0001', '0005', '001', '005', '01'], [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]):
        for attack_ in [__normalized_pgd, __normalized_zero_pgd, __zero_query_limited, __query_limited]:
            for momentum_name_, momentum_ in zip(['', '_momentum'], [0, 0.9]):
                for backtrack_name_, backtrack_ in zip(['', '_backtrack'], [False, True]):
                    for lr_factor_, iterations_ in zip([1.5, 1.3, 1.25, 1.1, 1.05, 1.025], [1, 5, 10, 50, 100, 200]):
                        for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
                            for epsilon_name_, epsilon_ in zip(['', '_4e3', '_6e3'],
                                                               [epsilon, 4 * epsilon / 3., 6 * epsilon / 3.]):
                                for run_ in ['', '_b', '_c', '_d', '_e']:
                                    directory = '%s_%d_%s_%s%s%s%s%s%s' % (
                                        attack_.__name__.replace('__', ''),
                                        iterations_,
                                        objective_name_,
                                        lr_name_,
                                        momentum_name_,
                                        backtrack_name_,
                                        attempts_name_,
                                        epsilon_name_,
                                        run_
                                    )
                                    __attack_config(directory=directory,
                                                    attack=attack_(base_lr=lr_, max_iterations=iterations_ * max_iterations, backtrack=backtrack_,
                                                                   lr_factor=lr_factor_,
                                                                   epsilon=epsilon_, momentum=momentum_),
                                                    objective=objective_, attempts=attempts_)

# ACET attacks
for objective_name_, objective_ in zip(['f0_max_p', 'f0_max_log'], [__untargeted_f0(loss=common.torch.max_p_loss), __untargeted_f0(loss=common.torch.max_log_loss)]):
    for lr_name_, lr_ in zip(['00001', '00005', '0001', '0005', '001', '005', '01'], [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]):
        for attack_ in [__acet_testing_pgd, __acet_testing_smoothed_pgd]:
            for momentum_name_, momentum_ in zip(['', '_momentum'], [0, 0.9]):
                for backtrack_name_, backtrack_ in zip(['', '_backtrack'], [False, True]):
                    for lr_factor_, iterations_ in zip([1.5, 1.3, 1.25, 1.1, 1.05, 1.025], [1, 5, 10, 50, 100, 200]):
                        for attempts_name_, attempts_ in zip(['', '_5', '_10', '_25', '_50'], [1, 5, 10, 25, 50]):
                            for epsilon_name_, epsilon_ in zip(['', '_e03'], [epsilon, 0.3]):
                                for run_ in ['', '_b', '_c', '_d', '_e']:
                                    directory = '%s_%d_%s_%s%s%s%s%s%s' % (
                                        attack_.__name__.replace('__', ''),
                                        iterations_,
                                        objective_name_,
                                        lr_name_,
                                        momentum_name_,
                                        backtrack_name_,
                                        attempts_name_,
                                        epsilon_name_,
                                        run_,
                                    )
                                    __attack_config(directory=directory,
                                                    attack=attack_(base_lr=lr_, max_iterations=iterations_ * max_iterations, backtrack=backtrack_,
                                                                   lr_factor=lr_factor_,
                                                                   epsilon=epsilon_, momentum=momentum_),
                                                    objective=objective_, attempts=attempts_, testloader=randomloader)

# adversarial frames
for objective_name_, objective_ in zip(['f7p', 'f0'], [__untargeted_f7p, __untargeted_f0()]):
    for lr_name_, lr_ in zip(['0005', '001', '005', '01', '05'], [0.005, 0.01, 0.05, 0.1, 0.5]):
            for iterations_ in [1, 5, 10, 50, 100, 200]:
                for attempts_name_, attempts_ in zip(['', '_5', '_10'], [1, 5, 10]):
                    for run_ in ['', '_b', '_c', '_d', '_e']:
                        directory = 'frames_%d_%s_%s%s%s' % (
                            iterations_,
                            objective_name_,
                            lr_name_,
                            attempts_name_,
                            run_
                        )
                        __attack_config(directory=directory,
                                        attack=__frames(base_lr=lr_, max_iterations=iterations_ * max_iterations),
                                        objective=objective_, attempts=attempts_)

set_frames = [
    'frames_50_f7p_001_5',
    'frames_10_f0_005_10',
    'frames_10_f0_005_10_b',
    'frames_10_f0_005_10_c',
    'frames_10_f0_005_10_d',
    'frames_10_f0_005_10_e',
]
set_linf_acet = [
    #
    'acet_testing_pgd_50_f0_max_log_0001_momentum_backtrack_10',
    'acet_testing_smoothed_pgd_50_f0_max_log_0001_momentum_backtrack_10',
    #
    'acet_testing_pgd_50_f0_max_log_0001_momentum_backtrack_10_e03',
    'acet_testing_smoothed_pgd_50_f0_max_log_0001_momentum_backtrack_10_e03',
]
set_linf_white = [
    #
    'normalized_zero_pgd_50_f7p_0001_momentum_backtrack',
    'normalized_pgd_50_f7p_0001_momentum_backtrack_10',
    #
    'reference_pgd_10_005_10',
    'reference_pgd_10_005_10_b',
    'reference_pgd_10_005_10_c',
    'reference_pgd_10_005_10_d',
    'reference_pgd_10_005_10_e',
    'normalized_pgd_10_f0_005_momentum_backtrack_10',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_b',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_c',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_d',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_e',
]
set_linf_black = [
    'cube2_250_f7p_005',
    #
    'simple_50_f7p_10',
    #
    'geometry_f7p_50',
    #
    'zero_query_limited_50_f7p_0001_momentum_backtrack',
    'query_limited_50_f7p_0001_momentum_backtrack_10',
    #
    'random_50',
    'random_50_b',
    'random_50_c',
    'random_50_d',
    'random_50_e',
]
set_linf = set_linf_white + set_linf_black
set_linf_minus_backtrack_white = [
    #
    'normalized_zero_pgd_50_f7p_0001',
    'normalized_pgd_50_f7p_0001_10',
    #
    'reference_pgd_10_005_10',
    'reference_pgd_10_005_10_b',
    'reference_pgd_10_005_10_c',
    'reference_pgd_10_005_10_d',
    'reference_pgd_10_005_10_e',
    'normalized_pgd_10_f0_005_10',
    'normalized_pgd_10_f0_005_10_b',
    'normalized_pgd_10_f0_005_10_c',
    'normalized_pgd_10_f0_005_10_d',
    'normalized_pgd_10_f0_005_10_e',
]
set_linf_minus_backtrack_black = [
    'zero_query_limited_50_f7p_0001',
    'query_limited_50_f7p_0001_10',
]
set_linf_minus_backtrack = set_linf_minus_backtrack_white + set_linf_minus_backtrack_black
set_linf_epsilon_white = [
    #
    'normalized_zero_pgd_50_f7p_0001_momentum_backtrack_6e3',
    'normalized_pgd_50_f7p_0001_momentum_backtrack_10_6e3',
    #
    'reference_pgd_10_005_10_6e3',
    'reference_pgd_10_005_10_6e3_b',
    'reference_pgd_10_005_10_6e3_c',
    'reference_pgd_10_005_10_6e3_d',
    'reference_pgd_10_005_10_6e3_e',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_6e3',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_b',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_c',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_d',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_e',
]
set_linf_epsilon_black = [
    'cube2_250_f7p_005_6e3',
    #
    'simple_50_f7p_10_6e3',
    #
    'zero_query_limited_50_f7p_0001_momentum_backtrack_6e3',
    'query_limited_50_f7p_0001_momentum_backtrack_10_6e3',
    #
    'random_50_6e3',
    'random_50_6e3_b',
    'random_50_6e3_c',
    'random_50_6e3_d',
    'random_50_6e3_e',
]
set_linf_epsilon = set_linf_epsilon_white + set_linf_epsilon_black
set_linf_epsilon_white_mnist = [
    #
    'normalized_zero_pgd_50_f7p_0001_momentum_backtrack_4e3',
    'normalized_pgd_50_f7p_0001_momentum_backtrack_10_4e3',
    #
    'reference_pgd_10_005_10_4e3',
    'reference_pgd_10_005_10_4e3_b',
    'reference_pgd_10_005_10_4e3_c',
    'reference_pgd_10_005_10_4e3_d',
    'reference_pgd_10_005_10_4e3_e',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_4e3',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_4e3_b',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_4e3_c',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_4e3_d',
    'normalized_pgd_10_f0_005_momentum_backtrack_10_4e3_e',
]
set_linf_epsilon_black_mnist = [
    'cube2_250_f7p_005_4e3',
    #
    'simple_50_f7p_10_4e3',
    #
    'zero_query_limited_50_f7p_0001_momentum_backtrack_4e3',
    'query_limited_50_f7p_0001_momentum_backtrack_10_4e3',
    #
    'random_50_4e3',
    'random_50_4e3_b',
    'random_50_4e3_c',
    'random_50_4e3_d',
    'random_50_4e3_e',
]
set_linf_epsilon_mnist = set_linf_epsilon_white_mnist + set_linf_epsilon_black_mnist
set_l2_12e3_white = [
    #
    'l2_normalized_zero_pgd_50_f7p_0001_momentum_backtrack_12e3',
    'l2_normalized_pgd_50_f7p_0001_momentum_backtrack_5_12e3',
    #
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_12e3',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_12e3_b',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_12e3_c',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_12e3_d',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_12e3_e',
]
set_l2_12e3_black = [
    'l2_cube2_250_f7p_005_12e3',
    'l2_geometry_f7p_50_12e3_check',
    #
    'l2_random_50_12e3',
    'l2_random_50_12e3_b',
    'l2_random_50_12e3_c', #
    'l2_random_50_12e3_d',
    'l2_random_50_12e3_e',
]
set_l2_12e3 = set_l2_12e3_white + set_l2_12e3_black
set_l2_9e3_white = [
    #
    'l2_normalized_zero_pgd_50_f7p_0001_momentum_backtrack_9e3',
    'l2_normalized_pgd_50_f7p_0001_momentum_backtrack_5_9e3',
    #
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_9e3',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_9e3_b',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_9e3_c',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_9e3_d',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_9e3_e',
]
set_l2_9e3_black = [
    'l2_cube2_250_f7p_005_9e3',
    'l2_geometry_f7p_50_9e3_check',
    #
    'l2_random_50_9e3',
    'l2_random_50_9e3_b',
    'l2_random_50_9e3_c',
    'l2_random_50_9e3_d',
    'l2_random_50_9e3_e',
]
set_l2_9e3 = set_l2_9e3_white + set_l2_9e3_black
set_l2_6e3_white = [
    #
    'l2_normalized_zero_pgd_50_f7p_0001_momentum_backtrack_6e3',
    'l2_normalized_pgd_50_f7p_0001_momentum_backtrack_5_6e3',
    #
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_6e3',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_b',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_c',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_d',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_6e3_e',
]
set_l2_6e3_black = [
    'l2_cube2_250_f7p_005_6e3',
    'l2_geometry_f7p_50_6e3_check',
    #
    'l2_random_50_6e3',
    'l2_random_50_6e3_b',
    'l2_random_50_6e3_c',
    'l2_random_50_6e3_d',
    'l2_random_50_6e3_e',
]
set_l2_6e3 = set_l2_6e3_white + set_l2_6e3_black
set_l2_3e3_white = [
    #
    'l2_normalized_zero_pgd_50_f7p_0001_momentum_backtrack',
    'l2_normalized_pgd_50_f7p_0001_momentum_backtrack_5',
    #
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_b',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_c',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_d',
    'l2_normalized_pgd_10_f0_005_momentum_backtrack_10_e',
]

set_l2_3e3_black = [
    'l2_cube2_250_f7p_005',
    'l2_geometry_f7p_50_check',
    #
    'l2_random_50',
    'l2_random_50_b',
    'l2_random_50_c',
    'l2_random_50_d',
    'l2_random_50_e',
]
set_l2_3e3 = set_l2_3e3_white + set_l2_3e3_black
set_l1_white = [
    'l1_normalized_zero_pgd_50_f7p_05_momentum_backtrack',
    'l1_normalized_pgd_50_f7p_05_momentum_backtrack_5',
    #
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_b',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_c',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_d',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e',
]
set_l1_black = [
    'l1_geometry_f7p_50_check',
    'l1_random_50',
    'l1_random_50_b',
    'l1_random_50_c',
    'l1_random_50_d',
    'l1_random_50_e',
]
set_l1 = set_l1_white + set_l1_black
set_l1_12_white = [
    'l1_normalized_zero_pgd_50_f7p_05_momentum_backtrack_e12',
    'l1_normalized_pgd_50_f7p_05_momentum_backtrack_5_e12',
    #
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e12',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e12_b',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e12_c',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e12_d',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e12_e',
]
set_l1_12_black = [
    'l1_geometry_f7p_50_e12_check',
    'l1_random_50_e12',
    'l1_random_50_e12_b',
    'l1_random_50_e12_c',
    'l1_random_50_e12_d',
    'l1_random_50_e12_e',
]
set_l1_12 = set_l1_12_white + set_l1_12_black
set_l1_18_white = [
    'l1_normalized_zero_pgd_50_f7p_05_momentum_backtrack_e18',
    'l1_normalized_pgd_50_f7p_05_momentum_backtrack_5_e18',
    #
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e18',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e18_b',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e18_c',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e18_d',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e18_e',
]
set_l1_18_black = [
    'l1_geometry_f7p_50_e18_check',
    'l1_random_50_e18',
    'l1_random_50_e18_b',
    'l1_random_50_e18_c',
    'l1_random_50_e18_d',
    'l1_random_50_e18_e',
]
set_l1_18 = set_l1_18_white + set_l1_18_black
set_l1_24_white = [
    'l1_normalized_zero_pgd_50_f7p_05_momentum_backtrack_e24',
    'l1_normalized_pgd_50_f7p_05_momentum_backtrack_5_e24',
    #
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e24',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e24_b',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e24_c',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e24_d',
    'l1_normalized_pgd_10_f0_05_momentum_backtrack_10_e24_e',
]
set_l1_24_black = [
    'l1_geometry_f7p_50_e24_check',
    'l1_random_50_e24',
    'l1_random_50_e24_b',
    'l1_random_50_e24_c',
    'l1_random_50_e24_d',
    'l1_random_50_e24_e',
]
set_l1_24 = set_l1_24_white + set_l1_24_black
set_l0_white = [
    'l0_normalized_zero_pgd_50_f7p_250_momentum_backtrack',
    'l0_normalized_pgd_50_f7p_250_momentum_backtrack_5',
    #
    'l0_normalized_pgd_10_f0_250_momentum_backtrack_10',
    'l0_normalized_pgd_10_f0_250_momentum_backtrack_10_b',
    'l0_normalized_pgd_10_f0_250_momentum_backtrack_10_c',
    'l0_normalized_pgd_10_f0_250_momentum_backtrack_10_d',
    'l0_normalized_pgd_10_f0_250_momentum_backtrack_10_e',
]
set_l0_black = [
    'l0_corner_search_f0_5',
    'l0_sigma_corner_search_f0_5',
    'l0_geometry_f7p_50_check',
    #
    'l0_random_50',
    'l0_random_50_b',
    'l0_random_50_c',
    'l0_random_50_d',
    'l0_random_50_e',
]
set_l0 = set_l0_white + set_l0_black