"""
Configuration for experiments on MNIST.
"""

import common.datasets
import experiments.config.helper as helper
helper.guard()


helper.batch_size = 100
helper.lr = 0.1
helper.momentum = 0.9
helper.lr_decay = 0.93
helper.weight_decay = 0.001
helper.epochs = 100
helper.snapshot = 5
helper.attempts = 1
helper.epsilon = 0.3
helper.l2_epsilon = 1.5
helper.l1_epsilon = 10
helper.l0_epsilon = 15
helper.max_iterations = 20
helper.base_lr = 0.1
helper.population = 2
helper.augmentation_crop = False
helper.augmentation_flip = False
helper.augmentation_contrast = False
helper.augmentation_add = False
helper.augmentation_saturation = False
helper.augmentation_value = False
helper.base_directory = 'MNIST'

helper.trainset = common.datasets.MNISTTrainSet()
helper.testset = common.datasets.MNISTTestSet()
helper.adversarialset = common.datasets.MNISTTestSet(indices=range(1000))
helper.randomset = common.datasets.RandomTestSet(1000, [28, 28, 1])


from experiments.config.common import *