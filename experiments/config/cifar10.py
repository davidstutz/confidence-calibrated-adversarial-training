"""
Configuration for experiments on Cifar10.
"""

import common.datasets
import experiments.config.helper as helper
helper.guard()


helper.batch_size = 100
helper.lr = 0.075
helper.momentum = 0.9
helper.lr_decay = 0.95
helper.weight_decay = 0.001
helper.epochs = 200
helper.snapshot = 10
helper.attempts = 1
helper.epsilon = 0.03
helper.l2_epsilon = 0.5
helper.l1_epsilon = 7.85 # Tramer paper
helper.l0_epsilon = 10
helper.max_iterations = 20
helper.base_lr = 0.1
helper.population = 2
helper.augmentation_crop = True
helper.augmentation_flip = True
helper.augmentation_contrast = True
helper.augmentation_add = False
helper.augmentation_saturation = False
helper.augmentation_value = False
helper.base_directory = 'Cifar10'

helper.trainset = common.datasets.Cifar10TrainSet()
helper.testset = common.datasets.Cifar10TestSet()
helper.adversarialset = common.datasets.Cifar10TestSet(indices=range(1000))
helper.randomset = common.datasets.RandomTestSet(1000, [32, 32, 3])


from experiments.config.common import *