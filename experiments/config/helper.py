"""
Helper module to allow to specialize hyper-parameters for eachd ataset individually.
"""

batch_size = None
lr = None
momentum = None
lr_decay = None
weight_decay = None
epochs = None
snapshot = None
attempts = None
epsilon = None
l2_epsilon = None
l1_epsilon = None
l0_epsilon = None
max_iterations = None
base_lr = None
augmentation_crop = None
augmentation_flip = None
augmentation_contrast = None
augmentation_add = None
augmentation_saturation = None
augmentation_value = None
base_directory = None

trainset = None
testset = None
adversarialset = None
randomset = None


def guard():
    """
    Guard to check that no variables have been defined yet.
    """

    for key, value in globals().items():
        if not callable(value) and not key.endswith('__'):
            assert value is None, '%s=%r is None' % (key, value)


def check():
    """
    Guard to check whether all parameters above have been defined.
    """

    for key, value in globals().items():
        if not callable(value) and not key.endswith('__'):
            assert value is not None, '%s=%r is None' % (key, value)