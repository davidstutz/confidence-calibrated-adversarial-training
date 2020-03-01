import os
import io
import sys
import numpy
import torch
import torch.utils.data
import datetime
import common.paths
import common.utils
import common.train
import common.test
import common.state
from common.log import log
import attacks
from imgaug import augmenters as iaa


def find_incomplete_state_file(model_file):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])


def find_incomplete_state_files(model_file):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = sorted(epochs)

            return [os.path.join(base_directory, file_name + '.%d' % epoch) for epoch in epochs]


class NormalTrainingConfig:
    """
    Configuration for normal training.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.cuda = True
        self.augmentation = None
        self.loss = None
        self.summary_gradients = None
        self.trainloader = None
        self.testloader = None
        self.epochs = None
        self.snapshot = None

        # Writer depends on log directory
        self.get_writer = None
        # Optimizer is based on parameters
        self.get_optimizer = None
        # Scheduler is based on optimizer
        self.get_scheduler = None
        # Model is based on data resolution
        self.get_model = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert self.augmentation is None or isinstance(self.augmentation, iaa.meta.Augmenter)
        assert isinstance(self.trainloader, torch.utils.data.DataLoader)
        assert len(self.trainloader) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert self.epochs > 0
        assert self.snapshot is None or self.snapshot > 0
        assert callable(self.get_optimizer)
        assert callable(self.get_scheduler)
        assert callable(self.get_model)
        assert callable(self.get_writer)


class AdversarialTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialTrainingConfig, self).__init__()

        # Fixed parameters
        self.attack = None
        self.objective = None
        self.fraction = None

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialTrainingConfig, self).validate()

        assert isinstance(self.attack, attacks.Attack)
        assert isinstance(self.objective, attacks.objectives.Objective)
        assert self.fraction > 0 and self.fraction <= 1


class ConfidenceCalibratedAdversarialTrainingConfig(AdversarialTrainingConfig):
    """
    Configuration for confidence calibrated training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(ConfidenceCalibratedAdversarialTrainingConfig, self).__init__()

        # Fixed parameters
        self.loss = None
        self.transition = None

    def validate(self):
        """
        Check validity.
        """

        super(ConfidenceCalibratedAdversarialTrainingConfig, self).validate()

        assert callable(self.loss)
        assert callable(self.transition)
        assert self.fraction > 0 and self.fraction < 1


class NormalTrainingInterface:
    """
    Interface for normal training for expeirments.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, NormalTrainingConfig)
        config.validate()

        self.config = config
        """ (NormalTrainingConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.epochs = None
        """ (int) Epochs. """

        self.epoch = None
        """ (int) Start epoch. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.augmentation = None
        """ (None or iaa.meta.Augmenter) Data augmentation. """

        self.trainloader = None
        """ (torch.utils.data.DataLoader) Train loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.model = None
        """ (torch.nn.Module) Model. """

        self.optimizer = None
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = None
        """ (torch.optim.lr_scheduler.LRScheduler) Scheduler. """

    def setup(self):
        """
        Setup.
        """

        dt = datetime.datetime.now()
        self.log_dir = common.paths.log_file(self.config.directory, 'logs/%s' % dt.strftime('%d%m%y%H%M%S'))
        self.model_file = common.paths.experiment_file(self.config.directory, 'classifier', common.paths.STATE_EXT)

        self.cuda = self.config.cuda
        self.epochs = self.config.epochs

        self.epoch = 0
        state = None

        self.writer = self.config.get_writer(self.log_dir)
        self.augmentation = self.config.augmentation
        self.trainloader = self.config.trainloader
        self.testloader = self.config.testloader

        N_class = numpy.max(self.trainloader.dataset.labels) + 1
        resolution = [
            self.trainloader.dataset.images.shape[3],
            self.trainloader.dataset.images.shape[1],
            self.trainloader.dataset.images.shape[2],
        ]

        incomplete_model_file = find_incomplete_state_file(self.model_file)
        load_file = self.model_file
        if incomplete_model_file is not None:
            load_file = incomplete_model_file

        if os.path.exists(load_file):
            state = common.state.State.load(load_file)
            self.model = state.model
            self.epoch = state.epoch + 1
            log('loaded %s' % load_file)
        else:
            self.model = self.config.get_model(N_class, resolution)

        if self.cuda:
            self.model = self.model.cuda()

        self.optimizer = self.config.get_optimizer(self.model)
        if state is not None:
            self.optimizer.load_state_dict(state.optimizer)

        self.scheduler = self.config.get_scheduler(self.optimizer)
        if state is not None:
            self.scheduler.load_state_dict(state.scheduler)

    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler, augmentation=self.augmentation, writer=self.writer, cuda=self.cuda)

        if self.config.loss is not None:
            trainer.loss = self.config.loss
        if self.config.summary_gradients is not None:
            trainer.summary_gradients = self.config.summary_gradients

        return trainer

    def main(self):
        """
        Main.
        """

        self.setup()
        trainer = self.trainer()

        trainer.test(self.epoch - 1)
        e = self.epochs - 1
        for e in range(self.epoch, self.epochs):
            trainer.step(e)
            self.writer.flush()

            model_file = '%s.%d' % (self.model_file, e)
            common.state.State.checkpoint(model_file, self.model, self.optimizer, self.scheduler, e)

            previous_model_file = '%s.%d' % (self.model_file, e - 1)
            if os.path.exists(previous_model_file) and (self.config.snapshot is None or (e - 1)%self.config.snapshot > 0):
                os.unlink(previous_model_file)

        previous_model_file = '%s.%d' % (self.model_file, e - 1)
        if os.path.exists(previous_model_file) and (self.config.snapshot is None or (e - 1) % self.config.snapshot > 0):
            os.unlink(previous_model_file)

        common.state.State.checkpoint(self.model_file, self.model, self.optimizer, self.scheduler, e)


class AdversarialTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, AdversarialTrainingConfig)

        super(AdversarialTrainingInterface, self).__init__(config)

    def trainer(self):
        """
        Trainer.
        """

        return common.train.AdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler, self.config.attack, self.config.objective, self.config.fraction, augmentation=self.augmentation, writer=self.writer, cuda=self.cuda)


class ConfidenceCalibratedAdversarialTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, ConfidenceCalibratedAdversarialTrainingConfig)

        super(ConfidenceCalibratedAdversarialTrainingInterface, self).__init__(config)

    def trainer(self):
        """
        Trainer.
        """

        return common.train.ConfidenceCalibratedAdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler, self.config.attack, self.config.objective, self.config.loss, self.config.transition, self.config.fraction, augmentation=self.augmentation, writer=self.writer, cuda=self.cuda)


class AttackConfig:
    """
    Configuration for attacks.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.testloader = None
        self.attack = None
        self.objective = None
        self.attempts = None
        self.snapshot = None

        # Depends on directory
        self.get_writer = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert isinstance(self.attack, attacks.Attack), self.attack
        assert isinstance(self.objective, attacks.objectives.Objective)
        assert callable(self.get_writer)


class AttackInterface:
    """
    Regular attack interface.
    """

    def __init__(self, target_config, attack_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackConfig)

        target_config.validate()
        attack_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.perturbations_file = None
        """ (str) Perturbations file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.model = None
        """ (torch.nn.Module) Model. """

    def main(self):
        """
        Main.
        """

        self.log_dir = common.paths.log_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.perturbations_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.attack_config.directory), 'perturbations', common.paths.HDF5_EXT)
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        if self.attack_config.snapshot is not None:
            self.log_dir = common.paths.log_dir('%s/%s_%d' % (self.target_config.directory, self.attack_config.directory, self.attack_config.snapshot))
            self.perturbations_file = common.paths.experiment_file('%s/%s_%d' % (self.target_config.directory, self.attack_config.directory, self.attack_config.snapshot), 'perturbations', common.paths.HDF5_EXT)
            self.model_file += '.%d' % self.attack_config.snapshot

        assert os.path.exists(self.model_file), 'file %s not found' % self.model_file

        attempts = 0
        samples = 0
        if os.path.exists(self.perturbations_file):
            errors = common.utils.read_hdf5(self.perturbations_file, key='errors')
            attempts = errors.shape[0]
            samples = errors.shape[1]

        if not os.path.exists(self.perturbations_file) \
                or attempts < self.attack_config.attempts \
                or samples < len(self.attack_config.testloader.dataset):

            self.cuda = self.target_config.cuda
            if callable(self.attack_config.get_writer):
                self.writer = common.utils.partial(self.attack_config.get_writer, self.log_dir)
            else:
                self.writer = self.attack_config.get_writer

            state = common.state.State.load(self.model_file)
            self.model = state.model

            if self.cuda:
                self.model = self.model.cuda()

            self.model.eval()
            perturbations, probabilities, errors = common.test.attack(self.model, self.attack_config.testloader, self.attack_config.attack, self.attack_config.objective, attempts=self.attack_config.attempts, writer=self.writer, cuda=self.cuda)

            common.utils.write_hdf5(self.perturbations_file, [
                perturbations,
                probabilities,
                errors,
            ], [
                'perturbations',
                'probabilities',
                'errors',
            ])
