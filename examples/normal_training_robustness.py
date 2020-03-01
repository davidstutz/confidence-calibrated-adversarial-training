import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.experiments
import common.utils
import common.eval
import common.paths
import common.imgaug
import common.datasets
import numpy
import attacks
from common.log import log
import models
from imgaug import augmenters as iaa
import torch


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


class Main:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        self.trainset = None
        """ (common.datasets.CleanDataset) Training dataset. """

        self.testset = None
        """ (common.datasets.CleanDataset) Test dataset. """

        self.adversarialset = None
        """ (common.datasets.CleanDataset) Dataset to attack. """

        self.trainloader = None
        """ (torch.utils.data.DataLoader) Training loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.adversarialloader = None
        """ (torch.utils.data.DataLoader) Loader to attack. """

        self.epsilon = 0
        """ (float) Epsilon for L_inf attacks. """

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Trains and attacks a normal model on the chosen dataset.')
        parser.add_argument('dataset', type=str, help='cifar10 | svhn | mnist')
        parser.add_argument('--directory', type=str, default='./checkpoints/')
        parser.add_argument('--tensorboard', action='store_true', help='use tensorboard for monitoring training')
        parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True, help='do not use cuda')

        return parser

    def setup(self):
        """
        Set dataloaders.
        """

        if self.args.dataset == 'cifar10':
            self.trainset = common.datasets.Cifar10TrainSet()
            self.testset = common.datasets.Cifar10TestSet()
            self.adversarialset = common.datasets.Cifar10TestSet(indices=range(100))
            self.epsilon = 0.03
        elif self.args.dataset == 'svhn':
            self.trainset = common.datasets.SVHNTrainSet()
            self.testset = common.datasets.SVHNTestSet()
            self.adversarialset = common.datasets.SVHNTestSet(indices=range(100))
            self.epsilon = 0.03
        elif self.args.dataset == 'mnist':
            self.trainset = common.datasets.MNISTTrainSet()
            self.testset = common.datasets.MNISTTestSet()
            self.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
            self.epsilon = 0.3
        else:
            assert False

        batch_size = 100
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.adversarialloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=batch_size, shuffle=False, num_workers=0)

        common.utils.makedir(self.args.directory)

    def train(self):
        """
        Training configuration.
        """

        def get_augmentation(crop=True, flip=True):
            augmenters = []
            if crop:
                augmenters.append(iaa.CropAndPad(
                    px=((0, 4), (0, 4), (0, 4), (0, 4)),
                    pad_mode='constant',
                    pad_cval=(0, 0),
                ))
            if flip:
                augmenters.append(iaa.Fliplr(0.5))

            return iaa.Sequential(augmenters)

        writer = common.summary.SummaryPickleWriter('%s/logs/' % self.args.directory, max_queue=100)
        if self.args.tensorboard:
            writer = torch.utils.tensorboard.SummaryWriter('%s/logs/' % self.args.directory, max_queue=100)

        crop = False
        flip = False
        if self.args.dataset == 'svhn':
            crop = True
        elif self.args.dataset == 'cifar10':
            crop = True
            flip = True

        epochs = 200
        snapshot = 10

        model_file = '%s/classifier.pth.tar' % self.args.directory
        incomplete_model_file = find_incomplete_state_file(model_file)
        load_file = model_file
        if incomplete_model_file is not None:
            load_file = incomplete_model_file

        start_epoch = 0
        if os.path.exists(load_file):
            state = common.state.State.load(load_file)
            self.model = state.model
            start_epoch = state.epoch + 1
            log('loaded %s' % load_file)
        else:
            self.model = models.ResNet(10, [self.trainset.images.shape[3], self.trainset.images.shape[1], self.trainset.images.shape[2]],
                                       blocks=[3, 3, 3])
        if self.args.cuda:
            self.model = self.model.cuda()

        augmentation = get_augmentation(crop=crop, flip=flip)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.075, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainloader),
                                                           gamma=0.97)
        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, optimizer, scheduler,
                                              augmentation=augmentation, writer=writer, cuda=self.args.cuda)

        self.model.train()
        for epoch in range(start_epoch, epochs):
            trainer.step(epoch)
            writer.flush()

            snapshot_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch)
            common.state.State.checkpoint(snapshot_model_file, self.model, optimizer, scheduler, epoch)

            previous_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch - 1)
            if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
                os.unlink(previous_model_file)

        previous_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch - 1)
        if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
            os.unlink(previous_model_file)

        common.state.State.checkpoint(model_file, self.model, optimizer, scheduler, epoch)

    def get_attack(self):
        """
        Get attacks to test.
        """

        pgd = attacks.BatchGradientDescent()
        pgd.max_iterations = 40
        pgd.base_lr = 0.005
        pgd.momentum = 0.9
        pgd.c = 0
        pgd.lr_factor = 1.5
        pgd.normalized = True
        pgd.backtrack = True
        pgd.initialization = attacks.initializations.LInfUniformNormInitialization(self.epsilon)
        pgd.projection = attacks.projections.SequentialProjections([
            attacks.projections.LInfProjection(self.epsilon),
            attacks.projections.BoxProjection()
        ])
        pgd.norm = attacks.norms.LInfNorm()
        objective = attacks.objectives.UntargetedF0Objective()

        return pgd, objective

    def evaluate(self):
        """
        Evaluate.
        """

        self.model.eval()
        clean_probabilities = common.test.test(self.model, self.testloader, cuda=self.args.cuda)

        attack, objective = self.get_attack()
        adversarial_perturbations, adversarial_probabilities, adversarial_errors = common.test.attack(self.model, self.adversarialloader,
                                                                  attack, objective, attempts=5, cuda=self.args.cuda)

        eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities,
                                                             self.testset.labels, validation=0.9, errors=adversarial_errors)
        log('test error in %%: %g' % (eval.test_error() * 100))
        log('test error @99%%tpr in %%: %g' % (eval.test_error_at_99tpr() * 100))
        log('robust test error in %%: %g' % (eval.robust_test_error()*100))
        log('robust test error @99%%tpr in %%: %g' % (eval.robust_test_error_at_99tpr()*100))

    def main(self):
        """
        Main.
        """

        self.setup()

        model_file = '%s/classifier.pth.tar' % self.args.directory
        if not os.path.exists(model_file):
            self.train()
        else:
            state = common.state.State.load(model_file)
            self.model = state.model
            if self.args.cuda:
                self.model = self.model.cuda()
        self.evaluate()


if __name__ == '__main__':
    program = Main()
    program.main()