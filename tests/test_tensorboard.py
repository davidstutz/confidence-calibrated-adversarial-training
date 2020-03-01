import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.imgaug
import common.eval
import attacks
import torch
import torch.utils.data
from imgaug import augmenters as iaa
import torch.utils.tensorboard
import shutil


class TestTensorboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_dir = './logs/'
        cls.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=100, shuffle=True, num_workers=4)
        cls.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=100, num_workers=4)

    def setUp(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def testNormalTraining(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)

        cuda = True
        if cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        writer = torch.utils.tensorboard.SummaryWriter('./logs/')
        augmentation = None

        trainer = common.train.NormalTraining(model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=cuda)
        trainer.summary_gradients = True

        epochs = 25
        trainer.test(-1)
        for e in range(epochs):
            trainer.step(e)
            writer.flush()
            print(e)

    def testAdversarialTraining(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)

        cuda = True
        if cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        writer = torch.utils.tensorboard.SummaryWriter('./logs/')
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=augmentation, writer=writer, cuda=cuda)
        trainer.summary_gradients = True

        epochs = 10
        trainer.test(-1)
        for e in range(epochs):
            trainer.step(e)
            writer.flush()
            print(e)

    def testConfidenceCalibratedAdversarialTraining(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)

        cuda = True
        if cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        writer = torch.utils.tensorboard.SummaryWriter('./logs/')
        augmentation = None

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        loss = common.torch.cross_entropy_divergence
        transition = common.torch.linear_transition

        trainer = common.train.ConfidenceCalibratedAdversarialTraining(model, self.trainset, self.testset, optimizer, scheduler, attack, objective, loss, transition, fraction=0.5, augmentation=augmentation, writer=writer, cuda=cuda)
        trainer.summary_gradients = True

        epochs = 10
        trainer.test(-1)
        for e in range(epochs):
            trainer.step(e)
            writer.flush()
            print(e)

    def testAddScalars(self):
        writer = torch.utils.tensorboard.SummaryWriter('./logs/')
        r = 5
        for i in range(100):
            writer.add_scalars('run_14h', {'xsinx': i * numpy.sin(i / r),
                                           'xcosx': i * numpy.cos(i / r),
                                           'tanx': numpy.tan(i / r)}, i)
        writer.close()

    def testBatchGradientDescentNormalizedBacktrack(self):
        cuda = True
        model_file = 'mnist_lenet.pth.tar'
        if os.path.exists(model_file):
            state = common.state.State.load(model_file)
            model = state.model

            if cuda:
                model = model.cuda()
        else:
            model = models.LeNet(10, [1, 28, 28], channels=12)
            if cuda:
                model = model.cuda()

            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=cuda)
            for e in range(10):
                trainer.step(e)
                print(e)

            common.state.State.checkpoint(model_file, model, optimizer, scheduler, e)

        model.eval()
        epsilon = 0.3
        writer = torch.utils.tensorboard.SummaryWriter('./logs/')

        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        for b, (images, labels) in enumerate(self.testset):
            break

        images = common.torch.as_variable(images, cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, cuda)

        objective = attacks.objectives.UntargetedF0Objective()
        objective.set(labels)
        attack.run(model, images, objective, writer=writer)
        writer.flush()

    #def tearDown(self):
    #    if os.path.exists(self.log_dir):
    #        shutil.rmtree(self.log_dir)


if __name__ == '__main__':
    unittest.main()
