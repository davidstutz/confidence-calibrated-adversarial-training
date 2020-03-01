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
import common.eval
import attacks
import torch
import torch.utils.data


class TestAttacksAdversarialMNISTMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 100
        cls.cuda = True
        cls.setDatasets()

        if os.path.exists(cls.getModelFile()):
            state = common.state.State.load(cls.getModelFile())
            cls.model = state.model

            if cls.cuda:
                cls.model = cls.model.cuda()
        else:
            cls.model = cls.getModel()
            if cls.cuda:
                cls.model = cls.model.cuda()
            print(cls.model)

            optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1, momentum=0.9)
            scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(cls.trainloader))
            writer = common.summary.SummaryWriter()
            augmentation = None

            attack = attacks.batch_reference_pgd.BatchReferencePGD()
            attack.max_iterations = 5
            attack.epsilon = 0.3
            attack.base_lr = 0.1
            attack.norm = attacks.norms.LInfNorm()

            objective = attacks.objectives.UntargetedF0Objective()

            trainer = common.train.AdversarialTraining(cls.model, cls.trainloader, cls.testloader, optimizer, scheduler, attack=attack, objective=objective, augmentation=augmentation, writer=writer, cuda=cls.cuda)
            for e in range(10):
                trainer.step(e)

            common.state.State.checkpoint(cls.getModelFile(), cls.model, optimizer, scheduler, e)

            cls.model.eval()
            probabilities = common.test.test(cls.model, cls.testloader, cuda=cls.cuda)
            eval = common.eval.CleanEvaluation(probabilities, cls.testloader.dataset.labels, validation=0)
            assert 0.075 > eval.test_error(), '0.05 !> %g' % eval.test_error()
            assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        cls.model.eval()

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet()
        cls.testset = common.datasets.MNISTTestSet()
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'adversarial_mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[100, 100, 100], action=torch.nn.Sigmoid)

    def successRate(self, images, perturbations, labels):
        adversarialloader = torch.utils.data.DataLoader(common.datasets.AdversarialDataset(images, perturbations, labels), batch_size=100)
        testloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=100, shuffle=False)
        self.assertEqual(len(adversarialloader), len(testloader))

        # assumes one attempt only!
        clean_probabilities = common.test.test(self.model, testloader, cuda=self.cuda)
        adversarial_probabilities = numpy.array([common.test.test(self.model, adversarialloader, cuda=self.cuda)])

        eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, labels, validation=0)
        return eval.success_rate()

    def runTestAttackPerformance(self, attack, attempts=5, objective=attacks.objectives.UntargetedF0Objective()):
        for b, (images, labels) in enumerate(self.adversarialloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        success_rate = 0
        for t in range(attempts):
            objective.set(labels)
            perturbations, errors = attack.run(self.model, images, objective)

            perturbations = numpy.array([numpy.transpose(perturbations, (0, 2, 3, 1))])
            success_rate += self.successRate(numpy.transpose(images.cpu().numpy(), (0, 2, 3, 1)), perturbations, labels.cpu().numpy())

        success_rate /= attempts
        return success_rate

    def testCornerSearch(self):
        epsilon = 10
        attack = attacks.batch_corner_search.BatchCornerSearch()
        attack.max_iterations = 100
        attack.epsilon = epsilon
        attack.sigma = False

        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.45)

    def testBatchGradientDescentScaledBacktrackF7PL1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1000
        attack.base_lr = 100
        attack.momentum = 0.9
        attack.c = 0
        attack.lr_factor = 1.1
        attack.normalized = False
        attack.scaled = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF0L1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1000
        attack.base_lr = 75
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.1
        attack.normalized = True
        attack.scaled = False
        attack.backtrack = True
        attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF0Objective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7L1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1000
        attack.base_lr = 10
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.1
        attack.normalized = True
        attack.scaled = False
        attack.backtrack = True
        attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7Objective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7PL1(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1000
        attack.base_lr = 10
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.1
        attack.normalized = True
        attack.scaled = False
        attack.backtrack = True
        attack.initialization = attacks.initializations.UniformInitialization() # L1UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L1Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L1Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)

    def testBatchGradientDescentNormalizedBacktrackF7PL0(self):
        epsilon = 10
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 50
        attack.base_lr = 1
        attack.momentum = 0.9
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.L0UniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.L0Projection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.L0Norm()

        success_rate = self.runTestAttackPerformance(attack, attempts=5, objective=attacks.objectives.UntargetedF7PObjective())
        self.assertGreaterEqual(success_rate, 0.95)


class TestAttacksAdversarialMNISTLeNet(TestAttacksAdversarialMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'adversarial_mnist_lenet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.LeNet(10, [1, 28, 28])


class TestAttacksAdversarialMNISTResNet(TestAttacksAdversarialMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'adversarial_mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1])


if __name__ == '__main__':
    unittest.main()
