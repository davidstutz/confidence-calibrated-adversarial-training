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


class TestDistalAttacksAgainstNormalMNISTMLP(unittest.TestCase):
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

            optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(cls.model, cls.trainloader, cls.testloader, optimizer, scheduler, augmentation=augmentation, writer=writer,
                                                  cuda=cls.cuda)
            for e in range(10):
                trainer.step(e)
                print(e)

            common.state.State.checkpoint(cls.getModelFile(), cls.model, optimizer, scheduler, e)

        cls.model.eval()
        probabilities = common.test.test(cls.model, cls.testloader, cuda=cls.cuda)
        eval = common.eval.CleanEvaluation(probabilities, cls.testloader.dataset.labels, validation=0)
        assert 0.05 > eval.test_error(), '0.05 !> %g' % eval.test_error()
        assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet()
        cls.testset = common.datasets.MNISTTestSet()
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[100, 100, 100], action=torch.nn.Sigmoid)

    def ROCAUC(self, images, perturbations, labels):
        adversarialloader = torch.utils.data.DataLoader(common.datasets.AdversarialDataset(images, perturbations, labels), batch_size=100)
        testloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=100)

        clean_probabilities = common.test.test(self.model, testloader, cuda=self.cuda)
        adversarial_probabilities = numpy.array([common.test.test(self.model, adversarialloader, cuda=self.cuda)])

        eval = common.eval.DistalEvaluation(clean_probabilities, adversarial_probabilities, validation=0)
        return eval.receiver_operating_characteristic_auc()

    def runTestAttackPerformance(self, attack, attempts=5):
        for b, (images, labels) in enumerate(self.testloader):
            break

        roc_auc = 0
        images = common.torch.as_variable(numpy.random.uniform(0, 1, size=images.size()).astype(numpy.float32), self.cuda).permute(0, 3, 1, 2)

        for t in range(attempts):
            objective = attacks.objectives.UntargetedF0Objective(loss=common.torch.max_p_loss)
            perturbations, errors = attack.run(self.model, images, objective)

            perturbations = numpy.array([numpy.transpose(perturbations, (0, 2, 3, 1))])
            roc_auc += self.ROCAUC(numpy.transpose(images.cpu().numpy(), (0, 2, 3, 1)), perturbations, labels.cpu().numpy())

        roc_auc /= attempts
        return roc_auc

    def testBatchOptimSGDUnnormalized(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 100
        momentum = 0
        attack = attacks.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 20
        attack.c = 0
        attack.normalized = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.8, roc_auc)

    def testBatchOptimSGDNormalized(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 1
        momentum = 0
        attack = attacks.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 10
        attack.c = 0
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.7, roc_auc)

    def testBatchOptimSGDNormalizedSmooth(self):
        epsilon = 0.3
        optimizer = torch.optim.SGD
        lr = 1
        momentum = 0
        attack = attacks.BatchOptim(optimizer, lr=lr, momentum=momentum)
        attack.max_iterations = 10
        attack.c = 0
        attack.normalized = True
        attack.initialization = attacks.initializations.SequentialInitializations([attacks.initializations.LInfUniformInitialization(epsilon), attacks.initializations.SmoothInitialization()])
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.7, roc_auc)

    def testBatchGradientDescentUnnormalized(self):
        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.lr_factor = 1
        attack.c = 0
        attack.normalized = False
        attack.backtrack = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.8, roc_auc)

    def testBatchGradientDescentNormalized(self):
        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.lr_factor = 1
        attack.c = 0
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.7, roc_auc)

    def testBatchGradientDescentNormalizedBacktrack(self):
        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.lr_factor = 1
        attack.c = 0
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.7, roc_auc)

    def testBatchGradientDescentNormalizedBacktrackSmooth(self):
        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.lr_factor = 1
        attack.c = 0
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.SequentialInitializations([attacks.initializations.LInfUniformInitialization(epsilon), attacks.initializations.SmoothInitialization()])
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        roc_auc = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(0.7, roc_auc)


class TestDistalAttacksAgainstNormalMNISTLeNet(TestDistalAttacksAgainstNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_lenet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.LeNet(10, [1, 28, 28], channels=64)


class TestDistalAttacksAgainstNormalMNISTResNet(TestDistalAttacksAgainstNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1])


class TestDistalAttacksAgainstNormalSVHNesNet(TestDistalAttacksAgainstNormalMNISTMLP):
    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.SVHNTrainSet()
        cls.testset = common.datasets.SVHNTestSet()
        cls.adversarialset = common.datasets.SVHNTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'svhn_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [3, 32, 32], blocks=[2, 2, 2])


if __name__ == '__main__':
    unittest.main()
